package lp

import (
	"math"
	"time"
)

// Dual simplex re-optimization on a bounded-variable revised simplex basis.
//
// Called when the current basis is dual-feasible but possibly primal-
// infeasible — typically because a variable bound was tightened (e.g. by
// MIP branching) and a basic variable now sits outside its updated bound.
// The algorithm picks the most-violated basic row, computes the pivot row
// ρ = e_r^T B^{-1} via BTRAN, runs the dual ratio test over non-basics,
// and pivots until primal feasibility is restored. If at some iteration
// no eligible entering variable exists, the LP is reported infeasible
// (dual-unbounded ⇔ primal-infeasible).
//
// Sign convention mirrors runSimplex: internal MINIMIZE; reduced cost
// d_j ≥ 0 for non-basics at lower and d_j ≤ 0 for non-basics at upper.

// dualStallWindow is the number of consecutive pivots without a measurable
// drop in the largest primal-bound violation that triggers the cycle
// guard. 200 is a hand-tuned compromise: degenerate LPs that genuinely
// make progress (Hilbert-style staircases) get a few hundred zero-step
// pivots between drops; markshare_4_0 cycles forever, so anything in this
// range catches it without spurious bailouts elsewhere.
const dualStallWindow = 200

// applyWarmStart installs a warm-start basis into the current solver. It
// is the caller's responsibility to ensure the warm-start was produced by
// a revisedBackend solve on a request with the same standard-form column
// layout (same n, m, and senses). Returns an error if the resulting basis
// is numerically singular.
//
// After applyWarmStart the solver:
//   - has basicCols and status copied from the warm handle,
//   - has non-basic anchor values consistent with the request's bounds
//     (a non-basic that was atUpper but whose old upper exceeded the new
//     upper is clamped to the new upper and may flip status to atLower
//     when the new upper equals lower),
//   - has xBasic recomputed via B^{-1}(b − N·x_N) so the primal state
//     matches the bounds even after they shifted.
func (s *Solver) applyWarmStart(warm *WarmStart, bNorm []float64) error {
	if warm == nil {
		return errSingularBasis
	}
	if len(warm.basicCols) != s.m || len(warm.nonBasicStatus) != s.nCols {
		return errSingularBasis
	}
	copy(s.basicCols, warm.basicCols)
	copy(s.status, warm.nonBasicStatus)

	// Clamp non-basic anchors to the current bounds.
	for j := 0; j < s.nCols; j++ {
		switch s.status[j] {
		case ColBasic:
			continue
		case ColArtificial:
			continue
		case ColAtUpper:
			if math.IsInf(s.upper[j], 1) {
				s.status[j] = ColAtLower
			} else if s.upper[j] <= s.lower[j] {
				s.status[j] = ColAtLower
			}
		case ColAtLower:
			if s.lower[j] >= s.upper[j] && !math.IsInf(s.upper[j], 1) {
				// Lower == upper. Keep at lower; behaves the same as upper.
			}
		}
	}

	// Reuse the parent's factorization when persistence is on. The child's
	// basis matrix B equals the parent's (only a variable bound changed), so
	// warm.lu + warm.etas is exactly B^{-1}; we only need to recompute x_B for
	// the shifted bounds. Guard against (a) an eta chain already at the refactor
	// cap and (b) accumulated drift — either falls back to a fresh refactor so
	// correctness never depends on the reused factor staying accurate.
	if s.opts.PersistFactor && warm.lu != nil {
		maxEta := s.maxEtaUpdatesBeforeRefactor()
		if maxEta >= 0 && len(warm.etas) < maxEta {
			s.lu = warm.lu
			s.etaUpdates = append([]etaUpdate(nil), warm.etas...)
			// Restore the cost-based cadence counters installFactor would
			// have set: without them factorNnz stays 0 and the eta-vs-factor
			// rule refactors on the very first pivot, discarding the reused
			// LU this branch exists to keep.
			s.factorNnz = warm.lu.nnz()
			s.etaNnzSum = 0
			for _, eta := range s.etaUpdates {
				s.etaNnzSum += eta.col.Len() + 1
			}
			s.recomputeBasicValues()
			resid, relResid, _ := s.basisResidualInf()
			if resid > s.stats.MaxBasisResidual {
				s.stats.MaxBasisResidual = resid
			}
			if relResid > s.stats.MaxBasisRelResidual {
				s.stats.MaxBasisRelResidual = relResid
			}
			if relResid <= s.etaResidualTolerance() {
				s.stats.FactorReuses++
				return nil
			}
			// Reused factor drifted past tolerance — fall through to refactor.
		}
	}

	if err := s.refactor(); err != nil {
		return err
	}

	return nil
}

// runDualSimplex runs the dual simplex loop with the given (min-sense) cost
// vector. Returns Optimal when primal-feasible (and hence optimal given the
// pre-existing dual feasibility), Infeasible when the dual ratio test fails,
// or the standard limit statuses.
func (s *Solver) runDualSimplex(cost []float64) Status {
	return s.runDualSimplexCapped(cost, s.opts.MaxIterations)
}

// runDualSimplexCapped is the same loop bounded by an iteration ceiling that
// is independent of MaxIterations. Used by the warm-start path so a single
// dual stall doesn't consume the entire LP iteration budget.
//
// Also detects cycling on degenerate LPs: if the largest bound violation
// hasn't shrunk meaningfully over `dualStallWindow` consecutive iterations
// we declare a stall and return IterationLimit. The caller falls back to a
// cold solve which uses Phase I to detect infeasibility unambiguously. This
// guards against problems like markshare_4_0 where the dual ratio test
// without Bland's rule cycles through equal-step pivots indefinitely.
func (s *Solver) runDualSimplexCapped(cost []float64, iterCap int) Status {
	return s.runDualSimplexLimited(cost, iterCap, dualStallWindow)
}

// runDualSimplexLimited is the dual reoptimization loop with an optional
// no-progress guard. Ordinary node solves use dualStallWindow so a cycling
// warm start falls back to Phase I promptly. Strong branching passes zero:
// its own small iteration limit is the stopping rule, matching an LP
// interface strong-branch call where an iteration-limit exit is a useful
// estimate rather than a numerical failure that should trigger a cold solve.
func (s *Solver) runDualSimplexLimited(cost []float64, iterCap, stallWindow int) Status {
	t0 := time.Now()
	defer func() { s.stats.DualNanos += time.Since(t0).Nanoseconds() }()
	s.dualInfeasibilityCertified = false
	s.farkasProof = nil
	feasTol := s.opts.FeasibilityTol
	startIter := s.iterations
	bestSeenViol := math.Inf(1)
	stallCount := 0
	for {
		if err := s.ctx.Err(); err != nil {
			return TimeLimit
		}
		if s.opts.TimeLimit > 0 && time.Since(s.start) >= s.opts.TimeLimit {
			return TimeLimit
		}
		if s.iterations >= s.opts.MaxIterations || (s.iterations-startIter) >= iterCap {
			s.stats.DualIterationCaps++
			return IterationLimit
		}

		// 1. Find leaving basic row with the largest bound violation.
		leaveRow := -1
		sigmaR := 0
		bestViol := feasTol
		for i := 0; i < s.m; i++ {
			bc := s.basicCols[i]
			if !math.IsInf(s.lower[bc], -1) && s.xBasic[i] < s.lower[bc]-feasTol {
				if v := s.lower[bc] - s.xBasic[i]; v > bestViol {
					bestViol = v
					leaveRow = i
					sigmaR = 1
				}
			} else if !math.IsInf(s.upper[bc], 1) && s.xBasic[i] > s.upper[bc]+feasTol {
				if v := s.xBasic[i] - s.upper[bc]; v > bestViol {
					bestViol = v
					leaveRow = i
					sigmaR = -1
				}
			}
		}
		if leaveRow < 0 {
			return Optimal
		}

		// Stall detector. Require >0.1% relative decrease vs the best
		// violation we've ever seen; degenerate cycles repeat the same
		// bestViol forever, so any genuine progress fires the reset.
		if stallWindow > 0 {
			if bestViol < bestSeenViol*(1-1e-3) {
				bestSeenViol = bestViol
				stallCount = 0
			} else {
				stallCount++
			}
			if stallCount >= stallWindow {
				s.stats.DualStalls++
				debugf("dual: stall detected after %d iters (bestViol=%g, bestSeen=%g)",
					s.iterations-startIter, bestViol, bestSeenViol)
				return IterationLimit
			}
		}

		// 2. Compute ρ = e_r^T B^{-1}. scrEr stays all-zero between iterations:
		// set the single 1, BTRAN (which copies internally), then clear it.
		er := s.scrEr
		er[leaveRow] = 1
		rho := s.scrRho
		s.btranInto(rho, er)
		er[leaveRow] = 0

		// Dual y for reduced costs.
		cB := s.scrCB
		for i := 0; i < s.m; i++ {
			cB[i] = cost[s.basicCols[i]]
		}
		y := s.scrY
		s.btranInto(y, cB)

		// 3. Dual ratio test — Harris two-pass. Pass 1 collects every
		// eligible column with its pivot magnitude |alpha| and exact ratio
		// |d/alpha|, tracking the tolerance-relaxed minimum ratio
		// (|d|+optTol)/|alpha|. Pass 2 picks the LARGEST |alpha| among
		// candidates whose exact ratio fits under the relaxed minimum
		// (ties → smallest column index, for determinism). The old
		// smallest-ratio rule kept admitting near-zero pivots on degenerate
		// duals, ill-conditioning the basis until the sparse LU needed its
		// dense rescue — 628 rescues/30s on neos-3754480-nidda once warm
		// reopts started firing everywhere.
		optTol := s.opts.OptimalityTol
		nCand := 0
		relaxMin := math.Inf(1)
		collectCandidates := func(pivotTol float64) {
			nCand = 0
			relaxMin = math.Inf(1)
			for j := 0; j < s.nCols; j++ {
				st := s.status[j]
				if st == ColBasic || st == ColArtificial {
					continue
				}
				if s.upper[j]-s.lower[j] <= feasTol {
					continue
				}
				alpha := s.cols[j].dotDense(rho)
				if math.Abs(alpha) < pivotTol {
					continue
				}
				sigmaQ := 1
				if st == ColAtUpper {
					sigmaQ = -1
				}
				if float64(sigmaQ*sigmaR)*alpha >= 0 {
					continue
				}
				d := cost[j] - s.cols[j].dotDense(y)
				absAlpha := math.Abs(alpha)
				mag := math.Abs(d / alpha)
				if rel := (math.Abs(d) + optTol) / absAlpha; rel < relaxMin {
					relaxMin = rel
				}
				s.scrCandJ[nCand] = int32(j)
				s.scrCandAlpha[nCand] = alpha
				s.scrCandMag[nCand] = mag
				nCand++
			}
		}
		collectCandidates(feasTol)
		if nCand == 0 {
			s.stats.DualNoEntering++
			if s.opts.CertifyInfeasible && s.certifyDualLeavingRow(rho, leaveRow, sigmaR) {
				s.dualInfeasibilityCertified = true
			}
			debugf("dual: no eligible entering column at iter %d (violation=%g)", s.iterations, bestViol)
			return Infeasible
		}
		bestQ := -1
		bestAlpha := 0.0
		if harrisDual() {
			for k := 0; k < nCand; k++ {
				if s.scrCandMag[k] > relaxMin {
					continue
				}
				if a := math.Abs(s.scrCandAlpha[k]); a > math.Abs(bestAlpha) {
					bestQ = int(s.scrCandJ[k])
					bestAlpha = s.scrCandAlpha[k]
				}
			}
		}
		if bestQ < 0 {
			// Every exact ratio sits above the relaxed minimum (only possible
			// through float asymmetry between the two formulas); fall back to
			// the plain minimum-ratio candidate.
			bestMag := math.Inf(1)
			for k := 0; k < nCand; k++ {
				if s.scrCandMag[k] < bestMag {
					bestMag = s.scrCandMag[k]
					bestQ = int(s.scrCandJ[k])
					bestAlpha = s.scrCandAlpha[k]
				}
			}
		}

		// 4. Compute u = B^{-1} A_q for the primal step.
		u := s.scrU
		s.ftranInto(u, s.cols[bestQ].writeDense(s.scrCol))

		leaveCol := s.basicCols[leaveRow]
		var bound float64
		var leaveStatus ColStatus
		if sigmaR == 1 {
			bound = s.lower[leaveCol]
			leaveStatus = ColAtLower
		} else {
			bound = s.upper[leaveCol]
			leaveStatus = ColAtUpper
		}
		sigmaQ := 1.0
		if s.status[bestQ] == ColAtUpper {
			sigmaQ = -1.0
		}
		// θ_p such that x_B[r] post-step = bound:
		//   x_B[r] − σ_q θ_p u_r = bound, so θ_p = (x_B[r] − bound) / (σ_q · u_r).
		// Use bestAlpha (== u_r post-FTRAN, modulo tiny float drift).
		thetaP := (s.xBasic[leaveRow] - bound) / (sigmaQ * bestAlpha)

		for i := 0; i < s.m; i++ {
			s.xBasic[i] -= sigmaQ * thetaP * u[i]
		}

		var enterAnchor float64
		if s.status[bestQ] == ColAtLower {
			enterAnchor = s.lower[bestQ]
		} else {
			enterAnchor = s.upper[bestQ]
		}
		s.basicCols[leaveRow] = bestQ
		s.status[bestQ] = ColBasic
		s.status[leaveCol] = leaveStatus
		s.xBasic[leaveRow] = enterAnchor + sigmaQ*thetaP

		if err := s.updateFactorAfterPivot(leaveRow, u); err != nil {
			debugf("dual: basis update failed at iter %d: %v", s.iterations, err)
			return IterationLimit
		}
		s.iterations++
		s.stats.DualIterations++
	}
}

// certifyDualLeavingRow validates the Farkas proof behind a dual "no
// entering column" exit the way SCIP validates a solver-provided ray
// (SCIPlpGetDualfarkas under lp/checkfarkas, lp.c): the ray y = -sigmaR*rho
// with rho = e_r^T B^-1 is aggregated over EVERY column of the equality-form
// model, coef_j = y*A_j, and the proof is accepted when the box-maximal
// activity of y*Ax provably stays below y*b:
//
//	max{ sum_j coef_j*x_j : l <= x <= u } < y*b  ==>  no feasible x.
//
// The check is self-contained — it does not care how y was produced — so no
// basis-residual precondition is needed. (The previous derivation-gated form
// omitted the basic columns from the row and had to prove that omission
// legal against a near-exact identity-row test, which rejected drifted-basis
// candidates wholesale.)
//
// Aggregated coefficients at or below the dual ratio test's own candidacy
// tolerance are treated as zero, mirroring SCIP's SCIPsetIsDualfeasZero
// skip: a "no entering column" exit means every helpful column carries only
// a sub-tolerance coefficient, and rejecting the proof on such a column's
// infinite width blocked 374 of 376 candidates on neos17. Where q-opt stays
// more conservative than SCIP: a skipped column with FINITE bounds still
// contributes its worst-case activity to the acceptance margin (SCIP drops
// it entirely), so only the noise-times-unbounded-column judgment is taken
// on tolerance. A significant coefficient on an unbounded side still
// rejects — no finite activity bound exists — matching SCIP's invalid-proof
// cases.
func (s *Solver) certifyDualLeavingRow(rho []float64, leaveRow, sigmaR int) bool {
	if leaveRow < 0 || leaveRow >= s.m || len(rho) != s.m {
		s.stats.WarmCertRejectInvalid++
		return false
	}
	sgn := -float64(sigmaR)
	rhsVal := 0.0
	for i, v := range rho {
		rhsVal += v * s.bNorm[i]
	}
	rhsVal *= sgn
	if math.IsNaN(rhsVal) || math.IsInf(rhsVal, 0) {
		debugf("farkas cert: invalid rhsVal=%g", rhsVal)
		s.stats.WarmCertRejectInvalid++
		return false
	}
	// skipTol mirrors collectCandidates' pivot threshold: a coefficient the
	// ratio test would not even consider pivoting on is numerical noise for
	// the proof too.
	skipTol := s.opts.FeasibilityTol
	maxact := 0.0
	scale := math.Abs(rhsVal)
	slack := 0.0
	// Proof export runs alongside the acceptance test. Structural
	// coefficients are kept exactly (including sub-tolerance ones — their
	// worst case cannot be folded into the exported RHS because other solves
	// may use wider boxes); slack columns fold their favorable-bound activity
	// into slackFold, which is bound-independent across re-solves of the same
	// body. The acceptance arithmetic below is untouched.
	proofInds := s.scrProofInds[:0]
	proofCoefs := s.scrProofCoefs[:0]
	slackFold := 0.0
	for j := 0; j < s.nCols; j++ {
		// Artificial columns are Phase I devices: the model is the
		// artificial == 0 slice, so they contribute exactly zero.
		if s.isArtificial[j] {
			continue
		}
		coef := sgn * s.cols[j].dotDense(rho)
		if math.Abs(coef) <= skipTol {
			if coef != 0 {
				// Worst-case activity of a skipped finite column goes into
				// the margin; an unbounded one is trusted as true zero.
				lo, hi := s.lower[j], s.upper[j]
				if !math.IsInf(lo, 0) && !math.IsInf(hi, 0) {
					slack += math.Abs(coef) * math.Max(math.Abs(lo), math.Abs(hi))
					if j < s.n {
						proofInds = append(proofInds, int32(j))
						proofCoefs = append(proofCoefs, coef)
					} else if coef > 0 {
						slackFold += coef * hi
					} else {
						slackFold += coef * lo
					}
				}
			}
			continue
		}
		bound := s.upper[j]
		if coef < 0 {
			bound = s.lower[j]
		}
		if math.IsInf(bound, 0) || math.IsNaN(bound) {
			debugf("farkas cert: infinite reject col=%d coef=%g lo=%g hi=%g sigmaR=%d", j, coef, s.lower[j], s.upper[j], sigmaR)
			s.stats.WarmCertRejectInfinite++
			return false
		}
		term := coef * bound
		maxact += term
		scale += math.Abs(term)
		if math.IsInf(maxact, 0) || math.IsNaN(maxact) {
			s.stats.WarmCertRejectInvalid++
			return false
		}
		if j < s.n {
			proofInds = append(proofInds, int32(j))
			proofCoefs = append(proofCoefs, coef)
		} else {
			slackFold += term
		}
	}
	margin := 100*s.opts.FeasibilityTol*(1+scale) + slack
	if maxact+margin >= rhsVal {
		debugf("farkas cert: margin reject maxact=%g margin=%g rhsVal=%g sigmaR=%d", maxact, margin, rhsVal, sigmaR)
		s.stats.WarmCertRejectMargin++
		return false
	}
	s.scrProofInds, s.scrProofCoefs = proofInds, proofCoefs
	s.farkasProof = &FarkasProof{
		Inds:  append([]int32(nil), proofInds...),
		Coefs: append([]float64(nil), proofCoefs...),
		RHS:   rhsVal - slackFold,
	}
	return true
}
