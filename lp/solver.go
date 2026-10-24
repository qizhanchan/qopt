package lp

import (
	"context"
	"math"
	"time"
)

// Bounded-variable revised simplex with an LU-factored basis.
//
// The solver materializes the LP in standard equality form (Ax = b,
// lb ≤ x ≤ ub) by adding one slack per LE row, one surplus per GE row,
// and one artificial per GE/EQ row. The initial basis is identity (each
// LE row uses its slack, each GE/EQ row uses its artificial), so the
// first LU factor is trivial.
//
// Internal convention is MINIMIZE; the problem's c is already in MAX sense,
// so Phase II's cost is −c on the structural columns and 0 elsewhere.
// Phase I minimizes the sum of artificials directly. Pricing uses Dantzig's
// rule; the ratio test is the standard bounded-variable test that allows
// bound-flip pivots (the entering variable can hop from one bound to the
// other without changing the basis).

// Solver is the revised-simplex state for one solve. It is exported so
// basis-level consumers (Gomory cut generation, reduced-cost fixing) can read
// the final basis through the accessor methods at the bottom of this file;
// all fields are internal.
type Solver struct {
	ctx   context.Context
	opts  Options
	start time.Time

	m, n, nCols int

	cols  []SparseVector // cols[j] = compressed nonzero entries of column j (length nCols)
	cost  []float64      // user objective (maximization sense) on each column
	lower []float64
	upper []float64
	bNorm []float64 // normalized (b ≥ 0) RHS, length m — kept so dual reoptimize can recompute xBasic

	basicCols []int       // basicCols[i] = column basic in row i (length m)
	status    []ColStatus // length nCols
	xBasic    []float64   // length m, current value of basicCols[i]

	isArtificial []bool
	artificials  []int

	// form is the compiled standard form this solver was built on (shared,
	// read-only). Carried so warm-start handles record their flip pattern.
	form *compiledForm

	lu         basisFactor
	etaUpdates []etaUpdate
	refactors  int
	stats      Stats

	// Cost-based refactor rule: etaNnzSum accumulates the stored eta-column
	// sizes since the last refactor; factorNnz is the current factor's size.
	// Every FTRAN/BTRAN walks the whole eta chain (≈ etaNnzSum extra work per
	// solve op) while a fresh factor costs ≈ factorNnz — once the chain
	// outweighs the factor, refactoring is cheaper than dragging the chain.
	etaNnzSum int
	factorNnz int

	// forceDenseFactor pins this solve to the dense LU once the sparse LU has
	// been caught steering the simplex into a non-terminating degenerate
	// plateau (Bland's rule defeated by sparse-factor float noise — the glass4
	// class). factorBasis honors it; set by runSimplex's Bland-stall guard.
	forceDenseFactor bool

	// denseRescueFutile records that a dense-LU redo of an "unstable" sparse
	// factor failed to improve the basis residual — i.e. the basis itself is
	// ill-conditioned and no factorization fixes it. Further redos this solve
	// are skipped: on neos-3754480-nidda the futile redos were 601 × O(m³)
	// ≈ 55% of the 30s budget, with before/after residuals byte-comparable.
	// Correctness never rested on the redo (Phase I retries, the MIP
	// cold-verify, and the reopt verification pass still stand).
	denseRescueFutile bool
	// dualInfeasibilityCertified is scoped to one runDualSimplexCapped call.
	// The incremental warm path reads it immediately after an Infeasible
	// return and then either accepts the proof or performs the cold fallback.
	dualInfeasibilityCertified bool
	// farkasProof carries the exported proof row of the certificate above,
	// with the same scope. Nil whenever dualInfeasibilityCertified is false.
	farkasProof *FarkasProof
	// onDenseRescueOutcome, when set, reports each dense-redo diagnosis
	// (futile or helpful) to the owning Incremental, which schedules an
	// exponential-backoff cooldown for upcoming solves — near-singularity is
	// a per-basis property, so the tree backs off instead of latching
	// permanently (see Incremental.noteDenseRescueOutcome).
	onDenseRescueOutcome func(futile bool)

	iterations int

	// Per-solve scratch buffers for the simplex hot loops. Every simplex
	// pivot needs ~10 length-m vectors (dual, pivot column, FTRAN/BTRAN
	// workspaces, residual accumulators); allocating them per pivot made the
	// GC/scavenger the top CPU consumer on allocation-heavy MIPLIB runs.
	// Distinct roles get distinct buffers so no two live at once alias;
	// scrT1/scrT2 are used only inside ftranInto/btranInto.
	scrCB, scrY, scrU, scrRho, scrEr, scrRhs, scrCol, scrT1, scrT2, scrActivity []float64 // length m
	scrValues                                                                   []float64 // length nCols
	// Farkas-proof capture buffers (certifyDualLeavingRow); reused across
	// certificates, copied out only on acceptance.
	scrProofInds  []int32
	scrProofCoefs []float64
	// Dual-ratio-test candidate stash for the Harris two-pass test (indices,
	// pivot magnitudes, exact ratios per eligible column).
	scrCandJ                 []int32
	scrCandAlpha, scrCandMag []float64
}

// initScratch sizes the per-solve workspace. Called once from build.
func (s *Solver) initScratch() {
	m := s.m
	buf := make([]float64, 10*m)
	s.scrCB, buf = buf[:m], buf[m:]
	s.scrY, buf = buf[:m], buf[m:]
	s.scrU, buf = buf[:m], buf[m:]
	s.scrRho, buf = buf[:m], buf[m:]
	s.scrEr, buf = buf[:m], buf[m:]
	s.scrRhs, buf = buf[:m], buf[m:]
	s.scrCol, buf = buf[:m], buf[m:]
	s.scrT1, buf = buf[:m], buf[m:]
	s.scrT2, buf = buf[:m], buf[m:]
	s.scrActivity = buf[:m]
	s.scrValues = make([]float64, s.nCols)
	s.scrCandJ = make([]int32, s.nCols)
	cand := make([]float64, 2*s.nCols)
	s.scrCandAlpha = cand[:s.nCols]
	s.scrCandMag = cand[s.nCols:]
}

// ftranInto computes dst = B^{-1} rhs through the LU factor and the eta
// chain, allocation-free. dst must be distinct from rhs and from the
// internal scratch buffers.
func (s *Solver) ftranInto(dst, rhs []float64) {
	s.lu.ftranInto(dst, rhs, s.scrT1)
	for _, eta := range s.etaUpdates {
		applyEtaFTRAN(dst, eta)
	}
}

// btranInto computes dst = B^{-T} rhs, allocation-free. rhs is copied into
// an internal scratch first (the eta pass mutates it), so dst may alias rhs.
func (s *Solver) btranInto(dst, rhs []float64) {
	copy(s.scrT2, rhs)
	for i := len(s.etaUpdates) - 1; i >= 0; i-- {
		applyEtaBTRAN(s.scrT2, s.etaUpdates[i])
	}
	s.lu.btranInto(dst, s.scrT2, s.scrT1)
}

const defaultMaxEtaUpdatesBeforeRefactor = 64
const defaultEtaResidualFactor = 100.0

// denseRescueImproveFactor is the residual-improvement ratio below which a
// dense redo that misses tolerance still counts as helping (not futile).
const denseRescueImproveFactor = 10.0

// denseRescueWasFutile decides whether a dense redo of an unstable sparse
// factor was futile: the dense residual is still above tolerance AND within
// an order of magnitude of the sparse residual. A redo that misses tolerance
// but improves the residual ≥10× is treated as helping — the instability is
// in the factorization, not the basis — and must not feed the backoff.
func denseRescueWasFutile(sparseRel, denseRel, tol float64) bool {
	return denseRel > tol && denseRel*denseRescueImproveFactor > sparseRel
}

func (s *Solver) refactor() error {
	lu, err := s.factorBasis()
	if err != nil {
		return err
	}
	s.installFactor(lu)
	resid, relResid, scale := s.basisResidualInf()

	// Sparse LU can be numerically unstable on ill-conditioned bases (the
	// glass4 class): the factor "succeeds" but cannot represent B accurately,
	// so x_B is wrong and the primal simplex churns to MaxIterations. If a
	// fresh sparse factor's relative basis residual exceeds the eta-residual
	// tolerance, redo with the robust dense LU (full partial pivoting). This is
	// what lets the sparse LU stay correct on degenerate/ill-scaled inputs.
	if !s.opts.DenseLU && !s.denseRescueFutile && relResid > s.etaResidualTolerance() {
		debugf("revised: sparse LU unstable (relResidual=%g > %g) — dense fallback",
			relResid, s.etaResidualTolerance())
		s.stats.SparseDenseFallbacks++
		basis := make([][]float64, s.m)
		for i := 0; i < s.m; i++ {
			basis[i] = s.cols[s.basicCols[i]].toDense(s.m)
		}
		t0 := time.Now()
		dlu, derr := factorDense(basis)
		s.stats.DenseFactors++
		s.stats.DenseFactorNanos += time.Since(t0).Nanoseconds()
		if derr != nil {
			return derr
		}
		lu = dlu
		s.installFactor(lu)
		sparseRel := relResid
		resid, relResid, scale = s.basisResidualInf()
		futile := denseRescueWasFutile(sparseRel, relResid, s.etaResidualTolerance())
		if futile {
			// The dense factor left the residual in the sparse factor's
			// ballpark: the basis itself is near-singular and redos are
			// wasted O(m³). Diagnose once per solve, then stop paying.
			s.denseRescueFutile = true
			debugf("revised: dense redo futile, relResidual %g → %g (dense minPiv=%g) — disabling redos this solve",
				sparseRel, relResid, dlu.minPivot())
		}
		if s.onDenseRescueOutcome != nil {
			s.onDenseRescueOutcome(futile)
		}
	}

	s.refactors++
	s.stats.Refactors++
	if mp := lu.minPivot(); mp > 0 && (s.stats.MinPivot == 0 || mp < s.stats.MinPivot) {
		s.stats.MinPivot = mp
	}
	if g := lu.growthFactor(); g > s.stats.MaxGrowth {
		s.stats.MaxGrowth = g
	}
	if resid > s.stats.MaxBasisResidual {
		s.stats.MaxBasisResidual = resid
	}
	if relResid > s.stats.MaxBasisRelResidual {
		s.stats.MaxBasisRelResidual = relResid
	}
	if debugEnabled() {
		tol := s.opts.FeasibilityTol * math.Max(1, scale) * 100
		relTol := s.etaResidualTolerance()
		if resid > tol || relResid > relTol || lu.growthFactor() > 1e8 || lu.minPivot() < 1e-10 {
			debugf("revised: refactor #%d iter=%d minPivot=%g growth=%g basisResidual=%g relResidual=%g scale=%g",
				s.refactors, s.iterations, lu.minPivot(), lu.growthFactor(), resid, relResid, scale)
		}
	}
	return nil
}

// installFactor sets the current basis factor, clears the eta chain, and
// refreshes x_B. Stats/debug are recorded by refactor once the final factor
// (after any sparse→dense fallback) is chosen.
func (s *Solver) installFactor(lu basisFactor) {
	s.lu = lu
	s.etaUpdates = nil
	s.etaNnzSum = 0
	s.factorNnz = lu.nnz()
	s.recomputeBasicValues()
}

// factorBasis builds the basis factorization. Unless DenseLU is set it tries
// the sparse LU first and falls back to the dense LU when the sparse factor is
// singular; the numerical-instability fallback (non-singular but inaccurate) is
// handled in refactor via the basis residual.
func (s *Solver) factorBasis() (basisFactor, error) {
	if !s.opts.DenseLU && !s.forceDenseFactor {
		basisCols := make([]SparseVector, s.m)
		for i := 0; i < s.m; i++ {
			basisCols[i] = s.cols[s.basicCols[i]]
		}
		t0 := time.Now()
		lu, err := factorSparse(basisCols, s.m)
		s.stats.SparseFactorNanos += time.Since(t0).Nanoseconds()
		if err == nil {
			s.stats.SparseFactors++
			return lu, nil
		}
		debugf("revised: sparse LU singular — dense fallback: %v", err)
		s.stats.SparseDenseFallbacks++
		// fall through to dense
	}
	basis := make([][]float64, s.m)
	for i := 0; i < s.m; i++ {
		basis[i] = s.cols[s.basicCols[i]].toDense(s.m)
	}
	t0 := time.Now()
	lu, err := factorDense(basis)
	s.stats.DenseFactors++
	s.stats.DenseFactorNanos += time.Since(t0).Nanoseconds()
	if err == nil && !s.opts.DenseLU && !s.forceDenseFactor {
		debugf("revised: dense fallback factored ok: m=%d minPiv=%g growth=%g",
			s.m, lu.minPivot(), lu.growthFactor())
	}
	return lu, err
}

// recomputeBasicValues refreshes x_B from the current basis and non-basic
// anchors:
//
//	B x_B = b - Σ A_j x_j.
//
// This is deliberately tied to refactorization. The primal updates inside a
// pivot are cheap, but over long MIPLIB runs they accumulate drift; rebuilding
// x_B whenever we rebuild LU keeps the algebraic state synchronized with the
// current basis.
func (s *Solver) recomputeBasicValues() {
	rhs := s.scrRhs
	copy(rhs, s.bNorm)
	for j := 0; j < s.nCols; j++ {
		if s.status[j] == ColBasic {
			continue
		}
		anchor := s.nonBasicAnchor(j)
		if anchor == 0 {
			continue
		}
		s.cols[j].addToDense(rhs, -anchor)
	}
	s.ftranInto(s.xBasic, rhs)
}

func (s *Solver) ftran(rhs []float64) []float64 {
	x := s.lu.ftran(rhs)
	for _, eta := range s.etaUpdates {
		applyEtaFTRAN(x, eta)
	}
	return x
}

func (s *Solver) btran(rhs []float64) []float64 {
	y := append([]float64(nil), rhs...)
	for i := len(s.etaUpdates) - 1; i >= 0; i-- {
		applyEtaBTRAN(y, s.etaUpdates[i])
	}
	return s.lu.btran(y)
}

func (s *Solver) updateFactorAfterPivot(leaveRow int, u []float64) error {
	if leaveRow < 0 || leaveRow >= s.m || len(u) != s.m {
		return errSingularBasis
	}
	maxEta := s.maxEtaUpdatesBeforeRefactor()
	if maxEta < 0 {
		return s.refactor()
	}
	pivot := u[leaveRow]
	if math.Abs(pivot) <= 1e-12 || math.IsNaN(pivot) || math.IsInf(pivot, 0) {
		s.stats.EtaRefactorFallbacks++
		return s.refactor()
	}

	eta := newEtaUpdate(leaveRow, u)
	s.etaUpdates = append(s.etaUpdates, eta)
	s.etaNnzSum += eta.col.Len() + 1
	s.stats.EtaUpdates++
	if len(s.etaUpdates) > s.stats.MaxEtaDepth {
		s.stats.MaxEtaDepth = len(s.etaUpdates)
	}
	if len(s.etaUpdates) >= maxEta {
		s.stats.EtaRefactorFallbacks++
		return s.refactor()
	}
	if k := etaCostFactor(); k > 0 && float64(s.etaNnzSum) > k*float64(s.factorNnz) {
		// Cost-based cadence: the chain now costs more per FTRAN/BTRAN than
		// a fresh factor's triangular solves — refactor early. On glass4 the
		// fixed 64-eta cap left the chain 9× the factor cost (34% of CPU in
		// eta application vs 3.7% in factorSparse).
		return s.refactor()
	}

	s.recomputeBasicValues()
	resid, relResid, _ := s.basisResidualInf()
	if resid > s.stats.MaxBasisResidual {
		s.stats.MaxBasisResidual = resid
	}
	if relResid > s.stats.MaxBasisRelResidual {
		s.stats.MaxBasisRelResidual = relResid
	}
	tol := s.etaResidualTolerance()
	if relResid > tol {
		s.stats.EtaRefactorFallbacks++
		debugf("revised: eta relative residual %g exceeds %g after %d updates; full refactor", relResid, tol, len(s.etaUpdates))
		return s.refactor()
	}
	return nil
}

func (s *Solver) maxEtaUpdatesBeforeRefactor() int {
	if s.opts.MaxEtaUpdates < 0 {
		return -1
	}
	if s.opts.MaxEtaUpdates > 0 {
		return s.opts.MaxEtaUpdates
	}
	return defaultMaxEtaUpdatesBeforeRefactor
}

func (s *Solver) etaResidualTolerance() float64 {
	factor := s.opts.EtaResidualFactor
	if factor <= 0 {
		factor = defaultEtaResidualFactor
	}
	return factor * s.opts.FeasibilityTol
}

func (s *Solver) nonBasicAnchor(j int) float64 {
	switch s.status[j] {
	case ColAtUpper:
		if math.IsInf(s.upper[j], 1) {
			return s.lower[j]
		}
		return s.upper[j]
	default:
		return s.lower[j]
	}
}

func (s *Solver) basisResidualInf() (float64, float64, float64) {
	values := s.scrValues
	for i, bc := range s.basicCols {
		values[bc] = s.xBasic[i]
	}
	for j := 0; j < s.nCols; j++ {
		if s.status[j] != ColBasic {
			values[j] = s.nonBasicAnchor(j)
		}
	}

	activity := s.scrActivity
	for i := range activity {
		activity[i] = 0
	}
	for j := 0; j < s.nCols; j++ {
		value := values[j]
		if value == 0 {
			continue
		}
		s.cols[j].addToDense(activity, value)
	}
	maxResid := 0.0
	maxRelResid := 0.0
	scale := 1.0
	for i := 0; i < s.m; i++ {
		rowResid := math.Abs(activity[i] - s.bNorm[i])
		if rowResid > maxResid {
			maxResid = rowResid
		}
		rowScale := math.Max(1, math.Max(math.Abs(activity[i]), math.Abs(s.bNorm[i])))
		if rel := rowResid / rowScale; rel > maxRelResid {
			maxRelResid = rel
		}
		if a := math.Abs(activity[i]); a > scale {
			scale = a
		}
		if b := math.Abs(s.bNorm[i]); b > scale {
			scale = b
		}
	}
	return maxResid, maxRelResid, scale
}

// reduce computes the reduced cost d_j = cost[j] − y^T A_j for the given
// dual vector y.
func (s *Solver) reduce(cost []float64, y []float64, j int) float64 {
	return cost[j] - s.cols[j].dotDense(y)
}

// dualValues returns y = B^{-T} cost_B for the given column-cost vector —
// the simplex dual consistent with the CURRENT basis. runSimplex recomputes
// this every iteration to price columns but discards it once the loop
// returns; callers that need it after the solve has reached Optimal (e.g.
// reduced-cost bound propagation) get it back via one extra BTRAN, which is
// O(m) against the already-factored basis.
func (s *Solver) dualValues(cost []float64) []float64 {
	cB := make([]float64, s.m)
	for i := 0; i < s.m; i++ {
		cB[i] = cost[s.basicCols[i]]
	}
	return s.btran(cB)
}

// phaseIResidualRetries is the number of refactor-and-resume attempts we
// make when Phase I terminates with residual artificial sum above the
// tolerance. Each retry rebuilds the LU from scratch and re-derives x_B,
// which corrects any drift accumulated through long eta chains, then
// resumes the Phase I simplex. Genuine infeasibility stays infeasible
// across retries; the markshare / nidda numerical-drift class collapses
// to a clean Optimal exit.
const phaseIResidualRetries = 2

func (s *Solver) phaseI() Status {
	if len(s.artificials) == 0 {
		return Optimal
	}
	t0 := time.Now()
	defer func() { s.stats.PhaseINanos += time.Since(t0).Nanoseconds() }()
	phaseCost := make([]float64, s.nCols)
	for _, j := range s.artificials {
		phaseCost[j] = 1
	}
	st := s.runSimplex(phaseCost, true /*phase I*/)
	if st != Optimal {
		return st
	}

	sum, ok := s.checkArtificialResidual()
	for retry := 0; !ok && retry < phaseIResidualRetries; retry++ {
		s.stats.PhaseIResidualRetries++
		debugf("revised: Phase I residual %g exceeds tolerance, refactor retry %d", sum, retry+1)
		if err := s.refactor(); err != nil {
			debugf("revised: Phase I residual refactor failed: %v", err)
			return IterationLimit
		}
		// recomputeBasicValues already ran inside refactor — re-check the
		// residual against the cleaned basis before paying for more pivots.
		if sum, ok = s.checkArtificialResidual(); ok {
			break
		}
		// Residual is real (or stuck above tol even with a clean factor).
		// Resume Phase I; it may now find a direction the drifted basis
		// couldn't see, or terminate cleanly at the same point.
		st = s.runSimplex(phaseCost, true)
		if st != Optimal {
			return st
		}
		sum, ok = s.checkArtificialResidual()
	}
	if !ok {
		debugf("revised: Phase I residual %g still exceeds tolerance after %d retries",
			sum, phaseIResidualRetries)
		return Infeasible
	}

	// Pin artificials at zero for Phase II by tightening their upper to 0
	// and tagging any non-basic one as ColArtificial (skipped by pricing).
	for _, j := range s.artificials {
		s.upper[j] = 0
		if s.status[j] != ColBasic {
			s.status[j] = ColArtificial
		}
	}
	return Optimal
}

func (s *Solver) infeasibilityCertificate() *InfeasibilityCertificate {
	if len(s.artificials) == 0 || s.form == nil {
		return nil
	}
	phaseCost := make([]float64, s.nCols)
	for _, j := range s.artificials {
		phaseCost[j] = 1
	}
	dual := s.dualValues(phaseCost)
	reduced := make([]float64, s.n)
	auxContribution := 0.0
	for j := 0; j < s.nCols; j++ {
		r := s.reduce(phaseCost, dual, j)
		if j < s.n {
			reduced[j] = r
			continue
		}
		if r < 0 {
			if math.IsInf(s.upper[j], 1) {
				return nil
			}
			auxContribution += r * s.upper[j]
		} else if r > 0 {
			auxContribution += r * s.lower[j]
		}
	}
	tol := s.opts.FeasibilityTol * math.Max(1, float64(len(s.artificials)))
	lowerBound := auxContribution
	for i, y := range dual {
		lowerBound += y * s.bNorm[i]
	}
	for j, r := range reduced {
		if r < 0 {
			if math.IsInf(s.upper[j], 1) {
				return nil
			}
			lowerBound += r * s.upper[j]
		}
	}
	if math.IsNaN(lowerBound) || lowerBound <= tol {
		return nil
	}
	return &InfeasibilityCertificate{
		Dual:            dual,
		ReducedCosts:    reduced,
		AuxContribution: auxContribution,
		Flip:            append([]bool(nil), s.form.flip...),
		Tolerance:       tol,
	}
}

// checkArtificialResidual returns the sum of positive artificial-variable
// values and whether it is within Phase I's feasibility tolerance. Used by
// phaseI to decide between "clean exit" and "force a refactor retry".
func (s *Solver) checkArtificialResidual() (float64, bool) {
	sum := 0.0
	for _, j := range s.artificials {
		if v := s.XValue(j); v > 0 {
			sum += v
		}
	}
	tol := s.opts.FeasibilityTol * math.Max(1, float64(len(s.artificials)))
	return sum, sum <= tol
}

func (s *Solver) phaseII() Status {
	t0 := time.Now()
	defer func() { s.stats.PhaseIINanos += time.Since(t0).Nanoseconds() }()
	// Internal MIN convention: negate the max-sense cost on structural cols.
	phaseCost := make([]float64, s.nCols)
	for j := 0; j < s.n; j++ {
		phaseCost[j] = -s.cost[j]
	}
	return s.runSimplex(phaseCost, false /*phase II*/)
}

// runSimplex is the primal bounded-variable simplex loop. The provided cost
// is assumed to be in minimization sense.
func (s *Solver) runSimplex(cost []float64, phaseI bool) Status {
	feasTol := s.opts.FeasibilityTol
	optTol := s.opts.OptimalityTol
	// Anti-cycling: Dantzig pricing has no termination guarantee on degenerate
	// LPs — it can pivot through a closed cycle of zero-length (degenerate)
	// steps forever, which on this backend manifested as a B&B node burning all
	// of MaxIterations (historically the reason the sparse LU could not be made
	// the default). After blandThreshold consecutive degenerate pivots we fall
	// back to Bland's rule — smallest eligible index in, smallest leaving index
	// out — which provably cannot cycle. The first productive (positive-length)
	// step reverts to Dantzig for speed, so the slow rule is paid only across an
	// actual stall.
	blandThreshold := s.m + 50
	degenerateIters := 0
	useBland := false
	for {
		if err := s.ctx.Err(); err != nil {
			return TimeLimit
		}
		if s.opts.TimeLimit > 0 && time.Since(s.start) >= s.opts.TimeLimit {
			return TimeLimit
		}
		if s.iterations >= s.opts.MaxIterations {
			return IterationLimit
		}

		// Dual y = B^{-T} c_B.
		cB := s.scrCB
		for i := 0; i < s.m; i++ {
			cB[i] = cost[s.basicCols[i]]
		}
		y := s.scrY
		s.btranInto(y, cB)

		// Find entering column. Dantzig's rule (most-improving reduced cost) by
		// default; Bland's rule (first eligible index) while stalling.
		bestJ := -1
		bestImprove := optTol
		bestSigma := 1.0
		for j := 0; j < s.nCols; j++ {
			st := s.status[j]
			if st == ColBasic || st == ColArtificial {
				continue
			}
			if s.upper[j]-s.lower[j] <= feasTol {
				// A fixed non-basic column cannot move, so its reduced-cost
				// sign is irrelevant to primal pricing and dual feasibility.
				continue
			}
			if !phaseI && s.isArtificial[j] {
				// Phase II: never let an artificial re-enter.
				continue
			}
			d := s.reduce(cost, y, j)
			var improve, sigma float64
			switch st {
			case ColAtLower:
				// Improving when d < −tol; magnitude = −d.
				improve, sigma = -d, 1.0
			case ColAtUpper:
				improve, sigma = d, -1.0
			default:
				continue
			}
			if improve <= optTol {
				continue
			}
			if useBland {
				// Bland's rule: take the smallest-index improving column.
				bestJ = j
				bestSigma = sigma
				break
			}
			if improve > bestImprove {
				bestImprove = improve
				bestJ = j
				bestSigma = sigma
			}
		}
		if bestJ < 0 {
			return Optimal
		}

		u := s.scrU
		s.ftranInto(u, s.cols[bestJ].writeDense(s.scrCol))

		// Ratio test. The entering variable moves by sigma*theta where
		// theta ∈ [0, theta_max], bounded by the entering var's own range
		// and by the basic variables' bounds. Under Dantzig pricing we use
		// the Harris two-pass test — pass 1 finds the tolerance-relaxed
		// minimum ratio, pass 2 picks the LARGEST |pivot| among rows whose
		// exact ratio fits under it — so tiny pivots (which ill-condition
		// the basis and used to force dense-LU rescues downstream) are only
		// chosen when no well-sized pivot is tied within tolerance. Under
		// Bland's rule we keep the classic single-pass smallest-index test:
		// its termination guarantee depends on that exact tie-break.
		var thetaMax float64
		boundFlip := true
		leaveRow := -1
		leaveAtUpper := false
		flipRange := s.upper[bestJ] - s.lower[bestJ]
		if useBland || !harrisPrimal() {
			thetaMax = flipRange
			for i := 0; i < s.m; i++ {
				su := bestSigma * u[i]
				if math.Abs(su) <= feasTol {
					continue
				}
				bc := s.basicCols[i]
				var limit float64
				atUpper := false
				if su > 0 {
					// x_B[i] decreases. Lower bound is binding.
					if math.IsInf(s.lower[bc], -1) {
						continue
					}
					limit = (s.xBasic[i] - s.lower[bc]) / su
				} else {
					if math.IsInf(s.upper[bc], 1) {
						continue
					}
					limit = (s.upper[bc] - s.xBasic[i]) / (-su)
					atUpper = true
				}
				if limit < -feasTol {
					// Numerical noise; clamp to zero step on this row.
					limit = 0
				}
				if limit < thetaMax-feasTol ||
					(math.Abs(limit-thetaMax) <= feasTol && leaveRow >= 0 && s.basicCols[i] < s.basicCols[leaveRow]) {
					thetaMax = limit
					leaveRow = i
					leaveAtUpper = atUpper
					boundFlip = false
				}
			}
			if math.IsInf(thetaMax, 1) {
				// No upper bound on the entering variable and no basic var
				// limits the step ⇒ LP is unbounded.
				return Unbounded
			}
		} else {
			// Harris pass 1: tolerance-relaxed minimum ratio over the rows.
			rowRelax := math.Inf(1)
			for i := 0; i < s.m; i++ {
				su := bestSigma * u[i]
				if math.Abs(su) <= feasTol {
					continue
				}
				bc := s.basicCols[i]
				var rel float64
				if su > 0 {
					if math.IsInf(s.lower[bc], -1) {
						continue
					}
					rel = (s.xBasic[i] - s.lower[bc] + feasTol) / su
				} else {
					if math.IsInf(s.upper[bc], 1) {
						continue
					}
					rel = (s.upper[bc] - s.xBasic[i] + feasTol) / (-su)
				}
				if rel < 0 {
					// Bound already violated beyond tolerance (drift); the
					// exact pass clamps such rows to a zero-length step, so
					// keep the relaxed cap consistent with that.
					rel = 0
				}
				if rel < rowRelax {
					rowRelax = rel
				}
			}
			if math.IsInf(rowRelax, 1) && math.IsInf(flipRange, 1) {
				return Unbounded
			}
			thetaCap := rowRelax
			if flipRange < thetaCap {
				thetaCap = flipRange
			}
			// Harris pass 2: exact ratios; among rows fitting under the
			// relaxed cap take the largest |pivot| (ties → smaller basic
			// column index, for determinism).
			minExact := math.Inf(1)
			bestPiv := 0.0
			bestLimit := 0.0
			for i := 0; i < s.m; i++ {
				su := bestSigma * u[i]
				asu := math.Abs(su)
				if asu <= feasTol {
					continue
				}
				bc := s.basicCols[i]
				var limit float64
				atUpper := false
				if su > 0 {
					if math.IsInf(s.lower[bc], -1) {
						continue
					}
					limit = (s.xBasic[i] - s.lower[bc]) / su
				} else {
					if math.IsInf(s.upper[bc], 1) {
						continue
					}
					limit = (s.upper[bc] - s.xBasic[i]) / (-su)
					atUpper = true
				}
				if limit < -feasTol {
					limit = 0
				}
				if limit < minExact {
					minExact = limit
				}
				if limit <= thetaCap {
					if asu > bestPiv ||
						(asu == bestPiv && leaveRow >= 0 && s.basicCols[i] < s.basicCols[leaveRow]) {
						bestPiv = asu
						bestLimit = limit
						leaveRow = i
						leaveAtUpper = atUpper
						boundFlip = false
					}
				}
			}
			if flipRange <= minExact {
				// The entering variable hits its opposite bound before any
				// basic variable leaves its own: bound flip, basis unchanged.
				thetaMax = flipRange
				boundFlip = true
				leaveRow = -1
			} else {
				thetaMax = bestLimit
			}
		}

		// Anti-cycling bookkeeping: a zero-length step makes no objective
		// progress and is the symptom of a degenerate cycle. Count consecutive
		// degenerate pivots; once they pass the threshold switch to Bland's
		// rule, and revert as soon as a productive step happens.
		if thetaMax > feasTol {
			degenerateIters = 0
			useBland = false
		} else {
			degenerateIters++
			if degenerateIters >= blandThreshold {
				useBland = true
			}
			// Bland's rule provably cannot cycle in exact arithmetic, but the
			// sparse LU's float noise can corrupt the ratio-test tie-breaks
			// enough to defeat it (glass4: ~100k consecutive degenerate Bland
			// pivots that the dense LU clears in tens). If Bland itself stalls
			// for another full threshold, pin this solve to the dense LU and
			// rebuild — exact directions restore termination. Bounds the wasted
			// work to ~2·(m+50) iters instead of the whole MaxIterations budget.
			if useBland && !s.opts.DenseLU && !s.forceDenseFactor &&
				degenerateIters >= 2*blandThreshold {
				debugf("revised: Bland stalled %d degenerate iters under sparse LU (phaseI=%v) — pinning to dense LU",
					degenerateIters, phaseI)
				s.forceDenseFactor = true
				s.stats.SparseDenseFallbacks++
				if err := s.refactor(); err != nil {
					return IterationLimit
				}
				degenerateIters = 0
				useBland = false
				continue
			}
		}

		// Apply the step.
		step := bestSigma * thetaMax
		if step != 0 {
			for i := 0; i < s.m; i++ {
				s.xBasic[i] -= step * u[i]
			}
		}

		if boundFlip {
			// Entering var hops between its bounds; basis unchanged.
			if s.status[bestJ] == ColAtLower {
				s.status[bestJ] = ColAtUpper
			} else {
				s.status[bestJ] = ColAtLower
			}
			s.iterations++
			s.countPhaseIter(phaseI)
			continue
		}

		// Basis change.
		oldBasic := s.basicCols[leaveRow]
		s.basicCols[leaveRow] = bestJ
		s.status[bestJ] = ColBasic
		if leaveAtUpper {
			s.status[oldBasic] = ColAtUpper
		} else {
			s.status[oldBasic] = ColAtLower
		}
		// The leaving row's xBasic is the entering var's new value.
		var enterAnchor float64
		if bestSigma > 0 {
			enterAnchor = s.lower[bestJ]
		} else {
			enterAnchor = s.upper[bestJ]
		}
		s.xBasic[leaveRow] = enterAnchor + step

		if err := s.updateFactorAfterPivot(leaveRow, u); err != nil {
			debugf("revised: basis update failed at iter %d: %v", s.iterations, err)
			return IterationLimit
		}
		s.iterations++
		s.countPhaseIter(phaseI)
	}
}

// countPhaseIter attributes one simplex pivot to the phase that performed it.
func (s *Solver) countPhaseIter(phaseI bool) {
	if phaseI {
		s.stats.PhaseIIterations++
	} else {
		s.stats.PhaseIIIterations++
	}
}

// dualFeasibleAt reports whether the current basis is dual-feasible for the
// given (min-sense) cost — i.e. no non-basic column would be an eligible
// entering column under runSimplex's Phase II pricing. Together with primal
// feasibility (what the dual simplex terminates with) this certifies a true
// optimum. One BTRAN plus one pricing pass.
func (s *Solver) dualFeasibleAt(cost []float64) bool {
	y := s.dualValues(cost)
	optTol := s.opts.OptimalityTol
	for j := 0; j < s.nCols; j++ {
		st := s.status[j]
		if st == ColBasic || st == ColArtificial {
			continue
		}
		if s.isArtificial[j] {
			continue
		}
		if s.upper[j]-s.lower[j] <= s.opts.FeasibilityTol {
			continue
		}
		d := s.reduce(cost, y, j)
		switch st {
		case ColAtLower:
			if -d > optTol {
				return false
			}
		case ColAtUpper:
			if d > optTol {
				return false
			}
		}
	}
	return true
}

// dualInfeasGap quantifies how far the current basis is from dual feasibility
// for the given (min-sense) cost: the sum over violated non-basic columns of
// |reduced cost| × bound range. Subtracting the gap from the vertex's
// min-sense objective gives a valid dual bound even though the vertex is not
// a proven optimum — each violated column j could improve the objective by at
// most |d_j|·(u_j−l_j) (the Neumaier–Shcherbina safe-bound shape) — so a
// primal-feasible point plus this correction is a sound node bound without a
// primal re-optimization. gap == 0 means the basis is dual-feasible (the
// dualFeasibleAt condition). finite is false when a violated column has an
// infinite bound range, in which case no finite correction exists and the
// caller must re-optimize. Costs one BTRAN plus one pricing pass.
func (s *Solver) dualInfeasGap(cost []float64) (gap float64, finite bool) {
	y := s.dualValues(cost)
	optTol := s.opts.OptimalityTol
	for j := 0; j < s.nCols; j++ {
		st := s.status[j]
		if st == ColBasic || st == ColArtificial {
			continue
		}
		if s.isArtificial[j] {
			continue
		}
		if s.upper[j]-s.lower[j] <= s.opts.FeasibilityTol {
			continue
		}
		d := s.reduce(cost, y, j)
		var viol float64
		switch st {
		case ColAtLower:
			if -d > optTol {
				viol = -d
			}
		case ColAtUpper:
			if d > optTol {
				viol = d
			}
		}
		if viol == 0 {
			continue
		}
		rng := s.upper[j] - s.lower[j]
		if math.IsInf(rng, 1) {
			return 0, false
		}
		gap += viol * rng
	}
	return gap, true
}

func (s *Solver) warmStart() *WarmStart {
	w := &WarmStart{
		basicCols:      append([]int(nil), s.basicCols...),
		nonBasicStatus: append([]ColStatus(nil), s.status...),
		upper:          append([]float64(nil), s.upper...),
	}
	if s.form != nil {
		w.flip = s.form.flip // immutable after compile — safe to share
		if s.form.ext != nil {
			w.extSenses = s.form.ext.Senses
			w.extRows = s.form.ext.Rows
		}
	}
	// Persist the factorization for child reuse. lu is read-only after
	// construction (pivots only append etas; refactor replaces the pointer),
	// so it is safe to share; etas is copied so a child's further pivots don't
	// alias a sibling's chain.
	if s.opts.PersistFactor {
		w.lu = s.lu
		w.etas = append([]etaUpdate(nil), s.etaUpdates...)
	}
	return w
}

// ---- Read-only accessors for basis-level consumers ----
//
// These expose exactly what Gomory cut generation and reduced-cost fixing
// need from the final basis, so those consumers stay outside the package
// without reaching into solver internals.

// M returns the number of rows.
func (s *Solver) M() int { return s.m }

// N returns the number of structural columns.
func (s *Solver) N() int { return s.n }

// NCols returns the total column count (structural + slack/surplus/artificial).
func (s *Solver) NCols() int { return s.nCols }

// RowSenses returns the actual post-orientation senses used by this solver's
// compiled standard form. The returned slice is immutable and may be shared.
func (s *Solver) RowSenses() []Sense {
	if s.form == nil {
		return nil
	}
	return s.form.senses
}

// RowFlips returns the row-orientation flags used by the compiled form. Warm
// reoptimization may deliberately keep a producing basis' orientation even
// when the current RHS sign differs, so callers must use this rather than
// reconstructing flips from the current RHS. The slice is immutable.
func (s *Solver) RowFlips() []bool {
	if s.form == nil {
		return nil
	}
	return s.form.flip
}

// BasicCols returns the basis assignment: element i is the column basic in
// row i. The returned slice is the solver's own — do not modify.
func (s *Solver) BasicCols() []int { return s.basicCols }

// StatusOf returns column j's simplex status.
func (s *Solver) StatusOf(j int) ColStatus {
	if j < 0 || j >= s.nCols {
		return ColAtLower
	}
	return s.status[j]
}

// XBasic returns the current value of the variable basic in row i.
func (s *Solver) XBasic(i int) float64 {
	if i < 0 || i >= s.m {
		return 0
	}
	return s.xBasic[i]
}

// BinvRow returns row i of B^{-1} (one BTRAN against the unit vector e_i).
func (s *Solver) BinvRow(i int) []float64 {
	unit := make([]float64, s.m)
	unit[i] = 1
	return s.btran(unit)
}

// ColDot returns the dot product of column j with the dense vector y.
func (s *Solver) ColDot(j int, y []float64) float64 {
	return s.cols[j].dotDense(y)
}

// XValue returns the current value of column j given the basis state.
func (s *Solver) XValue(j int) float64 {
	if j < 0 || j >= s.nCols {
		return 0
	}
	switch s.status[j] {
	case ColBasic:
		for i, bc := range s.basicCols {
			if bc == j {
				return s.xBasic[i]
			}
		}
		return 0
	case ColAtUpper:
		if math.IsInf(s.upper[j], 1) {
			return 0
		}
		return s.upper[j]
	default:
		return s.lower[j]
	}
}

// DualValues returns the simplex duals y = B^{-T} c_B for the problem's
// (max-sense) objective, consistent with the final basis.
func (s *Solver) DualValues() []float64 {
	return s.dualValues(s.cost)
}

// ReducedCost returns d_j = c_j − y·A_j for the problem's (max-sense)
// objective, against a dual vector y previously obtained from DualValues.
func (s *Solver) ReducedCost(j int, y []float64) float64 {
	return s.reduce(s.cost, y, j)
}
