package lp

import (
	"context"
	"math"
	"time"
)

// Incremental is the load-once entry point for solving many closely-related
// LPs on one constant problem body — the B&B pattern where only the RHS and
// the per-column upper bounds change between a parent node and its child.
//
// NewIncremental captures the constant body (objective, sparse columns, row
// senses) once; each Solve call supplies the node-dependent vectors. The
// standard equality form (slack/surplus/artificial layout, sign-normalized
// columns) depends only on which rows have a negative RHS, so it is compiled
// once per "flip pattern" and reused verbatim while the pattern is stable —
// on a typical B&B tree that means the per-node setup shrinks to three small
// array copies (basis template, column statuses, upper bounds) plus the RHS
// itself. When a warm-start basis is supplied, the initial identity-basis
// factorization is skipped entirely (it was computed and immediately
// overwritten by the warm basis install before), halving the
// factorizations per warm node solve.
//
// Every Solve still returns a fresh *Solver: callers hold the final basis
// (cut generation, reduced-cost fixing) after the solve returns, so state is
// never reused across solves — only immutable structure is shared.
//
// Extra rows (the MIP layer's per-subtree cut rows) are supported via
// SolveRows: a RowSet is appended after the base rows, compiled forms are
// cached per (RowSet, flip pattern), and a warm handle from a solve with a
// PREFIX of the current RowSet is extended in place — the new rows enter
// with their slack/surplus basic (possibly out of bounds) and the dual
// simplex repairs them, the textbook add-cuts-then-dual-reoptimize step.
type Incremental struct {
	c      []float64
	cols   []SparseVector
	senses []Sense
	ranges []float64
	n, m   int

	// costFull / zerosFull are the solver's cost and lower-bound vectors
	// extended with zeros to the worst-case base column count (n + 2m). They
	// are identical for every flip pattern and never written after
	// construction (phase costs are separate arrays; y-space lower bounds
	// are always 0), so every base-form solve shares one allocation of each.
	// Extended forms allocate their own (their column count exceeds this).
	costFull  []float64
	zerosFull []float64

	cached   *compiledForm   // base form for the most recent flip pattern
	extForms []*compiledForm // small MRU cache of extended forms

	// denseRescueFutileStreak counts CONSECUTIVE dense-LU redos of unstable
	// sparse factors that failed to improve the basis residual (the
	// ill-conditioning was in the basis, not the factorization). Near-
	// singularity is a per-basis property, not a model constant, so the tree
	// backs off exponentially instead of latching permanently: once the
	// streak reaches denseRescueFutilityLimit, redos are disabled for a
	// cooldown of solves that doubles with each further futile diagnosis
	// (capped), and any redo that helps resets the streak. This keeps the
	// O(m³) redo burn logarithmic on models where dense never helps
	// (neos-3754480-nidda: 601 futile redos ≈ 55% of a 30s budget under a
	// per-solve-only latch) while later bases that dense CAN fix — residual
	// instability is basis-dependent — still get rescued.
	denseRescueFutileStreak int
	denseRescueCooldown     int
}

// denseRescueFutilityLimit is how many consecutive futile dense redos the
// tree diagnoses before it starts backing off.
const denseRescueFutilityLimit = 3

// denseRescueCooldownBase/Max bound the backoff: once the futile streak
// reaches the limit, redos are disabled for base<<(streak-limit) solves.
const (
	denseRescueCooldownBase = 16
	denseRescueCooldownMax  = 4096
)

// noteDenseRescueOutcome feeds one dense-redo diagnosis into the backoff
// state. futile means the redo failed to meaningfully improve the basis
// residual (see denseRescueWasFutile).
func (inc *Incremental) noteDenseRescueOutcome(futile bool) {
	if !futile {
		inc.denseRescueFutileStreak = 0
		return
	}
	inc.denseRescueFutileStreak++
	if over := inc.denseRescueFutileStreak - denseRescueFutilityLimit; over >= 0 {
		cool := denseRescueCooldownBase << over
		if over >= 8 || cool > denseRescueCooldownMax {
			cool = denseRescueCooldownMax
		}
		inc.denseRescueCooldown = cool
	}
}

// denseRescueDisabled reports whether the next solve should skip dense
// redos, consuming one cooldown tick.
func (inc *Incremental) denseRescueDisabled() bool {
	if inc.denseRescueCooldown > 0 {
		inc.denseRescueCooldown--
		return true
	}
	return false
}

// RowSet is an immutable set of extra rows appended after the base rows —
// the per-subtree cut rows of the MIP layer. Rows[i] is a sparse vector over
// the structural columns [0, n) with entries in ascending column order. A
// RowSet must not be mutated after first use: compiled forms and warm-start
// handles keep references into it. Callers should reuse one *RowSet per
// distinct cut set (the pointer is the compiled-form cache key).
type RowSet struct {
	Senses []Sense
	Rows   []SparseVector
}

// compiledForm is the standard equality form for one (RowSet, row-sign flip
// pattern) pair. Everything here is read-only during a solve: cols entries
// are never written, isArtificial/artificials/cost/lower are only read, and
// basicCols0/status0 are templates copied into each solve's mutable state.
type compiledForm struct {
	ext          *RowSet // nil for the base form
	flip         []bool  // over base+ext rows
	senses       []Sense // post-flip, base+ext
	cols         []SparseVector
	nCols        int
	nColsBase    int // columns belonging to structural + base-row aux
	isArtificial []bool
	artificials  []int
	basicCols0   []int
	status0      []ColStatus
	cost         []float64 // max-sense cost per column (0 on aux)
	lower        []float64 // all zeros, length nCols
	rangeSlack   []int     // base ranged row -> bounded slack column, or -1
}

// NewIncremental captures a constant problem body. prob.B and prob.Upper are
// ignored — they are per-solve inputs. prob.Cols must follow the Problem
// contract (ascending row order, no explicit zeros) and must not be mutated
// for the lifetime of the Incremental.
func NewIncremental(prob Problem) *Incremental {
	n := len(prob.C)
	m := len(prob.Senses)
	inc := &Incremental{c: prob.C, cols: prob.Cols, senses: prob.Senses, ranges: prob.RangeWidths, n: n, m: m}
	maxCols := n + 2*m
	inc.costFull = make([]float64, maxCols)
	copy(inc.costFull, prob.C)
	inc.zerosFull = make([]float64, maxCols)
	return inc
}

// Solve runs the revised simplex for one node: the constant body captured at
// construction plus this node's RHS b, structural upper bounds (short or nil
// = +Inf), and optional warm-start basis. opts carries the per-solve knobs
// including Start for the TimeLimit clock.
func (inc *Incremental) Solve(ctx context.Context, opts Options, b, upper []float64, warm *WarmStart) Solution {
	return inc.SolveRows(ctx, opts, nil, b, upper, warm)
}

// SolveRows is Solve with an extra RowSet appended after the base rows: b
// covers base rows first, then one entry per ext row. ext may be nil.
func (inc *Incremental) SolveRows(ctx context.Context, opts Options, ext *RowSet, b, upper []float64, warm *WarmStart) Solution {
	k := 0
	if ext != nil {
		k = len(ext.Senses)
	}
	mTotal := inc.m + k
	if mTotal == 0 {
		return solveTrivial(inc.c, upper, opts)
	}

	// A cold solve normalizes each row so b_i >= 0. For a compatible warm
	// solve whose bounds only tightened, keep the producing basis' row
	// orientation instead. Branching changes the shifted RHS frequently; if
	// every sign crossing also rebuilds the standard-form aux layout, the
	// parent basis has to be remapped before nearly every child reoptimize.
	// Keeping the row orientation fixed matches a persistent LP: only sides
	// and bounds change, and dual simplex repairs the resulting primal
	// infeasibility. The separately retained cold form remains the fallback.
	coldBNorm := make([]float64, mTotal)
	coldFlip := make([]bool, mTotal)
	inc.normalizeRHS(b, coldBNorm, coldFlip)
	bNorm, flip := coldBNorm, coldFlip
	if inc.canKeepWarmOrientation(warm, ext, mTotal, upper) {
		bNorm = make([]float64, mTotal)
		flip = append([]bool(nil), warm.flip...)
		inc.orientRHS(b, bNorm, flip)
	}
	cf := inc.formFor(ext, flip)
	coldCF := inc.formFor(ext, coldFlip)

	// Warm-start compatibility means the handle came from a solve with the
	// same standard-form shape — same row/column counts — i.e. only
	// bounds/RHS changed. A handle whose RowSet is a strict PREFIX of the
	// current one is extended: the new rows enter with their slack/surplus
	// basic (possibly bound-violating) and the dual simplex repairs them.
	// On a match the initial factorization is deferred: applyWarmStart
	// installs and factors the warm basis directly.
	//
	// dualCertified additionally proves the handle's basis is dual-feasible
	// for THIS solve, which is what makes dual re-optimization terminate at
	// a true optimum: the standard form must have the identical row-sign
	// flip pattern (two compensating sense changes keep nCols equal while
	// shifting the aux-column layout, so the basis lands on columns with
	// different meanings), the handle's ext rows must be content-identical
	// to the current prefix, and no bound may have been relaxed since the
	// handle was produced (a non-basic column parked at its old upper may
	// now price favorably). B&B tightenings with append-only cut sets always
	// certify; when the certificate fails the reopt still runs, but its
	// Optimal is trusted only after an explicit dual-feasibility check — the
	// old behavior of accepting it blindly could return a primal-feasible
	// but suboptimal point as the node bound.
	warmUse, dualCertified := inc.classifyWarm(warm, cf, ext, flip, upper)
	warmOK := warmUse != nil
	warmFallback := warm != nil && !warmOK
	certificateCandidate := false
	s, err := inc.build(ctx, opts, cf, bNorm, upper, !warmOK)
	if err != nil {
		debugf("revised: setup failed: %v", err)
		return Solution{Status: IterationLimit}
	}
	if warm != nil && !warmOK {
		// A handle was offered but classifyWarm couldn't apply it (shape
		// mismatch, untranslatable basic artificial, non-prefix RowSet).
		s.stats.WarmFailures++
		s.stats.WarmRejected++
	}
	rebuildCold := func(prior *Solver) (*Solver, error) {
		fresh, ferr := inc.build(ctx, opts, coldCF, coldBNorm, upper, true)
		if ferr != nil {
			return nil, ferr
		}
		if prior != nil {
			fresh.iterations = prior.iterations
			fresh.stats.Add(prior.stats)
		}
		return fresh, nil
	}

	// Anti-cycling fallback: the dual simplex doesn't use Bland's rule and
	// can stall on degenerate problems. We cap dual at a fraction of the
	// global iteration budget; if it hits that ceiling (or any non-Optimal,
	// non-Infeasible status) we throw the warm-start away and cold-solve
	// instead. This preserves correctness at the cost of one wasted
	// re-optimization round, which is far better than reporting
	// IterationLimit on the whole MIP because of a single bad child.
	if warmOK {
		if warmUse.flipRemapped {
			s.stats.WarmFlipRemaps++
		}
		if err := s.applyWarmStart(warmUse, s.bNorm); err != nil {
			s.stats.WarmFailures++
			s.stats.WarmInstallFailures++
			warmFallback = true
			debugf("revised: warm-start install failed: %v — falling back to cold start", err)
			fresh, ferr := rebuildCold(s)
			if ferr != nil {
				return statusSolution(s, IterationLimit)
			}
			s = fresh
		} else {
			s.stats.WarmReopts++
			dualCap := opts.MaxIterations / 4
			if dualCap < 2000 {
				dualCap = 2000
			}
			if dualCap > opts.MaxIterations {
				dualCap = opts.MaxIterations
			}
			phaseIICost := makePhaseIICost(s)
			st := s.runDualSimplexCapped(phaseIICost, dualCap)
			switch st {
			case Optimal:
				// Dual reoptimize cannot bring an artificial back in, but
				// numerical drift can leave a basic artificial sitting at
				// a positive value after the warm-start basis is reapplied.
				// In that case the LP we're reporting Optimal on is not the
				// true LP of the problem — fall back to a cold solve.
				if _, ok := s.checkArtificialResidual(); ok {
					// The dual simplex only restores primal feasibility; its
					// Optimal is a true optimum only if the basis was (and
					// stayed) dual-feasible. Certified handles guarantee that
					// by construction; uncertified ones (shifted standard-form
					// layout, relaxed bounds) must pass an explicit
					// reduced-cost check — one BTRAN + one pricing pass —
					// before their result is trusted. Accepting it blindly
					// (the old behavior) could report a primal-feasible but
					// suboptimal point as the node's LP optimum: a weaker
					// prune bound and a wrong vertex for rounding/branching.
					// The Harris dual ratio test voids the certificate: its
					// tolerance-relaxed pivot choice can push other reduced
					// costs past zero (up to optTol per pivot, compounding),
					// so under QOPT_HARRIS the explicit check always runs.
					// The standard ratio test drifts too, just slower — the
					// certificate is therefore trusted only for certChainLimit
					// consecutive solves before the same explicit check is
					// forced and the chain restarts (see certChainLimit).
					certEligible := dualCertified && !harrisDual()
					if certEligible && (certChainLimit < 0 || warmUse.certChain < certChainLimit) {
						s.stats.WarmCertAccepts++
						sol := extractSolution(s, inc.c, opts)
						if sol.Warm != nil {
							sol.Warm.certChain = warmUse.certChain + 1
						}
						return sol
					}
					if certEligible {
						// Chain-limit verification. On failure the vertex is
						// still primal-feasible and near-optimal — the drift
						// per certified reuse is tolerance-scale — so instead
						// of a primal re-optimization (measured on glass4:
						// degenerate resumes consumed the whole node budget)
						// the objective is weakened by the dual-infeasibility
						// gap, which restores a valid bound at zero extra
						// cost. The chain keeps growing on a corrected accept
						// (the basis was NOT re-established dual-feasible), so
						// every further certified reuse re-verifies and
						// re-corrects. Only an infinite-range violation —
						// no finite correction exists — falls through to the
						// primal resume.
						s.stats.WarmCertVerifies++
						gap, finite := s.dualInfeasGap(phaseIICost)
						if gap == 0 {
							return extractSolution(s, inc.c, opts)
						}
						s.stats.WarmCertVerifyFails++
						if finite && gap <= certCorrectionCap(s, inc.c) {
							s.stats.WarmCertBoundCorrections++
							if gap > s.stats.WarmCertBoundCorrectionMax {
								s.stats.WarmCertBoundCorrectionMax = gap
							}
							sol := extractSolution(s, inc.c, opts)
							// phaseIICost is the negated (min-sense) objective,
							// so the min-sense correction raises the max-sense
							// bound.
							sol.Objective += gap
							if sol.Warm != nil {
								sol.Warm.certChain = warmUse.certChain + 1
							}
							return sol
						}
						// Infinite-range violation, or a correction so large
						// the weakened bound would be useless (measured on
						// gen-ip002/timtab1: pathological single corrections
						// on the objective's own scale dragged the global
						// dual bound and its gap far below the resume-based
						// reference): re-optimize for the true bound instead.
						s.stats.WarmCertVerifyResumes++
					} else if s.dualFeasibleAt(phaseIICost) {
						return extractSolution(s, inc.c, opts)
					}
					// Dual-infeasible: the point is primal-feasible, so the
					// cheap sound completion is to resume the PRIMAL simplex
					// from here (no Phase I needed) — pin the artificials
					// first, exactly as a cold Phase I exit would.
					debugf("revised: dual reoptimize Optimal but basis is dual-infeasible (uncertified warm handle) — resuming primal Phase II")
					for _, j := range s.artificials {
						s.upper[j] = 0
						if s.status[j] != ColBasic {
							s.status[j] = ColArtificial
						}
					}
					t0 := time.Now()
					st2 := s.runSimplex(phaseIICost, false)
					s.stats.PhaseIINanos += time.Since(t0).Nanoseconds()
					switch st2 {
					case Optimal:
						if _, ok := s.checkArtificialResidual(); ok {
							return extractSolution(s, inc.c, opts)
						}
						debugf("revised: primal resume Optimal but artificials drifted, cold-solving")
					case TimeLimit:
						return statusSolution(s, TimeLimit)
					case Unbounded:
						return statusSolution(s, Unbounded)
					}
					// IterationLimit or drift: fall through to the cold rebuild.
				} else {
					s.stats.WarmArtificialFailure++
					debugf("revised: dual reoptimize Optimal but artificials drifted, cold-solving")
				}
			case TimeLimit:
				return statusSolution(s, TimeLimit)
			}
			if st == Infeasible && s.dualInfeasibilityCertified {
				s.stats.WarmInfeasibleCandidates++
				if !infeasibilityCertificateShadow {
					s.stats.WarmDualInfeasible++
					s.stats.WarmInfeasibleCertificates++
					sol := statusSolution(s, Infeasible)
					sol.InfeasibilityCertified = true
					sol.FarkasProof = s.farkasProof
					return sol
				}
				certificateCandidate = true
			}
			// Dual reoptimize returned either Infeasible or IterationLimit
			// (stalled). Either status from the warm-start path can be
			// numerical noise — a degenerate LP after bound-tightening can
			// fool the dual ratio test into reporting infeasibility, or the
			// simplex can cycle. Cold Phase I from a fresh basis is the
			// correctness reference, so we always rebuild and re-solve.
			// The mistakenly-pruned-subtree class on neos-3754480-nidda
			// (qopt reporting Optimal at obj=29009 vs CBC's reachable 13243)
			// disappears entirely when this fallback is unconditional.
			if st == Infeasible {
				s.stats.WarmDualInfeasible++
				debugf("revised: dual reoptimize reported Infeasible from warm-start; verifying via cold solve")
			} else {
				s.stats.WarmDualIterationCaps++
				debugf("revised: dual reoptimize stalled after %d iters, falling back to cold start", s.iterations)
			}
			s.stats.WarmFailures++
			warmFallback = true
			fresh, ferr := rebuildCold(s)
			if ferr != nil {
				return statusSolution(s, IterationLimit)
			}
			s = fresh
		}
	}

	s.stats.ColdStarts++
	if st := s.phaseI(); st != Optimal {
		if certificateCandidate {
			s.stats.WarmInfeasibleChecks++
			if st == Infeasible {
				s.stats.WarmInfeasibleCertificates++
			} else {
				s.stats.WarmInfeasibleInconclusive++
			}
		}
		if warmFallback && st == Infeasible {
			s.stats.WarmColdInfeasible++
		}
		sol := statusSolution(s, st)
		if st == Infeasible {
			sol.Infeasibility = s.infeasibilityCertificate()
		}
		return sol
	}
	st := s.phaseII()
	if certificateCandidate {
		s.stats.WarmInfeasibleChecks++
		if st == Optimal {
			s.stats.WarmInfeasibleContradicted++
		} else {
			s.stats.WarmInfeasibleInconclusive++
		}
	}
	if st != Optimal {
		return statusSolution(s, st)
	}
	return extractSolution(s, inc.c, opts)
}

// compile builds the standard equality form for one flip pattern: one slack
// per LE row, one surplus per GE row, one artificial per GE/EQ row (senses
// AFTER the sign flip). Structural columns touching a flipped row get a
// negated copy of their values; all other columns share the body's storage
// (idx is always shared — nothing writes column entries after construction).
func (inc *Incremental) compile(flip []bool) *compiledForm {
	n, m := inc.n, inc.m

	senses := make([]Sense, m)
	anyFlip := false
	for i := 0; i < m; i++ {
		senses[i] = inc.senses[i]
		if flip[i] {
			anyFlip = true
			if !inc.isRange(i) {
				senses[i] = flipSense(senses[i])
			}
		}
	}

	nSlack, nArtificial := 0, 0
	for i := 0; i < m; i++ {
		if inc.isRange(i) {
			nSlack++
			nArtificial++
			continue
		}
		switch senses[i] {
		case LessOrEqual:
			nSlack++
		case GreaterOrEqual:
			nSlack++
			nArtificial++
		case Equal:
			nArtificial++
		}
	}
	nCols := n + nSlack + nArtificial

	cols := make([]SparseVector, nCols)
	for j := 0; j < n; j++ {
		src := inc.cols[j]
		if !anyFlip {
			cols[j] = src
			continue
		}
		needFlip := false
		for _, i := range src.idx {
			if flip[i] {
				needFlip = true
				break
			}
		}
		if !needFlip {
			cols[j] = src
			continue
		}
		vals := make([]float64, len(src.val))
		for k, i := range src.idx {
			if flip[i] {
				vals[k] = -src.val[k]
			} else {
				vals[k] = src.val[k]
			}
		}
		cols[j] = SparseVector{idx: src.idx, val: vals}
	}

	basicCols0 := make([]int, m)
	isArtificial := make([]bool, nCols)
	rangeSlack := make([]int, m)
	for i := range rangeSlack {
		rangeSlack[i] = -1
	}
	var artificials []int

	next := n
	for i := 0; i < m; i++ {
		if inc.isRange(i) {
			cols[next] = SparseVector{idx: []int{i}, val: []float64{1}}
			rangeSlack[i] = next
			next++
			cols[next] = SparseVector{idx: []int{i}, val: []float64{1}}
			isArtificial[next] = true
			artificials = append(artificials, next)
			basicCols0[i] = next
			next++
			continue
		}
		switch senses[i] {
		case LessOrEqual:
			cols[next] = SparseVector{idx: []int{i}, val: []float64{1}}
			basicCols0[i] = next
			next++
		case GreaterOrEqual:
			cols[next] = SparseVector{idx: []int{i}, val: []float64{-1}}
			next++
			cols[next] = SparseVector{idx: []int{i}, val: []float64{1}}
			isArtificial[next] = true
			artificials = append(artificials, next)
			basicCols0[i] = next
			next++
		case Equal:
			cols[next] = SparseVector{idx: []int{i}, val: []float64{1}}
			isArtificial[next] = true
			artificials = append(artificials, next)
			basicCols0[i] = next
			next++
		}
	}

	status0 := make([]ColStatus, nCols)
	for _, b := range basicCols0 {
		status0[b] = ColBasic
	}

	return &compiledForm{
		flip:         append([]bool(nil), flip...),
		senses:       senses,
		cols:         cols,
		nCols:        nCols,
		nColsBase:    nCols,
		isArtificial: isArtificial,
		artificials:  artificials,
		basicCols0:   basicCols0,
		status0:      status0,
		cost:         inc.costFull[:nCols],
		lower:        inc.zerosFull[:nCols],
		rangeSlack:   rangeSlack,
	}
}

// formFor returns the compiled standard form for (ext, flip), consulting the
// base cache (ext == nil) or a small MRU cache of extended forms. The RowSet
// pointer is the cache key — callers reuse one *RowSet per distinct cut set.
func (inc *Incremental) formFor(ext *RowSet, flip []bool) *compiledForm {
	if ext == nil || len(ext.Senses) == 0 {
		cf := inc.cached
		if cf == nil || !boolSliceEqual(cf.flip, flip) {
			cf = inc.compile(flip)
			inc.cached = cf
		}
		return cf
	}
	for i, cf := range inc.extForms {
		if cf.ext == ext && boolSliceEqual(cf.flip, flip) {
			if i != 0 { // move to front
				copy(inc.extForms[1:i+1], inc.extForms[:i])
				inc.extForms[0] = cf
			}
			return cf
		}
	}
	cf := inc.compileExt(ext, flip)
	const maxExtForms = 4
	if len(inc.extForms) < maxExtForms {
		inc.extForms = append(inc.extForms, nil)
	}
	copy(inc.extForms[1:], inc.extForms)
	inc.extForms[0] = cf
	return cf
}

// compileExt builds the standard form for the base body plus ext's rows. The
// base part (structural columns with base flips applied, base aux columns,
// base templates) is shared with the base compiled form; structural columns
// touched by an ext row get a copy with the ext entries appended (ext row i
// lives at matrix row m_base+i, so appended entries keep ascending order),
// and the ext rows' aux columns follow the base aux block in row order —
// the same layout revisedAuxColumnLayout derives from the full sense list.
func (inc *Incremental) compileExt(ext *RowSet, flip []bool) *compiledForm {
	mBase := inc.m
	k := len(ext.Senses)
	base := inc.formFor(nil, flip[:mBase])

	senses := make([]Sense, mBase+k)
	copy(senses, base.senses)
	for i := 0; i < k; i++ {
		s := ext.Senses[i]
		if flip[mBase+i] {
			s = flipSense(s)
		}
		senses[mBase+i] = s
	}

	nSlack, nArtificial := 0, 0
	for i := mBase; i < mBase+k; i++ {
		switch senses[i] {
		case LessOrEqual:
			nSlack++
		case GreaterOrEqual:
			nSlack++
			nArtificial++
		case Equal:
			nArtificial++
		}
	}
	nCols := base.nCols + nSlack + nArtificial

	cols := make([]SparseVector, nCols)
	copy(cols, base.cols)
	// Append ext entries to the structural columns they touch, in row order.
	for i := 0; i < k; i++ {
		row := ext.Rows[i]
		matrixRow := mBase + i
		neg := flip[matrixRow]
		for e, j := range row.idx {
			v := row.val[e]
			if neg {
				v = -v
			}
			src := cols[j]
			// Copy-on-extend the first time this column is touched (it still
			// aliases the shared base column); afterwards it is private and
			// can be appended to directly.
			if len(src.idx) == len(base.cols[j].idx) {
				idx := make([]int, len(src.idx), len(src.idx)+k)
				val := make([]float64, len(src.val), len(src.val)+k)
				copy(idx, src.idx)
				copy(val, src.val)
				src = SparseVector{idx: idx, val: val}
			}
			src.idx = append(src.idx, matrixRow)
			src.val = append(src.val, v)
			cols[j] = src
		}
	}

	basicCols0 := make([]int, mBase+k)
	copy(basicCols0, base.basicCols0)
	isArtificial := make([]bool, nCols)
	copy(isArtificial, base.isArtificial)
	artificials := append([]int(nil), base.artificials...)
	rangeSlack := make([]int, mBase+k)
	for i := range rangeSlack {
		rangeSlack[i] = -1
	}
	copy(rangeSlack, base.rangeSlack)

	next := base.nCols
	for i := 0; i < k; i++ {
		matrixRow := mBase + i
		switch senses[matrixRow] {
		case LessOrEqual:
			cols[next] = SparseVector{idx: []int{matrixRow}, val: []float64{1}}
			basicCols0[matrixRow] = next
			next++
		case GreaterOrEqual:
			cols[next] = SparseVector{idx: []int{matrixRow}, val: []float64{-1}}
			next++
			cols[next] = SparseVector{idx: []int{matrixRow}, val: []float64{1}}
			isArtificial[next] = true
			artificials = append(artificials, next)
			basicCols0[matrixRow] = next
			next++
		case Equal:
			cols[next] = SparseVector{idx: []int{matrixRow}, val: []float64{1}}
			isArtificial[next] = true
			artificials = append(artificials, next)
			basicCols0[matrixRow] = next
			next++
		}
	}

	status0 := make([]ColStatus, nCols)
	for _, b := range basicCols0 {
		status0[b] = ColBasic
	}

	cost := make([]float64, nCols)
	copy(cost, inc.c)
	lower := make([]float64, nCols)

	return &compiledForm{
		ext:          ext,
		flip:         append([]bool(nil), flip...),
		senses:       senses,
		cols:         cols,
		nCols:        nCols,
		nColsBase:    base.nCols,
		isArtificial: isArtificial,
		artificials:  artificials,
		basicCols0:   basicCols0,
		status0:      status0,
		cost:         cost,
		lower:        lower,
		rangeSlack:   rangeSlack,
	}
}

// build assembles a fresh Solver for one solve on the compiled form. Only the
// mutable per-solve state is allocated (basis, statuses, upper bounds, x_B);
// columns, costs, lower bounds, and the artificial bookkeeping are shared
// read-only with the compiled form. factorNow controls whether the initial
// basis is factorized here — false when a warm basis will be installed
// immediately (applyWarmStart factors the warm basis itself, so factoring
// the identity basis first would be pure waste).
func (inc *Incremental) build(ctx context.Context, opts Options, cf *compiledForm, bNorm, structUpper []float64, factorNow bool) (*Solver, error) {
	n, m, nCols := inc.n, len(cf.senses), cf.nCols

	upper := make([]float64, nCols)
	for j := 0; j < n; j++ {
		if j < len(structUpper) {
			upper[j] = structUpper[j]
		} else {
			upper[j] = math.Inf(1)
		}
	}
	for j := n; j < nCols; j++ {
		upper[j] = math.Inf(1)
	}
	for i, j := range cf.rangeSlack {
		if j >= 0 && i < len(inc.ranges) {
			upper[j] = inc.ranges[i]
		}
	}

	xBasic := make([]float64, m)
	copy(xBasic, bNorm)

	s := &Solver{
		ctx: ctx, opts: opts, start: opts.Start,
		m: m, n: n, nCols: nCols,
		form:         cf,
		cols:         cf.cols,
		cost:         cf.cost,
		lower:        cf.lower,
		upper:        upper,
		bNorm:        bNorm,
		basicCols:    append([]int(nil), cf.basicCols0...),
		status:       append([]ColStatus(nil), cf.status0...),
		xBasic:       xBasic,
		isArtificial: cf.isArtificial,
		artificials:  cf.artificials,
		stats: Stats{
			LPSolves: 1,
		},
	}
	s.initScratch()
	if inc.denseRescueDisabled() {
		s.denseRescueFutile = true
	}
	s.onDenseRescueOutcome = inc.noteDenseRescueOutcome
	if factorNow {
		if err := s.refactor(); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// resetToInitialBasis restores the identity slack/artificial basis after a
// failed warm-start install on a deferred-factor solver, so the cold Phase
// I/II fallback runs on a consistent state.
func (s *Solver) resetToInitialBasis(cf *compiledForm, bNorm []float64) error {
	copy(s.basicCols, cf.basicCols0)
	copy(s.status, cf.status0)
	s.etaUpdates = nil
	copy(s.xBasic, bNorm)
	return s.refactor()
}

// certCorrectionCap is the largest dual-infeasibility gap the verify-failure
// repair will absorb as a bound weakening rather than re-optimizing:
// certCorrectionRelTol relative to the vertex objective's magnitude.
// Tolerance-scale drift (the common case, and all of glass4's, where the
// degenerate primal resume is unaffordable) stays on the cheap corrected
// accept; a basis whose gap is on the objective's own scale produces a
// uselessly weak bound and pays the resume for the true one.
func certCorrectionCap(s *Solver, c []float64) float64 {
	obj := 0.0
	for j := 0; j < s.n; j++ {
		obj += c[j] * s.XValue(j)
	}
	return certCorrectionRelTol * math.Max(1, math.Abs(obj))
}

func makePhaseIICost(s *Solver) []float64 {
	cost := make([]float64, s.nCols)
	for j := 0; j < s.n; j++ {
		cost[j] = -s.cost[j]
	}
	return cost
}

func extractSolution(s *Solver, c []float64, opts Options) Solution {
	values := make([]float64, s.n)
	obj := 0.0
	for j := 0; j < s.n; j++ {
		v := s.XValue(j)
		values[j] = cleanZero(v, opts.FeasibilityTol)
		obj += c[j] * v
	}
	if debugEnabled() {
		maxViol := 0.0
		for j := 0; j < s.n; j++ {
			if values[j] < -maxViol {
				maxViol = -values[j]
			}
			if j < len(s.upper) && !math.IsInf(s.upper[j], 1) && values[j]-s.upper[j] > maxViol {
				maxViol = values[j] - s.upper[j]
			}
		}
		if maxViol > opts.FeasibilityTol*10 {
			debugf("revised: Optimal point violates a structural bound by %g", maxViol)
		}
	}
	obj = cleanZero(obj, opts.FeasibilityTol)
	return Solution{
		Status:     Optimal,
		Objective:  obj,
		Values:     values,
		Iterations: s.iterations,
		Warm:       s.warmStart(),
		Solver:     s,
		Stats:      s.stats,
	}
}

func statusSolution(s *Solver, status Status) Solution {
	if s == nil {
		return Solution{Status: status}
	}
	return Solution{Status: status, Iterations: s.iterations, Stats: s.stats}
}

// solveTrivial handles the m==0 case: each variable independently settles at
// one of its bounds depending on the sign of its cost.
func solveTrivial(c, upper []float64, opts Options) Solution {
	n := len(c)
	values := make([]float64, n)
	obj := 0.0
	stats := Stats{LPSolves: 1}
	for j := 0; j < n; j++ {
		ub := math.Inf(1)
		if j < len(upper) {
			ub = upper[j]
		}
		switch {
		case c[j] > opts.OptimalityTol:
			if math.IsInf(ub, 1) {
				return Solution{Status: Unbounded, Stats: stats}
			}
			values[j] = ub
			obj += c[j] * ub
		case c[j] < -opts.OptimalityTol:
			values[j] = 0
		}
	}
	return Solution{Status: Optimal, Values: values, Objective: obj, Stats: stats}
}

// Solve is the one-shot entry point: load prob's body and solve it once with
// prob.B / prob.Upper. Equivalent to NewIncremental(prob).Solve(...); B&B
// callers that re-solve one body many times should hold the Incremental.
func Solve(ctx context.Context, prob Problem, opts Options, warm *WarmStart) Solution {
	return NewIncremental(prob).Solve(ctx, opts, prob.B, prob.Upper, warm)
}

// newSolver builds a fresh, factored solver for prob from a cold start — the
// setup path of a cold one-shot Solve, exposed for tests that poke solver
// internals directly.
func newSolver(ctx context.Context, prob Problem, opts Options) (*Solver, error) {
	inc := NewIncremental(prob)
	if inc.m == 0 {
		return nil, errSingularBasis
	}
	bNorm := make([]float64, inc.m)
	flip := make([]bool, inc.m)
	inc.normalizeRHS(prob.B, bNorm, flip)
	return inc.build(ctx, opts, inc.compile(flip), bNorm, prob.Upper, true)
}

func (inc *Incremental) isRange(row int) bool {
	return row >= 0 && row < len(inc.ranges) && !math.IsInf(inc.ranges[row], 1) && !math.IsNaN(inc.ranges[row]) && inc.ranges[row] >= 0
}

func (inc *Incremental) normalizeRHS(b, normalized []float64, flip []bool) {
	for i := range normalized {
		normalized[i] = b[i]
		if i < inc.m && inc.isRange(i) {
			if normalized[i] < 0 {
				normalized[i] = inc.ranges[i] - normalized[i]
				flip[i] = true
			}
			continue
		}
		if normalized[i] < 0 {
			normalized[i] = -normalized[i]
			flip[i] = true
		}
	}
}

// orientRHS expresses b in an existing compiled row orientation. Unlike
// normalizeRHS, the selected flip pattern is retained even when the new RHS
// crosses zero, so a compatible warm basis keeps the same column layout.
func (inc *Incremental) orientRHS(b, oriented []float64, flip []bool) {
	for i := range oriented {
		oriented[i] = b[i]
		if !flip[i] {
			continue
		}
		if i < inc.m && inc.isRange(i) {
			oriented[i] = inc.ranges[i] - b[i]
		} else {
			oriented[i] = -b[i]
		}
	}
}

// canKeepWarmOrientation limits stable row orientation to the same row-slot
// layout and monotonically tightened structural bounds. Row coefficients may
// change (cut aging/reactivation); classifyWarm then forces a refactor and an
// explicit dual-feasibility check. Relaxed bounds and row extensions continue
// through the general remap/cold paths.
func (inc *Incremental) canKeepWarmOrientation(warm *WarmStart, ext *RowSet, mTotal int, upper []float64) bool {
	if warm == nil || len(warm.flip) != mTotal || len(warm.basicCols) != mTotal {
		return false
	}
	k := len(warm.extRows)
	if ext == nil {
		if k != 0 {
			return false
		}
	} else if k != len(ext.Rows) {
		return false
	}
	return rowSetSensesEqual(warm.extSenses, ext, k) &&
		warmUppersNotRelaxed(warm, upper, inc.n)
}

// auxLayout returns, for each row of the given post-flip sense list, the
// column indices of its slack/surplus (logical) and artificial columns
// (−1 when the row has none), following the sequential assignment compile
// and compileExt use. nCols is the resulting total column count.
func auxLayout(n int, senses []Sense) (logical, artificial []int, nCols int) {
	m := len(senses)
	logical = make([]int, m)
	artificial = make([]int, m)
	next := n
	for i := 0; i < m; i++ {
		logical[i], artificial[i] = -1, -1
		switch senses[i] {
		case LessOrEqual:
			logical[i] = next
			next++
		case GreaterOrEqual:
			logical[i] = next
			next++
			artificial[i] = next
			next++
		case Equal:
			artificial[i] = next
			next++
		}
	}
	return logical, artificial, next
}

func (inc *Incremental) auxLayoutWithRanges(senses []Sense) (logical, artificial []int, nCols int) {
	m := len(senses)
	logical = make([]int, m)
	artificial = make([]int, m)
	next := inc.n
	for i := 0; i < m; i++ {
		logical[i], artificial[i] = -1, -1
		if i < inc.m && inc.isRange(i) {
			logical[i] = next
			next++
			artificial[i] = next
			next++
			continue
		}
		switch senses[i] {
		case LessOrEqual:
			logical[i] = next
			next++
		case GreaterOrEqual:
			logical[i] = next
			next++
			artificial[i] = next
			next++
		case Equal:
			artificial[i] = next
			next++
		}
	}
	return logical, artificial, next
}

// remapWarmFlips translates a warm handle produced under a different row-sign
// flip pattern into cf's column layout. The two standard forms differ only by
// a row-sign diagonal D: every structural column, and the logical column of
// each row (slack under one flip ≡ surplus under the other, same mathematical
// variable), maps 1:1, so the remapped basis matrix is D·B_parent — exactly as
// nonsingular and dual-feasible as the parent's basis (reduced costs are
// invariant under row scaling). The one case with no target is a parent
// artificial that is BASIC on a row whose LE↔GE identity changed (the new
// layout has no artificial there); we bail to a cold start — artificials are
// basic at optimum only degenerately.
//
// sensesW must be the warm handle's post-flip sense list; cf carries the
// current one. Returns nil when the handle cannot be translated.
func (inc *Incremental) remapWarmFlips(warm *WarmStart, cf *compiledForm, sensesW []Sense, flip []bool, ext *RowSet) *WarmStart {
	n := inc.n
	mTotal := len(cf.senses)

	logW, artW, nColsW := inc.auxLayoutWithRanges(sensesW)
	if len(warm.nonBasicStatus) != nColsW {
		return nil
	}
	logC, artC, _ := inc.auxLayoutWithRanges(cf.senses)

	// tr[jW] = column in cf's layout carrying the same mathematical variable,
	// or -1 for a dropped artificial.
	tr := make([]int, nColsW)
	for j := 0; j < n; j++ {
		tr[j] = j
	}
	for i := 0; i < mTotal; i++ {
		if logW[i] >= 0 {
			tr[logW[i]] = logC[i] // slack ↔ surplus across a flip: same variable
		}
		if artW[i] >= 0 {
			tr[artW[i]] = artC[i] // -1 when the target row has no artificial
		}
	}

	st := make([]ColStatus, cf.nCols)
	// Artificials that exist only in the target layout start out of the
	// basis and excluded from pricing, exactly like a post-Phase-I state.
	for i := 0; i < mTotal; i++ {
		if artC[i] >= 0 && artW[i] < 0 {
			st[artC[i]] = ColArtificial
		}
	}
	for jW, jC := range tr {
		if jC >= 0 {
			st[jC] = warm.nonBasicStatus[jW]
		}
	}
	for i := 0; i < inc.m; i++ {
		if !inc.isRange(i) || warm.flip[i] == flip[i] || logC[i] < 0 || st[logC[i]] == ColBasic {
			continue
		}
		switch st[logC[i]] {
		case ColAtLower:
			st[logC[i]] = ColAtUpper
		case ColAtUpper:
			st[logC[i]] = ColAtLower
		}
	}

	bc := make([]int, mTotal)
	for r := 0; r < mTotal; r++ {
		nb := tr[warm.basicCols[r]]
		if nb < 0 {
			return nil // basic artificial with no column in the new layout
		}
		bc[r] = nb
	}

	return &WarmStart{
		basicCols:      bc,
		nonBasicStatus: st,
		upper:          warm.upper,
		flip:           append([]bool(nil), flip...),
		extSenses:      extSensesOf(ext),
		extRows:        extRowsOf(ext),
	}
}

// warmSenses reconstructs the post-flip sense list the warm handle's compiled
// form used: base senses from the body, ext senses from the handle's own row
// provenance, each flipped per the handle's recorded pattern.
func (inc *Incremental) warmSenses(warm *WarmStart) []Sense {
	mTotal := inc.m + len(warm.extSenses)
	if len(warm.flip) != mTotal {
		return nil
	}
	senses := make([]Sense, mTotal)
	for i := 0; i < inc.m; i++ {
		senses[i] = inc.senses[i]
	}
	copy(senses[inc.m:], warm.extSenses)
	for i := range senses {
		if warm.flip[i] && !(i < inc.m && inc.isRange(i)) {
			senses[i] = flipSense(senses[i])
		}
	}
	return senses
}

// classifyWarm decides how (and whether) a warm handle applies to the
// current form. Returns the handle to install — the original for an exact
// content match, a refactor-only retarget for same-sense row-slot changes, a
// flip-remapped translation when the row-sign pattern drifted, a synthesized
// extension when the handle's RowSet is a strict
// content-prefix of cf's, nil when unusable — plus whether dual feasibility
// of the installed basis is certified (see the SolveRows comment).
func (inc *Incremental) classifyWarm(warm *WarmStart, cf *compiledForm, ext *RowSet, flip []bool, upper []float64) (*WarmStart, bool) {
	if warm == nil {
		return nil, false
	}
	mTotal := len(cf.senses)

	// Exact shape + flip match. If row coefficients changed but senses did not,
	// retain the basis indices while dropping the persisted factorization: the
	// current matrix is refactored on install and the verification pass in
	// SolveRows keeps the uncertified result sound.
	if len(warm.basicCols) == mTotal && len(warm.nonBasicStatus) == cf.nCols &&
		boolSliceEqual(warm.flip, flip) && rowSetSensesEqual(warm.extSenses, ext, len(warm.extSenses)) {
		sameRows := rowSetPrefixEqual(warm.extSenses, warm.extRows, ext, len(warm.extRows))
		if sameRows {
			return warm, warmUppersNotRelaxed(warm, upper, inc.n)
		}
		return inc.retargetWarmRows(warm, cf, ext, flip), false
	}

	// Same row slots, drifted flip pattern (branching shifted some RHS entries
	// across zero): translate the handle into cf's layout. Coefficients may also
	// have changed through cut aging/reactivation; remapWarmFlips always drops
	// the old factorization, and we deliberately return it UNCERTIFIED so
	// SolveRows' verification pass vouches for the basis before trusting it.
	extLen := 0
	if ext != nil {
		extLen = len(ext.Senses)
	}
	if len(warm.basicCols) == mTotal && len(warm.extSenses) == extLen &&
		rowSetSensesEqual(warm.extSenses, ext, len(warm.extSenses)) {
		if sensesW := inc.warmSenses(warm); sensesW != nil {
			if remapped := inc.remapWarmFlips(warm, cf, sensesW, flip, ext); remapped != nil {
				inc.anchorEmptyExtRows(remapped, cf, warm.extRows, ext)
				remapped.flipRemapped = true
				return remapped, false
			}
		}
		return nil, false
	}

	// Prefix extension: the handle came from a solve on the same base body
	// with fewer (or zero) ext rows. Requires the flip prefix and the ext
	// content prefix to match exactly — otherwise the parent's column
	// indices would not mean the same thing in cf's layout.
	kw := len(warm.extRows)
	mw := inc.m + kw
	if ext == nil || mw >= mTotal || len(warm.basicCols) != mw {
		return nil, false
	}
	if !boolSliceEqual(warm.flip, flip[:mw]) {
		return nil, false
	}
	if !rowSetPrefixEqual(warm.extSenses, warm.extRows, ext, kw) {
		return nil, false
	}
	// Column count of the handle's form = base columns + aux of its ext
	// prefix; derivable from cf because the flip prefix matches.
	nColsPrefix := cf.nColsBase
	for i := inc.m; i < mw; i++ {
		switch cf.senses[i] {
		case LessOrEqual:
			nColsPrefix++
		case GreaterOrEqual:
			nColsPrefix += 2
		case Equal:
			nColsPrefix++
		}
	}
	if len(warm.nonBasicStatus) != nColsPrefix {
		return nil, false
	}
	// New rows must be inequalities: they enter with slack/surplus basic. An
	// EQ row would need its artificial basic — a Phase I state the dual
	// reoptimize can't repair — so bail to a cold solve instead.
	for i := mw; i < mTotal; i++ {
		if cf.senses[i] == Equal {
			return nil, false
		}
	}

	bc := make([]int, mTotal)
	copy(bc, warm.basicCols)
	st := make([]ColStatus, cf.nCols)
	copy(st, warm.nonBasicStatus)
	next := nColsPrefix
	for i := mw; i < mTotal; i++ {
		switch cf.senses[i] {
		case LessOrEqual:
			bc[i] = next
			st[next] = ColBasic
			next++
		case GreaterOrEqual:
			bc[i] = next
			st[next] = ColBasic
			next++
			st[next] = ColArtificial // never enters pricing
			next++
		}
	}
	extended := &WarmStart{
		basicCols:      bc,
		nonBasicStatus: st,
		upper:          warm.upper,
		flip:           flip,
		extSenses:      extSensesOf(ext),
		extRows:        extRowsOf(ext),
		// The extension inherits the source basis' unverified-accept history:
		// appending certified rows doesn't re-establish dual feasibility.
		certChain: warm.certChain,
	}
	// Appending rows with their slack/surplus basic preserves dual
	// feasibility (the new basic columns cost 0 and the new duals are 0), so
	// the extension is certified whenever no bound was relaxed.
	certified := warmUppersNotRelaxed(warm, upper, inc.n)
	return extended, certified
}

// retargetWarmRows carries basis column identities across a same-shape matrix
// change. The old factorization cannot be reused because structural columns
// changed, so lu/etas are intentionally omitted and applyWarmStart refactors
// the basis against the current compiled form.
func (inc *Incremental) retargetWarmRows(warm *WarmStart, cf *compiledForm, ext *RowSet, flip []bool) *WarmStart {
	retargeted := &WarmStart{
		basicCols:      append([]int(nil), warm.basicCols...),
		nonBasicStatus: append([]ColStatus(nil), warm.nonBasicStatus...),
		upper:          warm.upper,
		flip:           append([]bool(nil), flip...),
		extSenses:      extSensesOf(ext),
		extRows:        extRowsOf(ext),
	}
	inc.anchorEmptyExtRows(retargeted, cf, warm.extRows, ext)
	return retargeted
}

// anchorEmptyExtRows repairs the obvious singularity created when an active
// cut row is replaced by an empty slot while one of its structural columns is
// basic. The row's private slack/surplus column is guaranteed nonzero only in
// that row, so putting it into the corresponding basis position makes the
// empty row independently pivotable. If that logical column was basic in a
// different position, swap the displaced basis column there to preserve the
// basis set. A remaining numerical singularity is still caught by refactor.
func (inc *Incremental) anchorEmptyExtRows(warm *WarmStart, cf *compiledForm, oldRows []SparseVector, ext *RowSet) {
	if warm == nil || ext == nil || len(oldRows) != len(ext.Rows) {
		return
	}
	logical, _, _ := inc.auxLayoutWithRanges(cf.senses)
	for i := range ext.Rows {
		if ext.Rows[i].Len() != 0 || oldRows[i].Len() == 0 {
			continue
		}
		r := inc.m + i
		if r >= len(logical) || r >= len(warm.basicCols) {
			continue
		}
		target := logical[r]
		if target < 0 || target >= len(warm.nonBasicStatus) {
			continue
		}
		old := warm.basicCols[r]
		if old == target {
			continue
		}
		other := -1
		for q, col := range warm.basicCols {
			if col == target {
				other = q
				break
			}
		}
		if other >= 0 {
			warm.basicCols[other] = old
		} else if old >= 0 && old < len(warm.nonBasicStatus) {
			if cf.isArtificial[old] {
				warm.nonBasicStatus[old] = ColArtificial
			} else {
				warm.nonBasicStatus[old] = ColAtLower
			}
		}
		warm.basicCols[r] = target
		warm.nonBasicStatus[target] = ColBasic
	}
}

func extSensesOf(ext *RowSet) []Sense {
	if ext == nil {
		return nil
	}
	return ext.Senses
}

func extRowsOf(ext *RowSet) []SparseVector {
	if ext == nil {
		return nil
	}
	return ext.Rows
}

// rowSetPrefixEqual reports whether (senses, rows) — a warm handle's ext
// provenance — is content-identical to the first k rows of ext. Content
// comparison (exact float equality) because RowSet objects are rebuilt per
// cut-set by the caller; identical cut sets produce identical transforms.
func rowSetPrefixEqual(senses []Sense, rows []SparseVector, ext *RowSet, k int) bool {
	if len(senses) != k || len(rows) != k {
		return false
	}
	if k == 0 {
		return true
	}
	if ext == nil || len(ext.Senses) < k {
		return false
	}
	for i := 0; i < k; i++ {
		if senses[i] != ext.Senses[i] {
			return false
		}
		a, b := rows[i], ext.Rows[i]
		if len(a.idx) != len(b.idx) {
			return false
		}
		for e := range a.idx {
			if a.idx[e] != b.idx[e] || a.val[e] != b.val[e] {
				return false
			}
		}
	}
	return true
}

func rowSetSensesEqual(senses []Sense, ext *RowSet, k int) bool {
	if len(senses) != k {
		return false
	}
	if k == 0 {
		return ext == nil || len(ext.Senses) == 0
	}
	if ext == nil || len(ext.Senses) != k {
		return false
	}
	for i := 0; i < k; i++ {
		if senses[i] != ext.Senses[i] {
			return false
		}
	}
	return true
}

// warmUppersNotRelaxed reports whether every structural column's upper bound
// is at most what it was when the warm handle was produced. Only structural
// columns are compared: slack/surplus uppers are +Inf on both sides, and
// artificial uppers legitimately differ (the producing solve pins them to 0
// in Phase I; a warm re-solve keeps them out of the basis via their status
// instead).
func warmUppersNotRelaxed(warm *WarmStart, structUpper []float64, n int) bool {
	if len(warm.upper) < n {
		return false
	}
	for j := 0; j < n; j++ {
		u := math.Inf(1)
		if j < len(structUpper) {
			u = structUpper[j]
		}
		if u > warm.upper[j] {
			return false
		}
	}
	return true
}

func boolSliceEqual(a, b []bool) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
