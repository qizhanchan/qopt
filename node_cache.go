package qopt

import (
	"context"
	"math"
	"time"

	"github.com/qizhanchan/qopt/lp"
)

// canonCache holds the bound-independent canonical + scaled standard form for
// one Model under the revised backend, so per-B&B-node LP solves can skip
// rebuilding the constraint matrix and recomputing Ruiz scaling on every call.
//
// The benchmark profile of a small MIPLIB instance (gen-ip002) showed the
// actual dual-simplex work is only ~10% of per-node CPU; the rest is the LP
// *rebuild* — canonicalize allocating a fresh dense matrix and iterating every
// constraint's coefficient map, plus an 8-sweep Ruiz pass — repeated at every
// node even though the only thing that changes between a parent and its child
// is one variable bound. Ruiz factors depend solely on the matrix coefficients
// (not on b or c), and in the reduced y = x − lower space every column's lower
// bound is 0, so the matrix, senses, scale factors, and scaled objective are
// all constant across the whole tree. Only the scaled RHS, the per-variable
// upper bounds, and the objective offset are bound-dependent.
//
// The cache deliberately does NOT apply fixed-variable elimination or
// singleton-row deletion (both bound-dependent reductions that would change the
// standard-form shape between nodes and break warm-start basis reuse). Keeping
// the row/column structure byte-for-byte identical at every node is the
// precondition for the revised backend's warm start; the cost is a handful of
// redundant rows the reduction layer would otherwise drop. tightenBounds still
// runs per node — it is cheap and its infeasibility detection prunes nodes
// without any LP solve at all.
//
// Only valid for the revised backend (the tableau backend needs synthetic UB
// rows and the simplex state for cut generation, neither of which this path
// produces). Built lazily and reused for as long as the owning Model's
// constraint set is unchanged.
type canonCache struct {
	model *Model // identity guard — cache is only valid for this exact model
	n     int    // number of cached LP columns

	colToVar    []int
	varToCol    []int
	recoveries  []affineRecovery
	objConstant float64
	presolve    PresolveStats

	senses      []Sense
	rangeWidths []float64      // finite value marks a ranged LE row
	rowEntries  [][]canonEntry // sparse raw coefficients per kept row
	rawRHS      []float64      // raw RHS per kept row
	rowOrigIdx  []int          // kept row -> index into model.constraints
	infeasible  bool           // a structurally-empty, bound-independent infeasible row

	objRaw []float64 // raw objective per column (model sense)

	// inc is the loaded LP engine for the constant scaled body
	// (diag(rowScale)·A·diag(colScale), senses, max-sense objective). Node
	// solves pass only the bound-dependent RHS and upper bounds; the engine
	// reuses its compiled standard form across the whole tree.
	inc      *lp.Incremental
	cScaled  []float64 // scaled objective in MAX sense, constant
	rowScale []float64
	colScale []float64

	// sess is the focus-node LP session over inc, created lazily when
	// Options.RevisedFocusSession is set (see lp.Session). Nil otherwise.
	sess *lp.Session

	// rowKinds and identityCols are precomputed once and handed to every
	// node's revisedGomoryState verbatim (native per-node cut generation
	// needs them; see solveNode). identityCols works for both colToVar and
	// varToCol since this path never eliminates a variable.
	rowKinds []lpRowKind

	// dualProofs, when non-nil (Options.UseDualProofs), pools model-space
	// Farkas proofs from certified infeasible node exits and prunes later
	// nodes whose boxes provably violate a pooled proof (see dualproof.go).
	dualProofs *dualProofPool

	// rowSets is a small MRU cache of per-cut-set transforms (see rowSetFor).
	rowSets []*cachedRowSet
	// extraRowScale seeds the globally managed root-cut prefix with the exact
	// Ruiz row scales it had when baked into the fully augmented root model.
	// Local-cut tail rows keep the usual independent max-coefficient scaling.
	extraRowScale []float64
}

// cachedRowSet is one cut-slot state transformed into the cache's scaled y-space:
// set carries the column-scaled coefficients the lp engine consumes; sorted
// keeps the raw (variable, coefficient) entries in ascending order for the
// per-node RHS shift; kinds maps each cut row to its constraint index in the
// augmented model handed to cut generation. Keyed by the immutable extras
// slice identity plus an active-row mask: cutBundle.flatten memoizes extras,
// while cut aging toggles coefficients without changing row positions.
type cachedRowSet struct {
	extras []Constraint
	active []bool
	set    *lp.RowSet
	sorted [][]coefEntry
	senses []Sense
	kinds  []lpRowKind
	// scales holds the positive per-row equilibration factor applied to each
	// cut row's column-scaled coefficients (max |coef| normalized to 1). The
	// per-node RHS computation must multiply by the same factor. Base rows get
	// their Ruiz row scale; without this, deep cut-on-cut rows drift orders of
	// magnitude away from the base equilibration and break the factorization.
	scales []float64
}

// rowSetFor returns the transform for extras, consulting a small MRU cache.
func (c *canonCache) rowSetFor(extras []Constraint) *cachedRowSet {
	return c.rowSetForState(extras, extras)
}

// rowSetForState returns the transform for one fixed cut-slot layout and its
// current active coefficients. structure supplies stable row identity/senses;
// rows has the same length and contains either the original cut or an empty
// aged-out slot.
func (c *canonCache) rowSetForState(structure, rows []Constraint) *cachedRowSet {
	if len(structure) != len(rows) {
		structure = rows
	}
	active := make([]bool, len(rows))
	for i := range rows {
		active[i] = len(rows[i].Coeffs) != 0
	}
	for i, rs := range c.rowSets {
		if !sameConstraintSliceIdentity(rs.extras, structure) || !boolsEqual(rs.active, active) {
			continue
		}
		if i != 0 {
			copy(c.rowSets[1:i+1], c.rowSets[:i])
			c.rowSets[0] = rs
		}
		return rs
	}
	k := len(structure)
	rs := &cachedRowSet{
		extras: structure,
		active: active,
		set:    &lp.RowSet{Senses: make([]Sense, k), Rows: make([]lp.SparseVector, k)},
		sorted: make([][]coefEntry, k),
		senses: make([]Sense, k),
		kinds:  make([]lpRowKind, k),
		scales: make([]float64, k),
	}
	nBase := len(c.model.constraints)
	for i, cons := range rows {
		entries := sortedCoeffs(cons.Coeffs)
		kept := entries[:0]
		// First pass: map variables to LP columns and find the row's largest
		// column-scaled magnitude. For the plain cache the mapping is the
		// identity; the aggregated cache eliminates variables (varToCol < 0),
		// and callers must not pass cuts that reference them (see
		// rootCutMappableToCache).
		maxAbs := 0.0
		for _, e := range entries {
			if e.coef == 0 {
				continue
			}
			col := e.v
			if e.v < len(c.varToCol) {
				col = c.varToCol[e.v]
			}
			if col < 0 || col >= c.n {
				continue
			}
			kept = append(kept, e)
			maxAbs = math.Max(maxAbs, math.Abs(e.coef*c.colScale[col]))
		}
		rowScale := 1.0
		if i < len(c.extraRowScale) && c.extraRowScale[i] > 0 {
			rowScale = c.extraRowScale[i]
		} else if maxAbs > 0 {
			rowScale = 1 / maxAbs
		}
		var sv lp.SparseVector
		for _, e := range kept {
			col := e.v
			if e.v < len(c.varToCol) {
				col = c.varToCol[e.v]
			}
			sv.Append(col, e.coef*c.colScale[col]*rowScale)
		}
		rs.scales[i] = rowScale
		rs.sorted[i] = kept
		rs.set.Rows[i] = sv
		sense := structure[i].Sense
		rs.set.Senses[i] = sense
		rs.senses[i] = sense
		rs.kinds[i] = lpRowKind{idx: nBase + i}
	}
	const maxRowSets = 4
	if len(c.rowSets) < maxRowSets {
		c.rowSets = append(c.rowSets, nil)
	}
	copy(c.rowSets[1:], c.rowSets)
	c.rowSets[0] = rs
	return rs
}

func sameConstraintSliceIdentity(a, b []Constraint) bool {
	if len(a) != len(b) {
		return false
	}
	return len(a) == 0 || &a[0] == &b[0]
}

func boolsEqual(a, b []bool) bool {
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

type canonEntry struct {
	col  int
	coef float64
}

// buildCanonCache constructs the cache for a model under the revised backend.
// Mirrors canonicalizeWithReductions(m, nil, false) for row selection, minus
// the bound-dependent singleton-row deletion.
func buildCanonCache(m *Model) *canonCache {
	n := len(m.vars)
	c := &canonCache{model: m, n: n}
	c.colToVar = make([]int, n)
	c.varToCol = make([]int, n)
	for i := 0; i < n; i++ {
		c.colToVar[i] = i
		c.varToCol[i] = i
	}

	c.objRaw = make([]float64, n)
	for v, coef := range m.objective {
		if int(v) < n {
			c.objRaw[int(v)] = coef
		}
	}

	const tol = 1e-9
	rowRedundant := func(sense Sense, rhs float64) bool {
		switch sense {
		case LessOrEqual:
			return 0 <= rhs+tol
		case GreaterOrEqual:
			return 0 >= rhs-tol
		case Equal:
			return math.Abs(rhs) <= tol
		default:
			return false
		}
	}

	sorted := m.sortedRows()
	for idx, cons := range m.constraints {
		// Iterate the sorted view so the per-node RHS-shift accumulation in
		// solveNode runs in a deterministic order.
		entries := make([]canonEntry, 0, len(sorted[idx]))
		for _, e := range sorted[idx] {
			if e.coef == 0 || e.v >= n {
				continue
			}
			entries = append(entries, canonEntry{col: e.v, coef: e.coef})
		}
		if len(entries) == 0 {
			// Structurally empty row: feasibility depends only on the RHS
			// sign, which is bound-independent. Redundant rows are dropped;
			// a violated empty row makes the whole model infeasible.
			if !rowRedundant(cons.Sense, cons.RHS) {
				c.infeasible = true
			}
			continue
		}
		c.rowEntries = append(c.rowEntries, entries)
		c.rowOrigIdx = append(c.rowOrigIdx, idx)
		c.senses = append(c.senses, cons.Sense)
		c.rawRHS = append(c.rawRHS, cons.RHS)
	}

	// Build the dense raw matrix and Ruiz-scale it once. ruizScale mutates a,
	// b, c in place and the factors depend only on |a_ij|; we pass the
	// max-sense objective so cScaled comes out correct and a throwaway b
	// (recomputed per node from the actual bounds). The scaled matrix is then
	// stored as sparse columns — the form the lp engine consumes — so node
	// solves share one set of columns instead of re-extracting them from a
	// dense matrix at every node.
	rows := len(c.senses)
	aScaled := make([][]float64, rows)
	for r, entries := range c.rowEntries {
		row := make([]float64, n)
		for _, e := range entries {
			row[e.col] = e.coef
		}
		aScaled[r] = row
	}
	c.cScaled = make([]float64, n)
	for j := 0; j < n; j++ {
		if m.direction == Minimize {
			c.cScaled[j] = -c.objRaw[j]
		} else {
			c.cScaled[j] = c.objRaw[j]
		}
	}
	dummyB := make([]float64, rows)
	sf := ruizScale(aScaled, dummyB, c.cScaled, 8)
	c.rowScale = sf.rowScale
	c.colScale = sf.colScale
	c.inc = lp.NewIncremental(lp.Problem{
		C: c.cScaled, Cols: denseToCols(aScaled, n), Senses: c.senses,
		RangeWidths: c.rangeWidths,
	})

	c.rowKinds = make([]lpRowKind, rows)
	for r, idx := range c.rowOrigIdx {
		c.rowKinds[r] = lpRowKind{idx: idx}
	}
	c.presolve = PresolveStats{
		OriginalColumns: len(m.vars), OriginalRows: len(m.constraints),
		PresolvedColumns: n, PresolvedRows: len(c.senses),
	}
	for _, row := range m.sortedRows() {
		for _, e := range row {
			if e.coef != 0 {
				c.presolve.OriginalNonzeros++
			}
		}
	}
	for _, row := range c.rowEntries {
		c.presolve.PresolvedNonzeros += len(row)
	}
	return c
}

// buildCanonCacheScaledLike compiles base with the exact Ruiz scaling of the
// augmented reference model. reference must be base plus a suffix of global
// root cuts. The returned cache keeps only base rows in its fixed Incremental
// body and records the suffix row scales for RowSet slots, making the active
// all-root-cuts LP numerically identical to the baked reference while still
// allowing individual rows to be emptied later.
func buildCanonCacheScaledLike(base, reference *Model) *canonCache {
	c := buildCanonCache(base)
	if base == nil || reference == nil || len(reference.constraints) <= len(base.constraints) {
		return c
	}
	ref := buildCanonCache(reference)
	if c.n != ref.n || len(c.rowEntries) > len(ref.rowEntries) || len(c.colScale) != len(ref.colScale) {
		return c
	}
	for i := range c.colToVar {
		if c.colToVar[i] != ref.colToVar[i] {
			return c
		}
	}

	c.colScale = append(c.colScale[:0], ref.colScale...)
	c.rowScale = append(c.rowScale[:0], ref.rowScale[:len(c.rowEntries)]...)
	c.cScaled = make([]float64, c.n)
	for j := 0; j < c.n; j++ {
		coef := c.objRaw[j]
		if base.direction == Minimize {
			coef = -coef
		}
		c.cScaled[j] = coef * c.colScale[j]
	}
	aScaled := make([][]float64, len(c.rowEntries))
	for r, entries := range c.rowEntries {
		row := make([]float64, c.n)
		for _, e := range entries {
			row[e.col] = e.coef * c.rowScale[r] * c.colScale[e.col]
		}
		aScaled[r] = row
	}
	c.inc = lp.NewIncremental(lp.Problem{
		C: c.cScaled, Cols: denseToCols(aScaled, c.n), Senses: c.senses,
		RangeWidths: c.rangeWidths,
	})
	c.sess = nil
	c.rowSets = nil
	c.extraRowScale = append([]float64(nil), ref.rowScale[len(c.rowEntries):]...)
	return c
}

func buildAggregatedCanonCache(m *Model) *canonCache {
	canonical := canonicalizeWithOptions(m, nil, false, true)
	c := &canonCache{
		model:      m,
		n:          len(canonical.colToVar),
		colToVar:   canonical.colToVar,
		varToCol:   canonical.varToCol,
		recoveries: canonical.recoveries,
		presolve: PresolveStats{
			OriginalColumns: len(m.vars), OriginalRows: len(m.constraints),
			PresolvedColumns: len(canonical.colToVar), AggregatedColumns: len(canonical.recoveries),
		},
	}
	c.objRaw = append([]float64(nil), canonical.c...)
	c.objConstant = canonical.offset
	for j, coef := range canonical.c {
		c.objConstant -= coef * canonical.lower[j]
	}

	for _, row := range m.sortedRows() {
		for _, e := range row {
			if e.coef != 0 {
				c.presolve.OriginalNonzeros++
			}
		}
	}

	plans := continuousEqualityAggregations(m)
	byRow := make(map[int]equalityAggregation, len(plans))
	for _, plan := range plans {
		byRow[plan.row] = plan
	}
	appendRow := func(entries []coefEntry, sense Sense, rhs, width float64, originalRow int) {
		row := make([]canonEntry, 0, len(entries))
		for _, e := range entries {
			col := c.varToCol[e.v]
			if e.coef == 0 || col < 0 {
				continue
			}
			row = append(row, canonEntry{col: col, coef: e.coef})
		}
		if len(row) == 0 {
			const tol = 1e-9
			feasible := false
			if !math.IsInf(width, 1) {
				feasible = 0 <= rhs+tol && 0 >= rhs-width-tol
			} else {
				switch sense {
				case LessOrEqual:
					feasible = 0 <= rhs+tol
				case GreaterOrEqual:
					feasible = 0 >= rhs-tol
				case Equal:
					feasible = math.Abs(rhs) <= tol
				}
			}
			if !feasible {
				c.infeasible = true
			}
			return
		}
		c.rowEntries = append(c.rowEntries, row)
		c.senses = append(c.senses, sense)
		c.rawRHS = append(c.rawRHS, rhs)
		c.rangeWidths = append(c.rangeWidths, width)
		c.rowOrigIdx = append(c.rowOrigIdx, originalRow)
		c.rowKinds = append(c.rowKinds, lpRowKind{idx: originalRow})
		c.presolve.PresolvedNonzeros += len(row)
	}

	rows := m.sortedRows()
	for rowIdx, cons := range m.constraints {
		if plan, ok := byRow[rowIdx]; ok {
			switch {
			case plan.addLower && plan.addUpper:
				appendRow(plan.rest, LessOrEqual, plan.upper, plan.upper-plan.lower, rowIdx)
			case plan.addUpper:
				appendRow(plan.rest, LessOrEqual, plan.upper, math.Inf(1), rowIdx)
			case plan.addLower:
				appendRow(plan.rest, GreaterOrEqual, plan.lower, math.Inf(1), rowIdx)
			}
			continue
		}
		appendRow(rows[rowIdx], cons.Sense, cons.RHS, math.Inf(1), rowIdx)
	}
	c.presolve.PresolvedRows = len(c.senses)

	aScaled := make([][]float64, len(c.senses))
	for r, entries := range c.rowEntries {
		row := make([]float64, c.n)
		for _, e := range entries {
			row[e.col] = e.coef
		}
		aScaled[r] = row
	}
	c.cScaled = append([]float64(nil), c.objRaw...)
	if m.direction == Minimize {
		for j := range c.cScaled {
			c.cScaled[j] = -c.cScaled[j]
		}
	}
	dummyB := make([]float64, len(c.senses))
	sf := ruizScale(aScaled, dummyB, c.cScaled, 8)
	c.rowScale = sf.rowScale
	c.colScale = sf.colScale
	for i, width := range c.rangeWidths {
		if !math.IsInf(width, 1) {
			c.rangeWidths[i] = width * c.rowScale[i]
		}
	}
	c.inc = lp.NewIncremental(lp.Problem{
		C: c.cScaled, Cols: denseToCols(aScaled, c.n), Senses: c.senses,
		RangeWidths: c.rangeWidths,
	})
	return c
}

// solveNode solves the LP relaxation at a B&B node using the cached structure.
// It returns the same (Result, lpResponse) shape the node sites expect from
// solveLPModelWithBackend, but reuses the constant matrix and scale factors.
func (c *canonCache) solveNode(ctx context.Context, bounds []boundConstraint, opts Options, warm *lpWarmStart) (Result, lpResponse) {
	return c.solveNodeRows(ctx, bounds, nil, opts, warm)
}

// lpConflictCertificateBound re-evaluates a Phase-I dual certificate after
// changing only variable bounds. It uses the certificate's original row
// orientation, so RHS sign crossings do not change the standard-form matrix.
func (c *canonCache) lpConflictCertificateBound(cert *lp.InfeasibilityCertificate, bounds []boundConstraint) (float64, bool) {
	if cert == nil || len(cert.Dual) != len(c.senses) || len(cert.Flip) != len(c.senses) || len(cert.ReducedCosts) != c.n {
		return 0, false
	}
	lower, upper := effectiveBounds(c.model, bounds)
	// Recompute the row-propagation closure under the retained assumptions
	// before evaluating the certificate. This is the cheap reason-resolution
	// step conflict minimization needs: if a removed branch literal is implied
	// again by other retained literals, the certificate sees the same tightened
	// domain and can soundly discard it from the learned no-good.
	if tightenBounds(c.model, lower, upper, 5) {
		return math.Inf(1), true
	}
	bound := cert.AuxContribution
	for r, entries := range c.rowEntries {
		b := c.rawRHS[r]
		for _, e := range entries {
			b -= e.coef * lower[c.colToVar[e.col]]
		}
		b *= c.rowScale[r]
		if cert.Flip[r] {
			b = -b
		}
		bound += cert.Dual[r] * b
	}
	for j, reducedCost := range cert.ReducedCosts {
		orig := c.colToVar[j]
		width := upper[orig] - lower[orig]
		if width < 0 {
			return math.Inf(1), true
		}
		if reducedCost >= 0 {
			continue
		}
		if math.IsInf(width, 1) {
			return math.Inf(-1), true
		}
		bound += reducedCost * (width / c.colScale[j])
	}
	return bound, !math.IsNaN(bound)
}

// solveNodeRows is solveNode plus per-node extra constraint rows (the
// inherited cut set): the extras are transformed once per distinct cut set
// into the cache's scaled y-space (see rowSetFor) and appended after the
// base rows via the lp engine's RowSet path, so a cut-augmented node LP
// reuses the same compiled body and warm-start machinery as a plain one
// instead of rebuilding the whole problem from the Model.
//
// Active extra rows participate in bound propagation. Ordinarily they use a
// stable per-row equilibration over the cache's column scales; managed global
// root rows can instead carry the exact full-model Ruiz row/column scaling
// installed by buildCanonCacheScaledLike, so moving them into slots does not
// silently change the numerical representation before aging begins.
func (c *canonCache) solveNodeRows(ctx context.Context, bounds []boundConstraint, extras []Constraint, opts Options, warm *lpWarmStart) (Result, lpResponse) {
	return c.solveNodeRowsWithStructure(ctx, bounds, extras, extras, opts, warm)
}

type preparedNodeLP struct {
	lower, upper              []float64
	b, ub                     []float64
	ext                       *lp.RowSet
	rs                        *cachedRowSet
	offset                    float64
	propagationReductions     int
	objectiveCutoffReductions int
	infeasible                bool
	infeasibilityCertified    bool
	objectiveCutoffPruned     bool
	dualProofShadowPrune      bool
}

type cachedStrongBranchSession struct {
	cache     *canonCache
	lp        *lp.StrongBranchSession
	ext       *lp.RowSet
	structure []Constraint
	rhsExtras []Constraint
}

func (c *canonCache) newStrongBranchSession(parent *revisedGomoryState, baseBounds []boundConstraint, structure, rhsExtras []Constraint, opts Options) *cachedStrongBranchSession {
	if c == nil || parent == nil || parent.solver == nil {
		return nil
	}
	base := c.prepareNodeRowsWithStructure(baseBounds, structure, rhsExtras, opts)
	if base.infeasible {
		return nil
	}
	sb := c.inc.NewStrongBranchSession(parent.solver, base.ext)
	if sb == nil {
		return nil
	}
	return &cachedStrongBranchSession{
		cache: c, lp: sb, ext: base.ext,
		structure: structure, rhsExtras: rhsExtras,
	}
}

func (sb *cachedStrongBranchSession) probe(ctx context.Context, bounds []boundConstraint, opts Options, iterLimit int) (lp.StrongBranchResult, int, bool) {
	if sb == nil || sb.lp == nil {
		return lp.StrongBranchResult{}, 0, false
	}
	p := sb.cache.prepareNodeRowsWithStructure(bounds, sb.structure, sb.rhsExtras, opts)
	if p.infeasible {
		return lp.StrongBranchResult{InfeasibleCertified: p.infeasibilityCertified}, p.propagationReductions, true
	}
	if p.ext != sb.ext {
		return lp.StrongBranchResult{}, 0, false
	}
	probeOpts := lpOptions(opts, time.Now())
	probeOpts.MaxIterations = iterLimit
	probeOpts.TimeLimit = 0
	probeOpts.CertifyInfeasible = true
	result := sb.lp.Probe(ctx, probeOpts, p.b, p.ub, iterLimit)
	if result.EstimateValid {
		if sb.cache.model.direction == Minimize {
			result.Estimate = -result.Estimate + p.offset
		} else {
			result.Estimate += p.offset
		}
	}
	if result.BoundValid {
		if sb.cache.model.direction == Minimize {
			result.Bound = -result.Bound + p.offset
		} else {
			result.Bound += p.offset
		}
	}
	return result, p.propagationReductions, true
}

// prepareNodeRowsWithStructure performs the model-space bound propagation,
// lower-bound shift, scaling, and stable RowSet lookup shared by ordinary
// node solves and native strong-branch probes.
func (c *canonCache) prepareNodeRowsWithStructure(bounds []boundConstraint, structure, rhsExtras []Constraint, opts Options) preparedNodeLP {
	if len(structure) != len(rhsExtras) {
		structure = rhsExtras
	}
	p := preparedNodeLP{}
	p.lower, p.upper = effectiveBounds(c.model, bounds)
	var beforeLower, beforeUpper []float64
	if opts.ReliabilityProbeInference {
		beforeLower = append([]float64(nil), p.lower...)
		beforeUpper = append([]float64(nil), p.upper...)
	}

	propagationModel := c.model
	if len(rhsExtras) > 0 {
		// RowSet rows are part of the mathematical node relaxation even though
		// they live outside the cache's fixed matrix body. Include the currently
		// active rows in domain propagation so a dynamically managed root-cut LP
		// matches the baked-model path before simplex starts.
		propagationModel = modelWithExtraConstraints(c.model, rhsExtras)
	}
	if c.infeasible || tightenBounds(propagationModel, p.lower, p.upper, 5) {
		p.infeasible = true
		p.infeasibilityCertified = true
		return p
	}
	if opts.objectiveCutoffActive {
		for pass := 0; pass < 3; pass++ {
			n, infeasible := propagateObjectiveCutoff(propagationModel, p.lower, p.upper, opts.objectiveCutoff, opts.FeasibilityTol)
			p.objectiveCutoffReductions += n
			if infeasible {
				p.infeasible = true
				p.infeasibilityCertified = true
				p.objectiveCutoffPruned = true
				return p
			}
			if n == 0 {
				break
			}
			if tightenBounds(c.model, p.lower, p.upper, 5) {
				p.infeasible = true
				p.infeasibilityCertified = true
				p.objectiveCutoffPruned = true
				return p
			}
		}
	}
	for i := range beforeLower {
		if p.lower[i] > beforeLower[i]+1e-9 {
			p.propagationReductions++
		}
		if p.upper[i] < beforeUpper[i]-1e-9 {
			p.propagationReductions++
		}
	}
	for i := range c.model.vars {
		if p.upper[i] < p.lower[i] {
			p.infeasible = true
			p.infeasibilityCertified = true
			return p
		}
	}
	// Dual-proof conflict propagation: a pooled proof whose maximum activity
	// over the post-propagation box stays below its rhs certifies the node
	// infeasible before any LP work (SCIP's dualray constraint propagation).
	// Sound for strong-branch probe boxes too — proofs hold for every
	// integer-feasible point regardless of which node asks.
	if c.dualProofs.pruneNode(p.lower, p.upper) {
		if dualProofVerify {
			p.dualProofShadowPrune = true
		} else {
			p.infeasible = true
			p.infeasibilityCertified = true
			return p
		}
	}

	rows := len(c.senses)
	total := rows
	if len(structure) > 0 {
		p.rs = c.rowSetForState(structure, rhsExtras)
		p.ext = p.rs.set
		total += len(structure)
	}
	p.b = make([]float64, total)
	for r, entries := range c.rowEntries {
		sh := c.rawRHS[r]
		for _, e := range entries {
			sh -= e.coef * p.lower[c.colToVar[e.col]]
		}
		p.b[r] = c.rowScale[r] * sh
	}
	if p.rs != nil {
		for i, entries := range p.rs.sorted {
			sh := rhsExtras[i].RHS
			for _, e := range entries {
				sh -= e.coef * p.lower[e.v]
			}
			p.b[rows+i] = sh * p.rs.scales[i]
		}
	}

	p.ub = make([]float64, c.n)
	p.offset = c.objConstant
	for j := 0; j < c.n; j++ {
		orig := c.colToVar[j]
		p.offset += c.objRaw[j] * p.lower[orig]
		d := p.upper[orig] - p.lower[orig]
		if !math.IsInf(d, 1) {
			d /= c.colScale[j]
		}
		p.ub[j] = d
	}
	return p
}

func (c *canonCache) recoverNodeValues(p preparedNodeLP, yValues []float64) []float64 {
	values := make([]float64, len(c.model.vars))
	for i, y := range yValues {
		if i >= c.n {
			continue
		}
		orig := c.colToVar[i]
		values[orig] = y*c.colScale[i] + p.lower[orig]
	}
	for _, recovery := range c.recoveries {
		value := recovery.constant
		for _, term := range recovery.terms {
			value += term.coef * values[term.v]
		}
		values[recovery.variable] = value
	}
	return values
}

// solveNodeRowsWithStructure decouples stable cut-slot identity from the
// per-solve active rows. structure supplies immutable row identity/senses and
// rhsExtras supplies current coefficients/RHS values. They normally alias;
// cut aging clears selected coefficients while preserving row order and sense.
func (c *canonCache) solveNodeRowsWithStructure(ctx context.Context, bounds []boundConstraint, structure, rhsExtras []Constraint, opts Options, warm *lpWarmStart) (Result, lpResponse) {
	p := c.prepareNodeRowsWithStructure(bounds, structure, rhsExtras, opts)
	if p.infeasible {
		return Result{Status: Infeasible}, lpResponse{
			status:                    Infeasible,
			infeasibilityCertified:    p.infeasibilityCertified,
			objectiveCutoffReductions: p.objectiveCutoffReductions,
			objectiveCutoffPruned:     p.objectiveCutoffPruned,
		}
	}
	if opts.RevisedFocusSession && c.sess == nil {
		c.sess = lp.NewSession(c.inc)
	}
	var sess *lp.Session
	if opts.RevisedFocusSession {
		sess = c.sess
	}
	resp := solveRevisedSessionRows(sess, c.inc, ctx, opts, time.Now(), p.ext, p.b, p.ub, warm)
	resp.propagationReductions = p.propagationReductions
	resp.objectiveCutoffReductions = p.objectiveCutoffReductions
	if c.dualProofs != nil && resp.status == Infeasible && resp.farkasProof != nil && p.rs == nil {
		// Certified infeasible exit on the plain body: translate the proof to
		// model space and pool it for tree-wide reuse. Cut-augmented solves
		// (p.rs != nil) are excluded for now — their proofs may lean on cut
		// rows, and while globally valid cuts keep the proof valid for every
		// integer-feasible point, the conservative first slice pools only
		// proofs derived from the model's own rows.
		c.dualProofs.addFromLP(c, resp.farkasProof, p.lower)
	}
	if p.dualProofShadowPrune && resp.status != Infeasible {
		c.dualProofs.stats.VerifyContradicted++
		debugf("dual-proof VERIFY CONTRADICTED: shadow-pruned node solved to %v", resp.status)
	}

	// Populate the same metadata solveLPModelWithBackend stamps onto
	// revisedState for the generic path, so a cache-solved node can still
	// feed native per-node cut generation (revisedGomoryCuts) without an
	// extra tableau solve. colToVar/varToCol are both the identity mapping —
	// this cache path never eliminates a variable (see the type doc). With
	// extras, the per-row metadata is extended: cut rows carry row scale 1
	// and rowKinds indices pointing into the augmented model's constraints.
	if resp.revisedState != nil {
		if p.rs == nil {
			resp.revisedState.scales = scaleFactors{rowScale: c.rowScale, colScale: c.colScale}
			resp.revisedState.rowKinds = c.rowKinds
		} else {
			rowScale := make([]float64, len(p.b))
			copy(rowScale, c.rowScale)
			for i := len(c.senses); i < len(p.b); i++ {
				rowScale[i] = p.rs.scales[i-len(c.senses)]
			}
			rowKinds := make([]lpRowKind, 0, len(p.b))
			rowKinds = append(rowKinds, c.rowKinds...)
			rowKinds = append(rowKinds, p.rs.kinds...)
			resp.revisedState.scales = scaleFactors{rowScale: rowScale, colScale: c.colScale}
			resp.revisedState.rowKinds = rowKinds
		}
		resp.revisedState.colToVar = c.colToVar
		resp.revisedState.varToCol = c.varToCol
		resp.revisedState.modelLower = p.lower
		resp.revisedState.modelUpper = p.upper
	}

	res := Result{Status: resp.status, Iterations: resp.iterations, Stats: &SolverStats{Presolve: c.presolve}}
	if resp.revised.LPSolves > 0 {
		res.Stats.addRevised(resp.revised)
	}
	if resp.status != Optimal {
		return res, resp
	}

	res.Values = c.recoverNodeValues(p, resp.values)
	if c.model.direction == Minimize {
		res.Objective = -resp.objective + p.offset
	} else {
		res.Objective = resp.objective + p.offset
	}
	return res, resp
}
