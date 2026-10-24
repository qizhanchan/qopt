package qopt

import "math"

type lpProblem struct {
	direction  Direction
	c          []float64
	offset     float64
	a          [][]float64
	senses     []Sense
	b          []float64
	lower      []float64
	upper      []float64
	rowKinds   []lpRowKind // identifies each row (user constraint vs synthetic UB)
	colToVar   []int       // reduced LP column -> original Model var index
	varToCol   []int       // original Model var index -> reduced LP column, or -1 if fixed
	origLower  []float64   // original-space bounds after tightening
	origUpper  []float64
	fixed      []bool
	fixedVal   []float64
	aggregated []bool
	recoveries []affineRecovery
}

// lpRowKind identifies the origin of a row in the canonicalized LP, used by
// cut generation to substitute non-basic slacks correctly. User-constraint
// rows store the index into m.constraints; upper-bound rows store the
// variable index whose `y_var ≤ UB − lower` they enforce.
type lpRowKind struct {
	isUpperBound bool
	idx          int
}

type boundConstraint struct {
	varIndex int
	sense    Sense
	rhs      float64
}

// effectiveBounds returns the merged variable bounds: each m.vars[i]'s spec
// lower/upper, refined by any branch-and-bound bounds in `nodeBounds`. Binary
// vars are normalized to [0,1]. An unset (zero) upper for non-binary is
// promoted to +Inf to match Model.AddVar's contract.
func effectiveBounds(m *Model, nodeBounds []boundConstraint) (lower, upper []float64) {
	n := len(m.vars)
	lower = make([]float64, n)
	upper = make([]float64, n)
	for i, spec := range m.vars {
		// AddVar resolves bounds into non-nil concrete values (Binary clamped to
		// [0,1]), so we can read them directly.
		lower[i] = *spec.Lower
		upper[i] = *spec.Upper
	}
	for _, b := range nodeBounds {
		switch b.sense {
		case LessOrEqual:
			if b.rhs < upper[b.varIndex] {
				upper[b.varIndex] = b.rhs
			}
		case GreaterOrEqual:
			if b.rhs > lower[b.varIndex] {
				lower[b.varIndex] = b.rhs
			}
		case Equal:
			lower[b.varIndex] = b.rhs
			upper[b.varIndex] = b.rhs
		}
	}
	return lower, upper
}

func canonicalize(m *Model, bounds []boundConstraint) lpProblem {
	return canonicalizeWithReductions(m, bounds, true)
}

func canonicalizeWithReductions(m *Model, bounds []boundConstraint, reduceFixed bool) lpProblem {
	return canonicalizeWithOptions(m, bounds, reduceFixed, false)
}

func canonicalizeWithOptions(m *Model, bounds []boundConstraint, reduceFixed, aggregateEqualities bool) lpProblem {
	n := len(m.vars)
	lp := lpProblem{
		direction:  m.direction,
		varToCol:   make([]int, n),
		origLower:  make([]float64, n),
		origUpper:  make([]float64, n),
		fixed:      make([]bool, n),
		fixedVal:   make([]float64, n),
		aggregated: make([]bool, n),
	}
	for i := range lp.varToCol {
		lp.varToCol[i] = -1
	}
	lower, upper := effectiveBounds(m, bounds)

	// Bound-tightening presolve: walk the constraints repeatedly to derive
	// the tightest variable bounds implied by the current bounds plus the
	// constraint set. Each B&B node reruns this with its added bounds, so
	// branching that creates infeasibility is detected without an LP solve.
	// On infeasibility we plant a 0·y ≤ −1 row that the simplex catches.
	if tightenBounds(m, lower, upper, 5) {
		lp.a = append(lp.a, nil)
		lp.senses = append(lp.senses, LessOrEqual)
		lp.b = append(lp.b, -1)
		lp.rowKinds = append(lp.rowKinds, lpRowKind{isUpperBound: true, idx: -1})
		return lp
	}
	copy(lp.origLower, lower)
	copy(lp.origUpper, upper)

	var aggregations []equalityAggregation
	aggregationByRow := make(map[int]equalityAggregation)
	if aggregateEqualities {
		aggregations = continuousEqualityAggregations(m)
		for _, agg := range aggregations {
			lp.aggregated[agg.variable] = true
			aggregationByRow[agg.row] = agg
			terms := make([]coefEntry, len(agg.rest))
			for i, e := range agg.rest {
				terms[i] = coefEntry{v: e.v, coef: -e.coef / agg.coef}
			}
			lp.recoveries = append(lp.recoveries, affineRecovery{
				variable: agg.variable,
				constant: agg.rhs / agg.coef,
				terms:    terms,
			})
		}
	}

	for i := range m.vars {
		if upper[i] < lower[i] {
			lp.a = append(lp.a, nil)
			lp.senses = append(lp.senses, LessOrEqual)
			lp.b = append(lp.b, -1)
			lp.rowKinds = append(lp.rowKinds, lpRowKind{isUpperBound: true, idx: i})
			return lp
		}
		// Fixed-variable elimination must not collapse a merely narrow
		// continuous interval: a tiny width can still carry a large objective
		// contribution. Bound propagation writes genuinely fixed bounds exactly.
		if reduceFixed && !math.IsInf(lower[i], -1) && !math.IsInf(upper[i], 1) && upper[i] == lower[i] {
			lp.fixed[i] = true
			lp.fixedVal[i] = lower[i]
			continue
		}
		if lp.aggregated[i] {
			continue
		}
		lp.varToCol[i] = len(lp.colToVar)
		lp.colToVar = append(lp.colToVar, i)
		lp.lower = append(lp.lower, lower[i])
		lp.upper = append(lp.upper, upper[i])
	}

	objective := make([]float64, n)
	for v, c := range m.objective {
		if int(v) >= 0 && int(v) < n {
			objective[int(v)] = c
		}
	}
	for _, agg := range aggregations {
		c := objective[agg.variable]
		if c == 0 {
			continue
		}
		lp.offset += c * agg.rhs / agg.coef
		for _, e := range agg.rest {
			objective[e.v] -= c * e.coef / agg.coef
		}
		objective[agg.variable] = 0
	}

	lp.c = make([]float64, len(lp.colToVar))
	for i, c := range objective {
		if c != 0 {
			if lp.fixed[i] {
				lp.offset += c * lp.fixedVal[i]
				continue
			}
			col := lp.varToCol[i]
			if col < 0 {
				continue
			}
			lp.c[col] = c
			lp.offset += c * lp.lower[col]
		}
	}

	// addRow consumes coefficients in ascending variable order (coefEntry
	// slices) so the RHS-shift accumulation is deterministic across runs —
	// map iteration order would perturb the sum's last ulp.
	addRow := func(entries []coefEntry, sense Sense, rhs float64) {
		row := make([]float64, len(lp.colToVar))
		shiftedRHS := rhs
		for _, e := range entries {
			idx, c := e.v, e.coef
			if lp.fixed[idx] {
				shiftedRHS -= c * lp.fixedVal[idx]
				continue
			}
			col := lp.varToCol[idx]
			if col < 0 {
				continue
			}
			row[col] = c
			shiftedRHS -= c * lp.lower[col]
		}
		lp.a = append(lp.a, row)
		lp.senses = append(lp.senses, sense)
		lp.b = append(lp.b, shiftedRHS)
	}

	rowRedundant := func(sense Sense, rhs float64) bool {
		const tol = 1e-9
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
	singletonRowRedundant := func(row []float64, sense Sense, rhs float64) (bool, int) {
		const tol = 1e-9
		col := -1
		coef := 0.0
		for j, c := range row {
			if c == 0 {
				continue
			}
			if col >= 0 {
				return false, -1
			}
			col = j
			coef = c
		}
		if col < 0 {
			return rowRedundant(sense, rhs), -1
		}

		ub := math.Inf(1)
		if col < len(lp.upper) {
			ub = lp.upper[col] - lp.lower[col]
		}
		minAct, maxAct := 0.0, 0.0
		minInf, maxInf := false, false
		if coef > 0 {
			if math.IsInf(ub, 1) {
				maxInf = true
			} else {
				maxAct = coef * ub
			}
		} else {
			if math.IsInf(ub, 1) {
				minInf = true
			} else {
				minAct = coef * ub
			}
		}

		switch sense {
		case LessOrEqual:
			return !maxInf && maxAct <= rhs+tol, col
		case GreaterOrEqual:
			return !minInf && minAct >= rhs-tol, col
		case Equal:
			return !minInf && !maxInf && math.Abs(minAct-rhs) <= tol && math.Abs(maxAct-rhs) <= tol, col
		default:
			return false, -1
		}
	}
	addInfeasibleRow := func(kind lpRowKind) {
		lp.a = append(lp.a, make([]float64, len(lp.colToVar)))
		lp.senses = append(lp.senses, LessOrEqual)
		lp.b = append(lp.b, -1)
		lp.rowKinds = append(lp.rowKinds, kind)
	}

	appendConstraintRow := func(entries []coefEntry, sense Sense, rhs float64, kind lpRowKind) {
		before := len(lp.a)
		addRow(entries, sense, rhs)
		empty := true
		for _, coef := range lp.a[before] {
			if coef != 0 {
				empty = false
				break
			}
		}
		if empty {
			shiftedRHS := lp.b[before]
			lp.a = lp.a[:before]
			lp.senses = lp.senses[:before]
			lp.b = lp.b[:before]
			if rowRedundant(sense, shiftedRHS) {
				return
			}
			addInfeasibleRow(kind)
			return
		}
		if redundant, col := singletonRowRedundant(lp.a[before], sense, lp.b[before]); redundant {
			if col >= 0 {
				orig := lp.colToVar[col]
				if m.vars[orig].Type != Continuous {
					lp.rowKinds = append(lp.rowKinds, kind)
					return
				}
			}
			lp.a = lp.a[:before]
			lp.senses = lp.senses[:before]
			lp.b = lp.b[:before]
			return
		}
		lp.rowKinds = append(lp.rowKinds, kind)
	}

	rows := m.sortedRows()
	for idx, c := range m.constraints {
		if agg, ok := aggregationByRow[idx]; ok {
			if agg.addLower {
				appendConstraintRow(agg.rest, GreaterOrEqual, agg.lower, lpRowKind{idx: idx})
			}
			if agg.addUpper {
				appendConstraintRow(agg.rest, LessOrEqual, agg.upper, lpRowKind{idx: idx})
			}
			continue
		}
		appendConstraintRow(rows[idx], c.Sense, c.RHS, lpRowKind{idx: idx})
	}

	for i := range m.vars {
		if lp.fixed[i] || lp.aggregated[i] {
			continue
		}
		if !math.IsInf(upper[i], 1) {
			addRow([]coefEntry{{v: i, coef: 1}}, LessOrEqual, upper[i])
			lp.rowKinds = append(lp.rowKinds, lpRowKind{isUpperBound: true, idx: i})
		}
	}

	return lp
}

func (lp lpProblem) hasWarmStartBreakingReductions() bool {
	for _, fixed := range lp.fixed {
		if fixed {
			return true
		}
	}
	return false
}

func (lp lpProblem) presolveStats(m *Model) PresolveStats {
	stats := PresolveStats{
		OriginalColumns:   len(m.vars),
		OriginalRows:      len(m.constraints),
		PresolvedColumns:  len(lp.colToVar),
		AggregatedColumns: len(lp.recoveries),
	}
	for _, row := range m.sortedRows() {
		for _, e := range row {
			if e.coef != 0 {
				stats.OriginalNonzeros++
			}
		}
	}
	for i, row := range lp.a {
		if i < len(lp.rowKinds) && lp.rowKinds[i].isUpperBound {
			continue
		}
		stats.PresolvedRows++
		for _, c := range row {
			if c != 0 {
				stats.PresolvedNonzeros++
			}
		}
	}
	return stats
}

// stripUBRows removes the synthetic UB rows (rowKind.isUpperBound with a
// non-negative variable index) from an lpProblem. The infeasibility-planting
// row (idx == −1) and all real user constraints are preserved. Used by the
// revised backend, which handles upper bounds natively via lpRequest.upper.
func stripUBRows(lp lpProblem) lpProblem {
	out := lpProblem{
		direction:  lp.direction,
		c:          lp.c,
		offset:     lp.offset,
		lower:      lp.lower,
		upper:      lp.upper,
		colToVar:   lp.colToVar,
		varToCol:   lp.varToCol,
		origLower:  lp.origLower,
		origUpper:  lp.origUpper,
		fixed:      lp.fixed,
		fixedVal:   lp.fixedVal,
		aggregated: lp.aggregated,
		recoveries: lp.recoveries,
	}
	for i, rk := range lp.rowKinds {
		if rk.isUpperBound && rk.idx >= 0 {
			continue
		}
		out.a = append(out.a, lp.a[i])
		out.senses = append(out.senses, lp.senses[i])
		out.b = append(out.b, lp.b[i])
		out.rowKinds = append(out.rowKinds, rk)
	}
	return out
}
