package qopt

import "math"

type equalityAggregation struct {
	row      int
	variable int
	coef     float64
	rhs      float64
	rest     []coefEntry
	lower    float64
	upper    float64
	addLower bool
	addUpper bool
}

type affineRecovery struct {
	variable int
	constant float64
	terms    []coefEntry
}

// continuousEqualityAggregations finds a deliberately narrow substitution
// class: a continuous variable that occurs in one row, and that row is an
// equality. Eliminating such a column cannot create fill in any other row.
// The plan is derived from root-global bounds so its row/column shape remains
// stable at every B&B node.
func continuousEqualityAggregations(m *Model) []equalityAggregation {
	n := len(m.vars)
	occurrences := make([]int, n)
	rows := m.sortedRows()
	for _, entries := range rows {
		for _, e := range entries {
			if e.coef != 0 && e.v >= 0 && e.v < n {
				occurrences[e.v]++
			}
		}
	}

	lower, upper := effectiveBounds(m, nil)
	if tightenBounds(m, lower, upper, 5) {
		return nil
	}

	plans := make([]equalityAggregation, 0)
	for rowIdx, cons := range m.constraints {
		if cons.Sense != Equal {
			continue
		}
		entries := rows[rowIdx]
		best := equalityAggregation{row: -1}
		bestRows := int(^uint(0) >> 1)
		bestNonzeros := int(^uint(0) >> 1)
		bestWidthScore := -1.0
		for _, candidate := range entries {
			v := candidate.v
			if candidate.coef == 0 || v < 0 || v >= n || m.vars[v].Type != Continuous || occurrences[v] != 1 {
				continue
			}
			if !math.IsInf(lower[v], -1) && !math.IsInf(upper[v], 1) && lower[v] == upper[v] {
				continue
			}

			rest := make([]coefEntry, 0, len(entries)-1)
			for _, e := range entries {
				if e.v != v && e.coef != 0 {
					rest = append(rest, e)
				}
			}
			minAct, maxAct := rowActivityBounds(rest, lower, upper)
			atLower := cons.RHS - candidate.coef*lower[v]
			atUpper := cons.RHS - candidate.coef*upper[v]
			restLower := math.Min(atLower, atUpper)
			restUpper := math.Max(atLower, atUpper)
			addLower := math.IsInf(restLower, -1) || minAct < restLower-1e-9
			addUpper := math.IsInf(restUpper, 1) || maxAct > restUpper+1e-9
			// Infinite implied sides do not produce constraints.
			if math.IsInf(restLower, -1) {
				addLower = false
			}
			if math.IsInf(restUpper, 1) {
				addUpper = false
			}
			generatedRows := 0
			if addLower {
				generatedRows++
			}
			if addUpper {
				generatedRows++
			}
			generatedNonzeros := generatedRows * len(rest)
			widthScore := math.Inf(1)
			if !math.IsInf(lower[v], -1) && !math.IsInf(upper[v], 1) {
				widthScore = math.Abs(candidate.coef) * (upper[v] - lower[v])
			}
			if generatedRows > bestRows ||
				(generatedRows == bestRows && generatedNonzeros > bestNonzeros) ||
				(generatedRows == bestRows && generatedNonzeros == bestNonzeros && widthScore < bestWidthScore) ||
				(generatedRows == bestRows && generatedNonzeros == bestNonzeros && widthScore == bestWidthScore && best.row >= 0 && v > best.variable) {
				continue
			}
			bestRows = generatedRows
			bestNonzeros = generatedNonzeros
			bestWidthScore = widthScore
			best = equalityAggregation{
				row: rowIdx, variable: v, coef: candidate.coef, rhs: cons.RHS,
				rest: rest, lower: restLower, upper: restUpper,
				addLower: addLower, addUpper: addUpper,
			}
		}
		if best.row >= 0 {
			plans = append(plans, best)
		}
	}
	return plans
}

func equalityAggregationEnabled(opts Options) bool {
	// Root cuts are generated and validated in original variable space before
	// the B&B cache is built, so the exact equality substitution can safely run
	// on the resulting fixed work model. Per-node cuts still consume row/basis
	// metadata from the aggregated cache and need a full original-row mapping;
	// keep that combination disabled until the mapping is implemented.
	return opts.AggregateContinuousEqualities && opts.MaxCutDepth == 0
}

func rowActivityBounds(entries []coefEntry, lower, upper []float64) (float64, float64) {
	minAct, maxAct := 0.0, 0.0
	for _, e := range entries {
		lo, hi := lower[e.v], upper[e.v]
		if e.coef < 0 {
			lo, hi = hi, lo
		}
		minAct = addActivity(minAct, e.coef*lo)
		maxAct = addActivity(maxAct, e.coef*hi)
	}
	return minAct, maxAct
}

func addActivity(total, value float64) float64 {
	if math.IsInf(total, 0) {
		return total
	}
	return total + value
}
