package qopt

import "sort"

// Deterministic iteration over coefficient maps.
//
// Constraint.Coeffs and Model.objective are map[Var]float64 for API
// ergonomics, but Go map iteration order is randomized per run, and floating-
// point addition is not associative — so any activity/RHS sum accumulated in
// map order differs across runs in the last ulp. On degenerate instances
// (glass4, mad) that last-bit noise perturbs simplex tie-breaks enough to
// change pivot paths and iteration counts run-to-run. Every float
// accumulation over constraint coefficients must therefore iterate in a fixed
// order; ascending variable index is the canonical one.

// coefEntry is one (variable index, coefficient) pair of a constraint row in
// ascending variable order.
type coefEntry struct {
	v    int
	coef float64
}

// sortedCoeffs materializes a coefficient map in ascending variable order.
// For rows of a Model use Model.sortedRows, which caches; this is for ad-hoc
// Constraints (cut candidates, aggregated MIR rows) that live outside a
// Model.
func sortedCoeffs(coeffs map[Var]float64) []coefEntry {
	out := make([]coefEntry, 0, len(coeffs))
	for v, c := range coeffs {
		out = append(out, coefEntry{v: int(v), coef: c})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].v < out[j].v })
	return out
}

// sortedRows returns every constraint's coefficients in ascending variable
// order, cached on the Model. Built lazily on first use: a Model is treated
// as immutable while a Solve is running, and models with appended rows
// (cut-augmented copies, clones) are fresh Model values, so a length check is
// enough to invalidate. Not safe for concurrent Solve calls on one Model —
// give each goroutine its own Model, as the bench harness already does.
//
// Beyond determinism this is also the slice-based row iteration the
// gen-ip002 profile called for: tightenBounds reruns at every B&B node, and
// ranging a sorted slice is cheaper than ranging a map.
func (m *Model) sortedRows() [][]coefEntry {
	if len(m.rowsSorted) != len(m.constraints) {
		rows := make([][]coefEntry, len(m.constraints))
		for i := range m.constraints {
			rows[i] = sortedCoeffs(m.constraints[i].Coeffs)
		}
		m.rowsSorted = rows
	}
	return m.rowsSorted
}
