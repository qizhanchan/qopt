package qopt

import "math"

// Cut row aging: q-opt's per-node cutBundle chain is copy-on-extend but
// never shrinks — every cut a node emits is permanent for the rest of that
// subtree, so the cut-augmented LP only ever grows down any B&B path. This
// is the documented mechanical cause of "adding cuts always regresses"
// (see CLAUDE.md's P0 sub-goal 3 notes): the enlarged LP's per-node cost,
// not cut generation, is the bottleneck. SCIP avoids this with row aging —
// a cut that sits non-binding at the LP optimum for enough consecutive
// checks gets dropped from the active row set (but stays in a pool for
// potential reactivation; see cutpool.c). This file ports the aging half
// of that mechanism: a cut whose age exceeds Options.CutRowAgeLimit is
// replaced by an empty, redundant row in the LP a node and its descendants
// solve, without mutating the immutable cutBundle chain. Keeping the sense
// and row position stable preserves the standard-form column layout, so the
// revised LP can refactor and reoptimize the parent basis without a cold
// Phase I. Unlike the old RHS-only relaxation, clearing the coefficients
// actually removes the stale cut's matrix work.
//
// Reactivation is deliberate, not an oversight: updateCutAges re-evaluates
// EVERY inherited cut's binding status against the new LP point regardless
// of whether it was in the active set that produced that point (Constraint
// carries its own coefficients, so this needs no extra LP work) — a cut
// that drifts back into being tight gets its age reset to 0 and rejoins
// the active set at the next node, mirroring a pool's re-test-and-reuse
// behavior rather than a one-way kill.

// cutIsBinding reports whether constraint c is tight (within tol) at values.
func cutIsBinding(c Constraint, values []float64, tol float64) bool {
	lhs := 0.0
	for _, e := range sortedCoeffs(c.Coeffs) {
		if e.v < len(values) {
			lhs += e.coef * values[e.v]
		}
	}
	switch c.Sense {
	case LessOrEqual:
		return c.RHS-lhs <= tol
	case GreaterOrEqual:
		return lhs-c.RHS <= tol
	case Equal:
		return math.Abs(lhs-c.RHS) <= tol
	default:
		return false
	}
}

// relaxAgedCuts returns an index-aligned copy of extras in which rows older
// than limit have been replaced by 0 <= 0 or 0 >= 0. The sense and position
// stay unchanged, preserving the auxiliary-column layout, while the empty
// coefficient map removes the row from structural matrix operations.
func relaxAgedCuts(m *Model, extras []Constraint, ages []int, limit int) []Constraint {
	if m == nil || len(ages) == 0 {
		return extras
	}
	var out []Constraint
	for i, c := range extras {
		age := 0
		if i < len(ages) {
			age = ages[i]
		}
		if age <= limit {
			continue
		}
		if c.Sense == Equal {
			continue
		}
		if out == nil {
			out = append([]Constraint(nil), extras...)
		}
		out[i].Coeffs = nil
		out[i].RHS = 0
	}
	if out == nil {
		return extras
	}
	return out
}

func countActiveCutPrefix(rows []Constraint, prefix int) int {
	if prefix > len(rows) {
		prefix = len(rows)
	}
	n := 0
	for i := 0; i < prefix; i++ {
		if len(rows[i].Coeffs) > 0 || rows[i].Sense == Equal {
			n++
		}
	}
	return n
}

// updateCutAges recomputes the age of every entry in extras (the full,
// unfiltered flattened cut history) given the LP point just solved:
// binding resets to 0, non-binding increments by 1. extras/ages must be
// index-aligned; a longer extras (new cuts just appended by this node's
// own generation, not yet present in the inherited ages) gets age 0 for
// the new tail entries.
func updateCutAges(extras []Constraint, ages []int, values []float64, tol float64) []int {
	out := make([]int, len(extras))
	for i, c := range extras {
		age := 0
		if i < len(ages) {
			age = ages[i]
		}
		if cutIsBinding(c, values, tol) {
			out[i] = 0
		} else {
			out[i] = age + 1
		}
	}
	return out
}
