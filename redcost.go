package qopt

import (
	"math"

	"github.com/qizhanchan/qopt/lp"
)

// pseudoObjectiveBound computes the "box relaxation" objective: every
// variable is pinned to whichever of its current lower/upper bound most
// helps the objective, with every constraint row ignored entirely. Dropping
// rows can only enlarge the feasible region, so for Minimize this is a valid
// (if usually much weaker) lower bound on the true LP-relaxation optimum —
// and symmetrically a valid upper bound for Maximize — computed in O(n) with
// no matrix work at all. When a variable's objective coefficient points
// toward an infinite bound the contribution is ±Inf in the direction that
// makes the overall value uninformative for pruning (never the informative
// direction, and never a mix of +Inf and -Inf — see the two per-direction
// cases below), so passing the result straight into boundCannotImprove
// already does the right thing with no special-casing.
func pseudoObjectiveBound(m *Model, bounds []boundConstraint) float64 {
	lower, upper := effectiveBounds(m, bounds)
	total := 0.0
	for i := range m.vars {
		c := m.objective[Var(i)]
		if c == 0 {
			continue
		}
		var pick float64
		switch {
		case m.direction == Minimize && c > 0, m.direction == Maximize && c < 0:
			pick = lower[i]
		default:
			pick = upper[i]
		}
		total += c * pick
	}
	return total
}

// propagateObjectiveCutoff treats the incumbent-improvement requirement as a
// single <= row and derives variable bounds from its minimum activity. Unlike
// pseudoObjectiveBound, which only prunes when the whole box cannot improve,
// this is SCIP-style pseudo-objective propagation: each finite "all other
// variables" activity can tighten the remaining variable. The caller may
// alternate this with ordinary row propagation to reach a small fixpoint.
func propagateObjectiveCutoff(m *Model, lower, upper []float64, incumbent, tol float64) (reductions int, infeasible bool) {
	if len(lower) != len(m.vars) || len(upper) != len(m.vars) || math.IsNaN(incumbent) || math.IsInf(incumbent, 0) {
		return 0, false
	}
	if tol <= 0 {
		tol = 1e-8
	}
	sign := 1.0
	rhs := incumbent - tol
	if m.direction == Maximize {
		sign = -1
		rhs = -incumbent - tol
	}

	minActivity := 0.0
	minInf := 0
	for i := range m.vars {
		a := sign * m.objective[Var(i)]
		if a == 0 {
			continue
		}
		pick := lower[i]
		if a < 0 {
			pick = upper[i]
		}
		if math.IsInf(pick, 0) {
			minInf++
			continue
		}
		minActivity += a * pick
	}
	if minInf == 0 && minActivity > rhs+tol {
		return 0, true
	}

	for i, spec := range m.vars {
		a := sign * m.objective[Var(i)]
		if a == 0 {
			continue
		}
		pick := lower[i]
		if a < 0 {
			pick = upper[i]
		}
		ownInf := 0
		ownContribution := 0.0
		if math.IsInf(pick, 0) {
			ownInf = 1
		} else {
			ownContribution = a * pick
		}
		if minInf-ownInf != 0 {
			continue
		}
		candidate := (rhs - (minActivity - ownContribution)) / a
		if math.IsNaN(candidate) || math.IsInf(candidate, 0) {
			continue
		}
		integral := spec.Type == Integer || spec.Type == Binary
		if a > 0 {
			if integral {
				candidate = math.Floor(candidate + tol)
			}
			if candidate < upper[i]-tol {
				upper[i] = candidate
				reductions++
			}
		} else {
			if integral {
				candidate = math.Ceil(candidate - tol)
			}
			if candidate > lower[i]+tol {
				lower[i] = candidate
				reductions++
			}
		}
		if upper[i] < lower[i]-tol {
			return reductions, true
		}
	}
	return reductions, false
}

// reducedCostBounds derives variable-bound tightenings from a solved
// revised-simplex node's reduced costs, valid whenever a finite incumbent
// bound is known. This is the classical LP-duality reduced-cost-fixing
// argument (Nemhauser & Wolsey): writing the LP in standard form Ax=b,
// l≤x≤u with optimal dual y and reduced costs d_j = c_j − y^T A_j, the
// identity c^Tx = Σ_j d_j x_j + y^Tb holds for every x satisfying Ax=b (not
// just the optimum), and dual feasibility (d_j ≥ 0 at a lower-bound
// nonbasic, ≤ 0 at an upper-bound nonbasic, for a MIN-sense cost) means
// every OTHER nonbasic variable's term is already at its most favorable
// extreme. Isolating one variable x_k and requiring the objective to still
// beat the incumbent (gap := |bestObj − lpObj| > 0) yields:
//
//	x_k < l_k + gap/d_k   (nonbasic at lower, d_k > 0)
//	x_k > u_k + gap/d_k   (nonbasic at upper, d_k < 0)
//
// derived here in a MIN-sense internal convention that is direction-
// agnostic in practice: the revised backend's cost vector is already
// negated upstream for Minimize models (see lp_backend.go), so converting a
// column's scaled reduced cost back through colScale into this MIN-sense
// convention takes the same formula regardless of m.direction — see the
// dMin computation below.
//
// Returns nil whenever the revised state is missing/unusable, or the gap is
// not worth acting on — never a bound that could exclude the true optimum
// (any resulting infeasibility, e.g. crossing an existing opposite bound, is
// caught downstream by the ordinary tightenBounds/canonicalize checks that
// already run at every node).
func reducedCostBounds(m *Model, rg *revisedGomoryState, direction Direction, lpObj, bestObj float64, opts Options) []boundConstraint {
	if rg == nil || rg.solver == nil {
		return nil
	}
	s := rg.solver
	sn := s.N()
	if len(rg.colToVar) != sn || len(rg.modelLower) != len(m.vars) || len(rg.modelUpper) != len(m.vars) {
		return nil
	}
	if len(rg.scales.colScale) != sn {
		return nil
	}

	var gap float64
	if direction == Minimize {
		gap = bestObj - lpObj
	} else {
		gap = lpObj - bestObj
	}
	tol := opts.FeasibilityTol
	if tol <= 0 {
		tol = 1e-8
	}
	if gap <= tol {
		// The LP bound already meets/exceeds the incumbent (or is only
		// noise away from it) — boundCannotImprove would prune this node
		// outright, so there is nothing to derive a tighter bound from.
		return nil
	}

	y := s.DualValues()
	colScale := rg.scales.colScale

	var out []boundConstraint
	for j := 0; j < sn; j++ {
		st := s.StatusOf(j)
		if st != lp.ColAtLower && st != lp.ColAtUpper {
			continue // basic columns carry a zero reduced cost by definition
		}
		origJ := rg.colToVar[j]
		if origJ < 0 || origJ >= len(m.vars) {
			continue
		}
		cs := colScale[j]
		if cs == 0 {
			continue
		}
		// d_scaled is in the solver's scaled max-sense convention (s.cost
		// already carries the Minimize-negation, if any, from
		// lp_backend.go). Real objective value is scale-invariant by
		// construction (cScaled_j·ŷ_j = c_j·y_j pointwise), so unscaling a
		// reduced cost — a rate of change with respect to the variable, not
		// the objective itself — divides by colScale_j; the extra sign flip
		// folds the max-sense solver convention back to the MIN-sense
		// convention this function's derivation above uses, uniformly
		// across both Minimize and Maximize models.
		dScaled := s.ReducedCost(j, y)
		dMin := -dScaled / cs

		switch st {
		case lp.ColAtLower:
			if dMin <= tol {
				continue
			}
			x0 := rg.modelLower[origJ]
			cand := x0 + gap/dMin
			cur := rg.modelUpper[origJ]
			if cand < cur-tol {
				out = append(out, boundConstraint{
					varIndex: origJ,
					sense:    LessOrEqual,
					rhs:      roundFixingBound(m, origJ, cand, false),
				})
			}
		case lp.ColAtUpper:
			if dMin >= -tol {
				continue
			}
			x0 := rg.modelUpper[origJ]
			cand := x0 + gap/dMin
			cur := rg.modelLower[origJ]
			if cand > cur+tol {
				out = append(out, boundConstraint{
					varIndex: origJ,
					sense:    GreaterOrEqual,
					rhs:      roundFixingBound(m, origJ, cand, true),
				})
			}
		}
	}
	return out
}

// roundFixingBound rounds a derived bound inward for integer/binary
// variables, mirroring tightenBounds's integral rounding: a fractional
// upper bound can always be floored (no integer point sits above it
// anyway) and a fractional lower bound can always be ceiled.
func roundFixingBound(m *Model, idx int, v float64, isLower bool) float64 {
	spec := m.vars[idx]
	if spec.Type != Integer && spec.Type != Binary {
		return v
	}
	const tol = 1e-9
	if isLower {
		return math.Ceil(v - tol)
	}
	return math.Floor(v + tol)
}

// appendBoundsAll extends src with every entry in extra, then final. Used to
// attach reduced-cost-derived bound tightenings to a child node alongside
// its branching bound in one allocation.
func appendBoundsAll(src []boundConstraint, extra []boundConstraint, final boundConstraint) []boundConstraint {
	dst := make([]boundConstraint, len(src), len(src)+len(extra)+1)
	copy(dst, src)
	dst = append(dst, extra...)
	dst = append(dst, final)
	return dst
}
