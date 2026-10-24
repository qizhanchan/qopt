package qopt

import (
	"context"
	"math"
)

// Gomory mixed-integer (GMI) cuts at the MIP root.
//
// The pure Gomory fractional cut is only valid when every non-basic simplex
// column corresponds to an integer-typed variable. The GMI formula extends to
// mixed integer LPs by treating continuous non-basics specially: each
// continuous column contributes its raw coefficient (positive case) or a
// rescaled negative case, while each integer column contributes a value
// derived from the fractional part of its tableau coefficient.
//
// Implementation contract:
//   - Cuts are derived in UNSCALED simplex-column space (we undo Ruiz first).
//   - Each simplex non-basic column gets a non-negative GMI contribution
//     phi_j; the cut is `sum phi_j * w_j_unscaled ≥ f`.
//   - We then translate w_j back to user x-space:
//       y_k non-basic           → w_j = x_k − lower[k]
//       slack of user LE row c  → w_j = b_c_user − a_c · x
//       surplus of user GE row c→ w_j = a_c · x − b_c_user
//       slack of UB row for var v → w_j = upper_user[v] − x_v
//       artificial              → w_j = 0  (drop)
//   - Slacks of equality rows pin to 0, so we drop them too.
//   - For v1 we treat every aux column as continuous; this gives weaker cuts
//     than recognizing integer-implied slacks, but never invalid ones.

// gomoryCuts walks every basic row of the post-Phase-II simplex state and
// emits GMI cuts in user x-space. Each cut has form `sum c_k x_k ≥ rhs`.
func gomoryCuts(m *Model, modelLower, modelUpper []float64, state *simplexState, opts Options) []Constraint {
	if state == nil {
		return nil
	}
	const fracTol = 1e-6
	const coefTol = 1e-9
	n := state.n
	totalVars := state.totalVars
	tab := state.tab
	basis := state.basis
	colKind := state.colKind
	colScale := state.scales.colScale
	rowScale := state.scales.rowScale
	colToVar := state.colToVar
	if len(colToVar) == 0 {
		colToVar = make([]int, n)
		for i := range colToVar {
			colToVar[i] = i
		}
	}
	modelLower = stateModelLower(state, modelLower)
	modelUpper = stateModelUpper(state, modelUpper)

	inBasis := make([]bool, totalVars)
	for _, b := range basis {
		if b >= 0 && b < totalVars {
			inBasis[b] = true
		}
	}

	var cuts []Constraint
	var stats cutStats
	stats.totalRows = len(basis)

	for i, kBasic := range basis {
		if kBasic >= n {
			stats.nonOriginal++
			continue
		}
		origBasic := colToVar[kBasic]
		spec := m.vars[origBasic]
		if spec.Type != Integer && spec.Type != Binary {
			stats.nonInteger++
			continue
		}
		lb := modelLower[origBasic]
		if math.Abs(lb-math.Round(lb)) > fracTol {
			stats.badLB++
			continue
		}
		yVal := colScale[kBasic] * tab[i][totalVars]
		if math.Abs(yVal-math.Round(yVal)) < fracTol {
			stats.integralBasic++
			continue
		}
		f := yVal - math.Floor(yVal)
		if f < fracTol || f > 1-fracTol {
			stats.integralBasic++
			continue
		}

		// Per simplex non-basic column j, compute its unscaled coefficient
		// and GMI contribution phi. We hold off on substituting back into
		// x-space until all phi values are known.
		type colTerm struct {
			j        int
			coefUnsc float64
			phi      float64
		}
		var terms []colTerm
		skipRow := false

		for j := 0; j < totalVars; j++ {
			if j == kBasic || inBasis[j] {
				continue
			}
			aij := tab[i][j]
			if math.Abs(aij) < coefTol {
				continue
			}
			ck := colKind[j]
			// Compute unscaled tableau coefficient for non-basic column j.
			var coefUnsc float64
			var isInteger bool
			switch ck.kind {
			case colY:
				coefUnsc = aij * colScale[kBasic] / colScale[j]
				origJ := colToVar[j]
				vt := m.vars[origJ].Type
				isInteger = vt == Integer || vt == Binary
			case colSlack:
				// Equality rows don't have a slack column (they get only an
				// artificial), so any colSlack ck.row is a row with sense
				// LE — either a user LE constraint or a UB row.
				coefUnsc = aij * colScale[kBasic] * rowScale[ck.row]
				isInteger = false // conservative
			case colSurplus:
				coefUnsc = aij * colScale[kBasic] * rowScale[ck.row]
				isInteger = false
			case colArtificial:
				// w_j = 0 at any integer-feasible point ⇒ drop.
				continue
			}
			phi := gmiContribution(coefUnsc, f, isInteger)
			if phi <= coefTol {
				continue
			}
			terms = append(terms, colTerm{j: j, coefUnsc: coefUnsc, phi: phi})
		}

		if skipRow || len(terms) == 0 {
			stats.noFracCoef++
			continue
		}

		// Now translate the cut from simplex non-basic space to user x-space.
		xCoef := make(map[Var]float64)
		var constant float64 // accumulated on the LHS

		ok := true
		for _, t := range terms {
			ck := colKind[t.j]
			switch ck.kind {
			case colY:
				// w_j = x_j − lower[j]
				origJ := colToVar[t.j]
				xCoef[Var(origJ)] += t.phi
				constant -= t.phi * modelLower[origJ]
			case colSlack:
				if !addCanonicalAuxTerm(state, ck.row, -1, t.phi, modelLower, xCoef, &constant) {
					ok = false
				}
			case colSurplus:
				if !addCanonicalAuxTerm(state, ck.row, 1, t.phi, modelLower, xCoef, &constant) {
					ok = false
				}
			}
			if !ok {
				break
			}
		}

		if !ok || len(xCoef) == 0 {
			stats.badColKind++
			continue
		}

		// Strip near-zero coefficients accumulated from cancellation.
		for v, c := range xCoef {
			if math.Abs(c) < coefTol {
				delete(xCoef, v)
			}
		}
		if len(xCoef) == 0 {
			stats.noFracCoef++
			continue
		}
		// Numerical sanity: skip cuts whose coefficients are extreme. The
		// GMI formula has a 1/(1−f) factor that can blow up when the basic
		// fractional value is very close to an integer, producing
		// huge-but-essentially-degenerate cuts. Pushing them into the LP
		// wrecks the conditioning and can manifest as "unbounded" on the
		// next solve.
		const cutCoefCap = 1e6
		maxAbs := 0.0
		for _, c := range xCoef {
			if a := math.Abs(c); a > maxAbs {
				maxAbs = a
			}
		}
		if maxAbs > cutCoefCap || math.Abs(f-constant) > cutCoefCap {
			stats.badColKind++
			continue
		}

		// Cut: sum xCoef * x ≥ f − constant.
		rhs := f - constant

		// Sanity-check: this cut must be violated by the current LP optimum.
		// We compute LP-optimal x via state.basis + tableau and check.
		lhs := 0.0
		for _, e := range sortedCoeffs(xCoef) {
			lhs += e.coef * xValueAt(state, modelLower, e.v)
		}
		if lhs >= rhs-1e-6 {
			stats.notViolated++
			continue
		}

		cuts = append(cuts, Constraint{
			Name:   "gomory_root",
			Coeffs: xCoef,
			Sense:  GreaterOrEqual,
			RHS:    rhs,
		})
	}

	debugf("gomory diag: total_rows=%d emitted=%d  skipped {non-original=%d, non-integer=%d, bad-lb=%d, integral-basic=%d, bad-col-kind=%d, no-frac-coef=%d, not-violated=%d}",
		stats.totalRows, len(cuts), stats.nonOriginal, stats.nonInteger,
		stats.badLB, stats.integralBasic, stats.badColKind,
		stats.noFracCoef, stats.notViolated)

	return cuts
}

// addCanonicalAuxTerm expands one unscaled slack/surplus column from the
// canonical y-space row that actually produced it. Using the transformed row
// rather than looking the row up in Model is essential after equality
// aggregation: the equality may have become one or two bound rows over a
// strict subset of the original columns.
//
// direction is -1 for slack = rhs - a*y and +1 for surplus = a*y - rhs.
func addCanonicalAuxTerm(state *simplexState, row, direction int, phi float64, modelLower []float64, xCoef map[Var]float64, constant *float64) bool {
	if state == nil || row < 0 || row >= len(state.rowCoeffs) || row >= len(state.rowRHS) {
		return false
	}
	flipSign := 1.0
	if row < len(state.flippedRow) && state.flippedRow[row] {
		flipSign = -1
	}
	factor := phi * float64(direction) * flipSign
	*constant -= factor * state.rowRHS[row]
	for col, coef := range state.rowCoeffs[row] {
		if coef == 0 || col >= len(state.colToVar) {
			continue
		}
		orig := state.colToVar[col]
		xCoef[Var(orig)] += factor * coef
		if orig >= 0 && orig < len(modelLower) {
			*constant -= factor * coef * modelLower[orig]
		}
	}
	return true
}

type cutStats struct {
	totalRows     int
	nonOriginal   int
	nonInteger    int
	badLB         int
	integralBasic int
	badColKind    int
	noFracCoef    int
	notViolated   int
}

// gmiContribution computes the Gomory mixed-integer cut coefficient phi_j for
// non-basic column j with unscaled tableau coefficient a and basis fractional
// part f ∈ (0, 1). phi_j is always ≥ 0; the cut is sum phi_j · w_j ≥ f.
//
// For integer non-basics (isInteger=true), let g = frac(a):
//
//	phi = g                if g ≤ f
//	phi = f·(1 − g) / (1 − f)  otherwise
//
// For continuous non-basics:
//
//	phi = a                if a ≥ 0
//	phi = −a·f / (1 − f)   otherwise
func gmiContribution(a, f float64, isInteger bool) float64 {
	if isInteger {
		g := a - math.Floor(a)
		// Normalize g into [0, 1) accounting for sign of a.
		if g < 0 {
			g += 1
		}
		if g <= f {
			return g
		}
		return f * (1 - g) / (1 - f)
	}
	if a >= 0 {
		return a
	}
	return -a * f / (1 - f)
}

// generatePerNodeCuts emits Gomory cuts derived from a B&B node's LP state
// and validates that adding them to the LP leaves the relaxation Optimal.
//
// work is the original B&B model (with global root cuts only); nodeModel
// is work augmented with the inherited cut chain that produced `state`.
// bounds is the B&B node's bound stack. nodeExtras is the flattened
// inherited cut chain — we re-augment work with nodeExtras + newCuts for
// the validation re-solve so we never mutate the caller's slices.
//
// Returns the accepted cuts (possibly truncated to perNodeCutCap), the
// validation LP objective (used by the cut cadence to detect cuts that
// didn't move the relaxation), and a boolean saying whether to attach
// them. On any validation failure the boolean is false and the cuts are
// dropped — caller continues with the pre-cut LP solution it already has.
//
// Combines Gomory cuts (derived from the simplex state of nodeModel's LP)
// with row-MIR cuts (derived from work's original user constraints and the
// LP relaxation point); the two streams are concatenated and validated as
// one batch. MIR is fed `work` rather than `nodeModel` so it only re-MIRs
// original constraints, not previously-derived cuts — those compose poorly
// numerically when re-rounded.
func generatePerNodeCuts(ctx context.Context, cache *canonCache, work, nodeModel *Model, bounds []boundConstraint, state *simplexState, revisedState *revisedGomoryState, nodeExtras []Constraint, lpValues []float64, opts Options) ([]Constraint, float64, bool) {
	var cuts []Constraint
	switch {
	case state != nil:
		modelLower, modelUpper := effectiveBounds(nodeModel, bounds)
		cuts = gomoryCuts(nodeModel, modelLower, modelUpper, state, opts)
	case revisedState != nil:
		// Native path (P0 sub-goal 3, final piece): derive the GMI cut
		// directly from the revised solver's final basis via BTRAN instead of
		// re-solving this node on the tableau backend. See revised_cuts.go.
		modelLower, modelUpper := effectiveBounds(nodeModel, bounds)
		cuts = revisedGomoryCuts(nodeModel, modelLower, modelUpper, revisedState, opts)
	default:
		// Neither backend produced usable state (e.g. metadata mismatch from
		// an unexpected caller) — fall back to a tableau re-solve so cuts
		// still fire rather than silently going dark.
		tabOpts := opts
		tabOpts.LPBackend = LPBackendTableau
		if _, tabState, terr := solveLPModelWithState(ctx, nodeModel, bounds, tabOpts); terr == nil {
			modelLower, modelUpper := effectiveBounds(nodeModel, bounds)
			cuts = gomoryCuts(nodeModel, modelLower, modelUpper, tabState, opts)
		}
	}
	if opts.UseMIRCuts {
		cuts = append(cuts, mirCuts(work, lpValues, opts)...)
	}
	if len(cuts) == 0 {
		return nil, 0, false
	}
	cap := perNodeCutCap
	if opts.PerNodeCutCap > 0 {
		cap = opts.PerNodeCutCap
	}
	// Greedy efficacy + orthogonality selection (see cutsel.go) instead of
	// keeping whichever `cap` cuts the generator happened to emit first —
	// picks the most-violated, mutually non-redundant subset so a large
	// candidate batch collapses to the handful actually worth the extra LP
	// row. No-op (returns cuts unchanged) whenever len(cuts) <= cap already.
	cuts = selectCuts(cuts, lpValues, cap)
	// Skip the validation re-solve for small cut batches when opted in. The
	// re-solve only guards against numerically poisoned cuts; the child-node
	// cold-verify + non-root Unbounded prune already catch those, so for a
	// handful of cuts it is pure overhead. Returning NaN signals the caller
	// that there is no post-cut objective to feed the adaptive cadence.
	if opts.SkipCutValidationBelow > 0 && len(cuts) <= opts.SkipCutValidationBelow {
		return cuts, math.NaN(), true
	}
	candExtras := make([]Constraint, 0, len(nodeExtras)+len(cuts))
	candExtras = append(candExtras, nodeExtras...)
	candExtras = append(candExtras, cuts...)
	candModel := modelWithExtraConstraints(work, candExtras)
	res, _, err := solveNodeOrFull(ctx, cache, candModel, candExtras, bounds, opts, nil)
	if err != nil || res.Status != Optimal {
		debugf("per-node cut validation failed (status=%v err=%v); dropping %d cuts", res.Status, err, len(cuts))
		return nil, 0, false
	}
	return cuts, res.Objective, true
}

// xValueAt returns the LP-optimum value of user variable k from the simplex
// state. Non-basic original vars sit at their lower bound; basic vars
// unscale from the tableau.
func xValueAt(state *simplexState, modelLower []float64, k int) float64 {
	if state == nil || k < 0 {
		return 0
	}
	modelLower = stateModelLower(state, modelLower)
	if k < len(state.fixed) && state.fixed[k] {
		return state.fixedVal[k]
	}
	col := k
	if len(state.varToCol) > 0 {
		if k >= len(state.varToCol) || state.varToCol[k] < 0 {
			if k < len(modelLower) {
				return modelLower[k]
			}
			return 0
		}
		col = state.varToCol[k]
	}
	if col < 0 || col >= state.n {
		return 0
	}
	for i, b := range state.basis {
		if b == col {
			return state.scales.colScale[col]*state.tab[i][state.totalVars] + modelLower[k]
		}
	}
	return modelLower[k]
}

func stateModelLower(state *simplexState, fallback []float64) []float64 {
	if state != nil && len(state.modelLower) > 0 {
		return state.modelLower
	}
	return fallback
}

func stateModelUpper(state *simplexState, fallback []float64) []float64 {
	if state != nil && len(state.modelUpper) > 0 {
		return state.modelUpper
	}
	return fallback
}
