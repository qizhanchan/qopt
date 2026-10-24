package qopt

import (
	"math"

	"github.com/qizhanchan/qopt/lp"
)

// Native Gomory mixed-integer (GMI) cuts off the revised simplex's bounded-
// variable basis — no extra tableau re-solve.
//
// generatePerNodeCuts previously had to re-solve a cut-generating B&B node on
// the tableau backend whenever it ran under LPBackendRevised, because only
// the tableau exposes the post-optimization simplex tableau gomoryCuts reads.
// That extra solve is the "remaining piece" the P0 roadmap flagged: this file
// computes the same GMI cut directly from the revised solver's final LU
// factorization instead.
//
// Bounded-variable GMI derivation:
//
// The revised backend keeps upper bounds on variables directly (no synthetic
// UB rows), so a non-basic column sits at either its lower or its upper
// bound. Write the reduced row for basic row i as
//
//	x_B[i] = xBar_i - Σ_j beta_ij * w_j,   w_j ≥ 0
//
// where xBar_i = s.xBasic[i] is the basic value when every non-basic sits at
// its current anchor, and per non-basic column j:
//
//	at lower:  w_j = x_j − l_j,  beta_ij = alpha_ij
//	at upper:  w_j = u_j − x_j,  beta_ij = −alpha_ij
//
// with alpha_ij = (B⁻¹A)_ij obtained via one BTRAN against the unit vector
// e_i (giving row i of B⁻¹) followed by a sparse dot product per column —
// exactly the row a dense tableau would already have materialized, computed
// on demand instead. Substituting beta_ij in place of the plain tableau
// coefficient reduces this to the same all-non-basics-at-lower formula
// gomoryCuts already uses, so the well-tested gmiContribution formula and the
// per-term translation back into user x-space are unchanged; only how beta_ij
// and which of the two w_j substitutions applies are new.
//
// Because slack/surplus columns always have upper = +Inf (only structural
// columns can carry a finite upper bound — see newRevisedSolver), the
// "at upper" substitution only ever applies to structural (colY) columns.

// revisedGomoryState bundles a completed revised-simplex solve with the
// canonicalization metadata (scale factors, row/column provenance) needed to
// translate a cut from simplex column space back to user x-space. Populated
// in two steps: extractRevisedResult sets solver on every successful solve;
// solveLPModelWithBackend fills in the rest, mirroring how it stamps
// simplexState for the tableau backend.
type revisedGomoryState struct {
	solver *lp.Solver

	scales     scaleFactors
	rowKinds   []lpRowKind
	normSenses []Sense // per-row sense AFTER the b<0 normalization flip
	flippedRow []bool  // per-row: was this row's sign flipped during normalization
	colToVar   []int
	varToCol   []int
	modelLower []float64
	modelUpper []float64
	fixed      []bool
	fixedVal   []float64
}

// revisedGomoryCuts is the revised-backend analog of gomoryCuts. Returns nil
// (never a wrong cut) whenever the metadata needed to translate a cut back to
// user x-space is missing or inconsistent — e.g. rg came from the canonCache
// fast path, which never populates it.
func revisedGomoryCuts(m *Model, modelLower, modelUpper []float64, rg *revisedGomoryState, opts Options) []Constraint {
	if rg == nil || rg.solver == nil {
		debugf("revised-gomory: no revisedState/solver — skipping")
		return nil
	}
	s := rg.solver
	sm, sn, sNCols := s.M(), s.N(), s.NCols()
	if len(rg.rowKinds) != sm || len(rg.colToVar) != sn || len(rg.normSenses) != sm || len(rg.flippedRow) != sm {
		debugf("revised-gomory: metadata length mismatch (rowKinds=%d/%d colToVar=%d/%d normSenses=%d/%d flippedRow=%d/%d) — skipping",
			len(rg.rowKinds), sm, len(rg.colToVar), sn, len(rg.normSenses), sm, len(rg.flippedRow), sm)
		return nil
	}
	// colScale is only ever sized to the structural columns [0,n) — same as
	// the tableau backend's simplexState.scales, which never indexes
	// colScale past n either (aux columns use rowScale instead; see the
	// colY-vs-slack/surplus branch below).
	if len(rg.scales.colScale) != sn || len(rg.scales.rowScale) != sm {
		debugf("revised-gomory: scale length mismatch (colScale=%d/%d rowScale=%d/%d) — skipping",
			len(rg.scales.colScale), sn, len(rg.scales.rowScale), sm)
		return nil
	}

	const fracTol = 1e-6
	const coefTol = 1e-9
	const cutCoefCap = 1e6

	if len(rg.modelLower) > 0 {
		modelLower = rg.modelLower
	}
	if len(rg.modelUpper) > 0 {
		modelUpper = rg.modelUpper
	}
	colScale := rg.scales.colScale
	rowScale := rg.scales.rowScale

	auxKind, auxRow, ok := revisedAuxColumnLayout(sNCols, sn, rg.normSenses)
	if !ok {
		debugf("revised-gomory: auxiliary layout mismatch (cols=%d structural=%d rows=%d) — skipping", sNCols, sn, len(rg.normSenses))
		return nil
	}

	inBasis := make([]bool, sNCols)
	for _, b := range s.BasicCols() {
		if b >= 0 && b < sNCols {
			inBasis[b] = true
		}
	}

	var cuts []Constraint
	for i, kBasic := range s.BasicCols() {
		if kBasic >= sn {
			continue // basic column is a slack/surplus/artificial, not structural
		}
		origBasic := rg.colToVar[kBasic]
		spec := m.vars[origBasic]
		if spec.Type != Integer && spec.Type != Binary {
			continue
		}
		lb := modelLower[origBasic]
		if math.Abs(lb-math.Round(lb)) > fracTol {
			continue
		}
		yVal := colScale[kBasic] * s.XBasic(i)
		if math.Abs(yVal-math.Round(yVal)) < fracTol {
			continue
		}
		f := yVal - math.Floor(yVal)
		if f < fracTol || f > 1-fracTol {
			continue
		}

		y := s.BinvRow(i) // row i of B^{-1}

		type colTerm struct {
			j        int
			coefUnsc float64
			phi      float64
			atUpper  bool
		}
		var terms []colTerm
		for j := 0; j < sNCols; j++ {
			if j == kBasic || inBasis[j] {
				continue
			}
			if auxKind[j] == colArtificial {
				continue // w_j = 0 at any integer-feasible point ⇒ drop
			}
			alpha := s.ColDot(j, y)
			if math.Abs(alpha) < coefTol {
				continue
			}
			atUpper := j < sn && s.StatusOf(j) == lp.ColAtUpper
			if atUpper && j >= sn {
				// Unreachable: aux columns always have upper = +Inf, so they
				// can never sit at an upper bound. Guarded defensively rather
				// than silently mistranslating an impossible case.
				continue
			}
			beta := alpha
			if atUpper {
				beta = -alpha
			}
			var coefUnsc float64
			var isInteger bool
			if j < sn {
				coefUnsc = beta * colScale[kBasic] / colScale[j]
				origJ := rg.colToVar[j]
				vt := m.vars[origJ].Type
				isInteger = vt == Integer || vt == Binary
			} else {
				coefUnsc = beta * colScale[kBasic] * rowScale[auxRow[j]]
				isInteger = false
			}
			phi := gmiContribution(coefUnsc, f, isInteger)
			if phi <= coefTol {
				continue
			}
			terms = append(terms, colTerm{j: j, coefUnsc: coefUnsc, phi: phi, atUpper: atUpper})
		}
		if len(terms) == 0 {
			continue
		}

		xCoef := make(map[Var]float64)
		var constant float64
		ok := true
		for _, t := range terms {
			if t.j < sn {
				origJ := rg.colToVar[t.j]
				if t.atUpper {
					// w_j = upper[v] − x_v.
					xCoef[Var(origJ)] -= t.phi
					constant += t.phi * modelUpper[origJ]
				} else {
					// w_j = x_v − lower[v].
					xCoef[Var(origJ)] += t.phi
					constant -= t.phi * modelLower[origJ]
				}
				continue
			}
			row := auxRow[t.j]
			if row >= len(rg.rowKinds) {
				ok = false
				break
			}
			rk := rg.rowKinds[row]
			cons := m.constraints[rk.idx]
			flipSign := 1.0
			if rg.flippedRow[row] {
				flipSign = -1.0
			}
			switch auxKind[t.j] {
			case colSlack:
				// User LE row: w_j = b_c_user − a_c · x.
				constant += t.phi * cons.RHS * flipSign
				for vid, coef := range cons.Coeffs {
					xCoef[vid] -= t.phi * flipSign * coef
				}
			case colSurplus:
				// User GE row: w_j = a_c · x − b_c_user.
				constant -= t.phi * cons.RHS * flipSign
				for vid, coef := range cons.Coeffs {
					xCoef[vid] += t.phi * flipSign * coef
				}
			default:
				ok = false
			}
			if !ok {
				break
			}
		}
		if !ok || len(xCoef) == 0 {
			continue
		}

		for v, c := range xCoef {
			if math.Abs(c) < coefTol {
				delete(xCoef, v)
			}
		}
		if len(xCoef) == 0 {
			continue
		}
		maxAbs := 0.0
		for _, c := range xCoef {
			if a := math.Abs(c); a > maxAbs {
				maxAbs = a
			}
		}
		rhs := f - constant
		if maxAbs > cutCoefCap || math.Abs(rhs) > cutCoefCap {
			continue
		}

		lhs := 0.0
		for _, e := range sortedCoeffs(xCoef) {
			lhs += e.coef * revisedXValueAt(rg, modelLower, e.v)
		}
		if lhs >= rhs-1e-6 {
			continue
		}

		cuts = append(cuts, Constraint{
			Name:   "gomory_revised",
			Coeffs: xCoef,
			Sense:  GreaterOrEqual,
			RHS:    rhs,
		})
	}
	debugf("revised-gomory: basic_rows=%d emitted=%d", len(s.BasicCols()), len(cuts))
	return cuts
}

// revisedNormalizeSenses mirrors newRevisedSolver's row-sign normalization
// (flip any row with b < 0 so every row the solver sees has bNorm ≥ 0)
// without needing an exported hook into solver construction: the flip
// decision depends only on the sign of b, which Ruiz scaling preserves
// (scale factors are positive), so it can be recomputed from the exact
// senses/b slices the backend was called with.
func revisedNormalizeSenses(senses []Sense, b []float64) (normSenses []Sense, flipped []bool) {
	normSenses = make([]Sense, len(senses))
	flipped = make([]bool, len(senses))
	for i, sense := range senses {
		if i < len(b) && b[i] < 0 {
			normSenses[i] = flipSense(sense)
			flipped[i] = true
		} else {
			normSenses[i] = sense
		}
	}
	return normSenses, flipped
}

// revisedAuxColumnLayout replicates newRevisedSolver's deterministic column
// numbering (structural columns [0,n), then per-row auxiliaries in row
// order: one slack for LE, one surplus + one artificial for GE, one
// artificial for EQ) without needing an exported hook into solver
// construction — the layout is a pure function of n and the per-row
// normalized senses.
func revisedAuxColumnLayout(nCols, n int, normSenses []Sense) (kind []colKind, row []int, ok bool) {
	needed := n
	for _, sense := range normSenses {
		switch sense {
		case LessOrEqual, Equal:
			needed++
		case GreaterOrEqual:
			needed += 2
		default:
			return nil, nil, false
		}
	}
	if needed != nCols || n < 0 || n > nCols {
		return nil, nil, false
	}
	kind = make([]colKind, nCols)
	row = make([]int, nCols)
	for j := 0; j < n; j++ {
		row[j] = -1
	}
	next := n
	for i, sense := range normSenses {
		switch sense {
		case LessOrEqual:
			kind[next], row[next] = colSlack, i
			next++
		case GreaterOrEqual:
			kind[next], row[next] = colSurplus, i
			next++
			kind[next], row[next] = colArtificial, i
			next++
		case Equal:
			kind[next], row[next] = colArtificial, i
			next++
		}
	}
	return kind, row, true
}

// revisedXValueAt returns the LP-optimum value of user variable k, the
// revised-backend analog of xValueAt.
func revisedXValueAt(rg *revisedGomoryState, modelLower []float64, k int) float64 {
	if rg == nil || rg.solver == nil || k < 0 {
		return 0
	}
	if k < len(rg.fixed) && rg.fixed[k] {
		return rg.fixedVal[k]
	}
	col := -1
	if k < len(rg.varToCol) {
		col = rg.varToCol[k]
	}
	if col < 0 {
		if k < len(modelLower) {
			return modelLower[k]
		}
		return 0
	}
	lower := 0.0
	if k < len(modelLower) {
		lower = modelLower[k]
	}
	return rg.solver.XValue(col)*rg.scales.colScale[col] + lower
}
