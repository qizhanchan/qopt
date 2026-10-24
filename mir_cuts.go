package qopt

import (
	"fmt"
	"math"
	"sort"
)

// Marchand-Wolsey single-row MIR cuts.
//
// For a mixed-integer LE row `sum_I a_j x_j + sum_C c_j y_j ≤ b` with integer
// variables x_j ≥ 0 and continuous variables y_j ≥ 0, the MIR cut at scaling
// factor k is
//
//   sum_I φ_int(k·a_j) · x_j + sum_C φ_cont(k·c_j) · y_j ≤ floor(k·b)
//
//   φ_int(α)    = floor(α) + max(0, frac(α) − f_0) / (1 − f_0)
//   φ_cont(γ)   = γ / (1 − f_0) when γ < 0, else 0
//
// where f_0 = frac(k·b) ∈ (0,1). This recovers Chvátal / 0-1/2 cuts on
// pure-integer equality constraints with odd RHS (k = 1/2) and the
// continuous-slack variant that pk1 / markshare-class instances actually
// need (slack vars y with coef −1 carry through with cut-coef −1/(1−f_0)).
//
// EQ rows are processed in both LE directions (a^T x ≤ b and -a^T x ≤ -b);
// GE rows negate to LE.
//
// Limitations (v1):
//   - every variable in the row must have lower bound 0 (binary always does;
//     non-zero-lower vars would need a y = x − lower shift before applying
//     the formula and translating coefficients back — deferred)
//
// Composes with both LP backends because the generator only consumes user
// constraints and the LP-relaxation point — no simplex state needed.

// mirScales are the scaling factors `k` tried for each eligible row.
// {1, 1/2} covers Chvátal cuts on already-fractional rows and 0-1/2 cuts
// on odd-RHS integer rows. We previously tried {1/3..1/6} too, but those
// produce denominators (1-f_0) too small for stable downstream LP solves;
// the minDenom guard inside singleRowMIR rejects any individual cut whose
// denominator falls below 0.25, which also keeps k=1 honest on rows like
// gen-ip054's RHS = −210.189022 (f_0 ≈ 0.81). CBC's CglMixedIntegerRounding2
// searches a much larger set driven by row structure rather than a fixed
// list — v1 stops here.
var mirScales = []float64{1.0, 0.5}

func mirCuts(m *Model, lpValues []float64, opts Options) []Constraint {
	if !opts.UseMIRCuts {
		return nil
	}
	cuts := singleRowMIRCuts(m, lpValues, opts)
	cuts = append(cuts, pairwiseAggregationMIRCuts(m, lpValues, opts)...)
	cuts = append(cuts, aggregationCMIRCuts(m, lpValues, opts)...)
	return cuts
}

type mirAggregationRow struct {
	cIdx  int
	tag   string
	coefs map[Var]float64
	rhs   float64
	score float64
}

func singleRowMIRCuts(m *Model, lpValues []float64, opts Options) []Constraint {
	const violationTol = 1e-6
	var cuts []Constraint
	considered, ineligible, hasInteger := 0, 0, 0

	for ci, cons := range m.constraints {
		considered++
		if !mirEligible(m, cons) {
			ineligible++
			continue
		}
		// MIR is only meaningful on rows with at least one integer-typed var
		// — pure-continuous rows produce no rounding gain.
		anyInt := false
		for v := range cons.Coeffs {
			if t := m.vars[v].Type; t == Integer || t == Binary {
				anyInt = true
				break
			}
		}
		if !anyInt {
			continue
		}
		hasInteger++
		for _, variant := range mirLEForms(cons) {
			for _, k := range mirScales {
				cut, ok := singleRowMIR(m, variant.coefs, variant.rhs, k, lpValues, violationTol)
				if !ok {
					continue
				}
				cut.Name = fmt.Sprintf("mir_c%d_%s_k%g", ci, variant.tag, k)
				cuts = append(cuts, cut)
			}
		}
	}
	debugf("mir single: considered=%d ineligible=%d had-integer=%d emitted=%d", considered, ineligible, hasInteger, len(cuts))
	return cuts
}

// pairwiseAggregationMIRCuts produces MIR cuts from linear combinations
// `λ_i · row_i + λ_j · row_j` with positive multipliers. Each input
// constraint contributes its LE-form(s) — EQ rows contribute two directions
// (a^T x ≤ b and −a^T x ≤ −b), LE/GE each contribute one. Combined with
// the existing mirScales this covers the aggregations that fire on
// markshare-style problems (pk1 / markshare_4_0) where single-row MIR
// can't violate the LP root.
//
// The number of aggregations is O(R²) in the row count, so we cap at
// `maxAggCandidates` attempts per call. CBC's CglMixedIntegerRounding2
// builds aggregations driven by the LP basis; this is the cheap fixed-
// multiplier alternative.
func pairwiseAggregationMIRCuts(m *Model, lpValues []float64, opts Options) []Constraint {
	const violationTol = 1e-6
	const maxAggCandidates = 5000
	var cuts []Constraint

	type leRow struct {
		cIdx  int
		tag   string
		coefs map[Var]float64
		rhs   float64
	}
	var rows []leRow
	for ci, cons := range m.constraints {
		if !mirEligible(m, cons) {
			continue
		}
		anyInt := false
		for v := range cons.Coeffs {
			if t := m.vars[v].Type; t == Integer || t == Binary {
				anyInt = true
				break
			}
		}
		if !anyInt {
			continue
		}
		for _, v := range mirLEForms(cons) {
			rows = append(rows, leRow{cIdx: ci, tag: v.tag, coefs: v.coefs, rhs: v.rhs})
		}
	}
	if len(rows) < 2 {
		return nil
	}

	// Multiplier pairs. The asymmetric (1, 1/2) and (1/2, 1) create the
	// fractional RHS that makes MIR fire at k=1; (1, 1) handles cases where
	// the sum's k=1/2 reduction is sharp enough.
	multipliers := [][2]float64{{1, 1}, {1, 0.5}, {0.5, 1}}

	attempts := 0
outer:
	for i := 0; i < len(rows); i++ {
		for j := i + 1; j < len(rows); j++ {
			// Skip pairing a row with its own negation (same cIdx, opposite
			// tags from one EQ). Their sum is 0 ≤ 0 — trivial.
			if rows[i].cIdx == rows[j].cIdx {
				continue
			}
			for _, mult := range multipliers {
				if attempts >= maxAggCandidates {
					break outer
				}
				attempts++

				aggCoefs, aggRHS := aggregateRow(rows[i].coefs, rows[i].rhs, rows[j].coefs, rows[j].rhs, mult[0], mult[1])
				for _, k := range mirScales {
					cut, ok := singleRowMIR(m, aggCoefs, aggRHS, k, lpValues, violationTol)
					if !ok {
						continue
					}
					cut.Name = fmt.Sprintf("mir_agg_c%d_%s_c%d_%s_l%g_l%g_k%g",
						rows[i].cIdx, rows[i].tag, rows[j].cIdx, rows[j].tag,
						mult[0], mult[1], k)
					cuts = append(cuts, cut)
				}
			}
		}
	}
	debugf("mir agg: rows=%d attempts=%d emitted=%d", len(rows), attempts, len(cuts))
	return cuts
}

func aggregateRow(coefsA map[Var]float64, rhsA float64, coefsB map[Var]float64, rhsB, la, lb float64) (map[Var]float64, float64) {
	out := make(map[Var]float64, len(coefsA)+len(coefsB))
	for v, c := range coefsA {
		out[v] = la * c
	}
	for v, c := range coefsB {
		out[v] += lb * c
	}
	return out, la*rhsA + lb*rhsB
}

// aggregationCMIRCuts is the smallest useful subset of SCIP's aggregation
// separator: start from a nearly-tight row, cancel one continuous variable
// with a nonnegative multiple of another LE row, then run complemented MIR on
// the resulting row. Equality constraints already contribute both LE
// orientations, so the nonnegative-multiplier rule covers either equality
// sign without ever multiplying a one-sided inequality by a negative value.
func aggregationCMIRCuts(m *Model, lpValues []float64, _ Options) []Constraint {
	const (
		maxAttempts    = 5000
		maxRowFactor   = 1e4
		maxSlack       = 0.1
		violationTol   = 1e-6
		coefficientTol = 1e-10
	)
	if len(lpValues) < len(m.vars) {
		return nil
	}

	rows := make([]mirAggregationRow, 0, len(m.constraints))
	rowsByContinuous := make(map[Var][]int)
	maxRowNNZ := int(0.05*float64(len(m.vars))) + 100
	for ci, cons := range m.constraints {
		for _, form := range mirLEForms(cons) {
			if len(form.coefs) == 0 || len(form.coefs) > maxRowNNZ {
				continue
			}
			activity, norm, fracActivity, anyInt := mirRowMetrics(m, form.coefs, lpValues)
			if !anyInt || norm < coefficientTol || fracActivity <= coefficientTol {
				continue
			}
			slack := (form.rhs - activity) / math.Max(norm, 0.1)
			if slack < -1e-6 || slack > maxSlack {
				continue
			}
			rowIdx := len(rows)
			rows = append(rows, mirAggregationRow{
				cIdx: ci, tag: form.tag, coefs: form.coefs, rhs: form.rhs,
				score: fracActivity / math.Max(norm, 0.1),
			})
			for v, coef := range form.coefs {
				if m.vars[v].Type == Continuous && math.Abs(coef) > coefficientTol && mirBoundDistance(m.vars[v], lpValues[v]) > 1e-7 {
					rowsByContinuous[v] = append(rowsByContinuous[v], rowIdx)
				}
			}
		}
	}
	if len(rows) < 2 {
		return nil
	}

	order := make([]int, len(rows))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		return rows[order[i]].score > rows[order[j]].score
	})

	type badVar struct {
		v    Var
		dist float64
	}
	var cuts []Constraint
	attempts := 0
	diag := cmirDiagnostics{}
outer:
	for _, startIdx := range order {
		start := rows[startIdx]
		bad := make([]badVar, 0)
		for v, coef := range start.coefs {
			if m.vars[v].Type != Continuous || math.Abs(coef) <= coefficientTol {
				continue
			}
			if dist := mirBoundDistance(m.vars[v], lpValues[v]); dist > 1e-7 {
				bad = append(bad, badVar{v: v, dist: dist})
			}
		}
		sort.SliceStable(bad, func(i, j int) bool { return bad[i].dist > bad[j].dist })

		for _, target := range bad {
			a := start.coefs[target.v]
			for _, otherIdx := range rowsByContinuous[target.v] {
				if otherIdx == startIdx || rows[otherIdx].cIdx == start.cIdx {
					continue
				}
				b := rows[otherIdx].coefs[target.v]
				factor := -a / b
				if factor <= coefficientTol || factor > maxRowFactor {
					continue
				}
				if attempts >= maxAttempts {
					break outer
				}
				attempts++

				aggCoefs, aggRHS := aggregateRow(start.coefs, start.rhs, rows[otherIdx].coefs, rows[otherIdx].rhs, 1, factor)
				cleanSmallCoefficients(aggCoefs, coefficientTol)
				if len(aggCoefs) == 0 || len(aggCoefs) > int(0.2*float64(len(m.vars)))+100 {
					continue
				}
				cut, ok := complementedMIRBestWithDiagnostics(m, aggCoefs, aggRHS, lpValues, violationTol, &diag)
				if !ok {
					continue
				}
				cut.Name = fmt.Sprintf("cmir_agg_c%d_%s_c%d_%s_v%d_l%g",
					start.cIdx, start.tag, rows[otherIdx].cIdx, rows[otherIdx].tag, target.v, factor)
				cuts = append(cuts, cut)
			}
		}
	}
	debugf("cmir agg: rows=%d attempts=%d transformed=%d scales=%d rounded=%d violated=%d emitted=%d",
		len(rows), attempts, diag.transformed, diag.scales, diag.rounded, diag.violated, len(cuts))
	return cuts
}

func mirRowMetrics(m *Model, coefs map[Var]float64, lpValues []float64) (activity, norm, fracActivity float64, anyInt bool) {
	for _, e := range sortedCoeffs(coefs) {
		activity += e.coef * lpValues[e.v]
		norm += e.coef * e.coef
		if t := m.vars[e.v].Type; t == Integer || t == Binary {
			anyInt = true
			frac := math.Abs(lpValues[e.v] - math.Round(lpValues[e.v]))
			fracActivity += math.Abs(e.coef) * frac
		}
	}
	return activity, math.Sqrt(norm), fracActivity, anyInt
}

func mirBoundDistance(spec VarSpec, value float64) float64 {
	lower, upper := *spec.Lower, *spec.Upper
	dist := math.Inf(1)
	if !math.IsInf(lower, -1) {
		dist = math.Max(0, value-lower)
	}
	if !math.IsInf(upper, 1) {
		dist = math.Min(dist, math.Max(0, upper-value))
	}
	if math.IsInf(dist, 1) {
		return 0
	}
	return dist
}

func cleanSmallCoefficients(coefs map[Var]float64, tol float64) {
	for v, coef := range coefs {
		if math.Abs(coef) <= tol {
			delete(coefs, v)
		}
	}
}

// mirEligible enforces the lower-bound-0 precondition on every variable in
// the row. Type (integer vs continuous) is dispatched inside singleRowMIR.
func mirEligible(m *Model, c Constraint) bool {
	if len(c.Coeffs) == 0 {
		return false
	}
	for v := range c.Coeffs {
		spec := m.vars[v]
		// Binary lower is implicitly 0; integer / continuous must declare 0.
		if spec.Type != Binary && *spec.Lower != 0 {
			return false
		}
	}
	return true
}

type mirLEForm struct {
	coefs map[Var]float64
	rhs   float64
	tag   string
}

func mirLEForms(c Constraint) []mirLEForm {
	switch c.Sense {
	case LessOrEqual:
		return []mirLEForm{{coefs: c.Coeffs, rhs: c.RHS, tag: "le"}}
	case GreaterOrEqual:
		return []mirLEForm{{coefs: negateCoefs(c.Coeffs), rhs: -c.RHS, tag: "ge"}}
	case Equal:
		return []mirLEForm{
			{coefs: c.Coeffs, rhs: c.RHS, tag: "eq+"},
			{coefs: negateCoefs(c.Coeffs), rhs: -c.RHS, tag: "eq-"},
		}
	}
	return nil
}

func negateCoefs(coefs map[Var]float64) map[Var]float64 {
	out := make(map[Var]float64, len(coefs))
	for v, c := range coefs {
		out[v] = -c
	}
	return out
}

type mirBoundTransform struct {
	bound float64
	upper bool
}

type cmirDiagnostics struct {
	transformed int
	scales      int
	rounded     int
	violated    int
}

// complementedMIRBest transforms every variable to its closest finite bound,
// tests the small delta set used by SCIP's cMIR heuristic, and returns the
// most efficacious separating cut translated back to the original variables.
func complementedMIRBest(m *Model, coefs map[Var]float64, rhs float64, lpValues []float64, violationTol float64) (Constraint, bool) {
	return complementedMIRBestWithDiagnostics(m, coefs, rhs, lpValues, violationTol, nil)
}

func complementedMIRBestWithDiagnostics(m *Model, coefs map[Var]float64, rhs float64, lpValues []float64, violationTol float64, diag *cmirDiagnostics) (Constraint, bool) {
	transformed, transformedRHS, transforms, ok := complementMIRRow(m, coefs, rhs, lpValues)
	if !ok {
		return Constraint{}, false
	}
	if diag != nil {
		diag.transformed++
	}

	type deltaCandidate struct {
		delta float64
		dist  float64
	}
	deltas := []deltaCandidate{{delta: 1}}
	maxAbs := 0.0
	for v, coef := range transformed {
		if t := m.vars[v].Type; t != Integer && t != Binary {
			continue
		}
		z := lpValues[v] - transforms[v].bound
		if transforms[v].upper {
			z = transforms[v].bound - lpValues[v]
		}
		if z <= 1e-7 {
			continue
		}
		absCoef := math.Abs(coef)
		if absCoef <= 1e-10 {
			continue
		}
		deltas = append(deltas, deltaCandidate{delta: absCoef, dist: mirBoundDistance(m.vars[v], lpValues[v])})
		maxAbs = math.Max(maxAbs, absCoef)
	}
	if maxAbs > 0 {
		deltas = append(deltas, deltaCandidate{delta: maxAbs + 1})
	}
	sort.SliceStable(deltas, func(i, j int) bool { return deltas[i].dist > deltas[j].dist })
	if len(deltas) > 16 {
		deltas = deltas[:16]
	}

	seenScale := make(map[int64]bool)
	bestEfficacy := 0.0
	var best Constraint
	evaluateScale := func(scale float64) {
		if scale <= 0 || math.IsInf(scale, 0) || math.IsNaN(scale) {
			return
		}
		key := int64(math.Round(scale * 1e9))
		if seenScale[key] {
			return
		}
		seenScale[key] = true
		if diag != nil {
			diag.scales++
		}
		cut, rounded := roundedMIRZeroLower(m, transformed, transformedRHS, scale, 0.05)
		if !rounded {
			return
		}
		if diag != nil {
			diag.rounded++
		}
		cut = translateMIRCut(cut, transforms)
		cut = normalizeMIRCut(cut)
		norm := coeffNorm(cut.Coeffs)
		if norm <= 1e-12 {
			return
		}
		efficacy := cutEfficacy(cut, lpValues, norm)
		if efficacy <= violationTol || efficacy <= bestEfficacy {
			return
		}
		if diag != nil {
			diag.violated++
		}
		bestEfficacy = efficacy
		best = cut
	}

	// SCIPcalcIntegralScalar supplies large scales on decimal data such as
	// neos17's 1.001 coefficients. The nearest-integer distance recovers the
	// same useful candidates without a 1..10000 brute-force denominator scan.
	for v, coef := range transformed {
		if t := m.vars[v].Type; t != Integer && t != Binary {
			continue
		}
		fracDist := math.Abs(coef - math.Round(coef))
		if fracDist > 1e-8 {
			evaluateScale(1 / fracDist)
		}
	}
	rhsFracDist := math.Abs(transformedRHS - math.Round(transformedRHS))
	if rhsFracDist > 1e-8 {
		evaluateScale(0.25 / rhsFracDist)
		evaluateScale(0.5 / rhsFracDist)
	}
	for _, candidate := range deltas {
		if candidate.delta <= 1e-12 || math.IsInf(candidate.delta, 0) || math.IsNaN(candidate.delta) {
			continue
		}
		baseScale := 1 / candidate.delta
		for _, multiplier := range []float64{1, 2, 4, 8} {
			evaluateScale(baseScale * multiplier)
		}
	}
	return best, bestEfficacy > violationTol
}

func normalizeMIRCut(cut Constraint) Constraint {
	maxAbs := 0.0
	for _, coef := range cut.Coeffs {
		maxAbs = math.Max(maxAbs, math.Abs(coef))
	}
	if maxAbs <= 1 || math.IsInf(maxAbs, 0) || math.IsNaN(maxAbs) {
		return cut
	}
	for v, coef := range cut.Coeffs {
		cut.Coeffs[v] = coef / maxAbs
	}
	cut.RHS /= maxAbs
	return cut
}

func complementMIRRow(m *Model, coefs map[Var]float64, rhs float64, lpValues []float64) (map[Var]float64, float64, map[Var]mirBoundTransform, bool) {
	transformed := make(map[Var]float64, len(coefs))
	transforms := make(map[Var]mirBoundTransform, len(coefs))
	transformedRHS := rhs
	for _, e := range sortedCoeffs(coefs) {
		v := Var(e.v)
		spec := m.vars[v]
		lower, upper := *spec.Lower, *spec.Upper
		useUpper := false
		switch {
		case math.IsInf(lower, -1) && math.IsInf(upper, 1):
			return nil, 0, nil, false
		case math.IsInf(lower, -1):
			useUpper = true
		case !math.IsInf(upper, 1):
			useUpper = lpValues[e.v] > 0.5*(lower+upper)
		}
		if useUpper {
			transformed[v] = -e.coef
			transformedRHS -= e.coef * upper
			transforms[v] = mirBoundTransform{bound: upper, upper: true}
		} else {
			transformed[v] = e.coef
			transformedRHS -= e.coef * lower
			transforms[v] = mirBoundTransform{bound: lower}
		}
	}
	return transformed, transformedRHS, transforms, true
}

func translateMIRCut(cut Constraint, transforms map[Var]mirBoundTransform) Constraint {
	coefs := make(map[Var]float64, len(cut.Coeffs))
	rhs := cut.RHS
	for _, e := range sortedCoeffs(cut.Coeffs) {
		v := Var(e.v)
		transform := transforms[v]
		if transform.upper {
			coefs[v] = -e.coef
			rhs -= e.coef * transform.bound
		} else {
			coefs[v] = e.coef
			rhs += e.coef * transform.bound
		}
	}
	return Constraint{Coeffs: coefs, Sense: LessOrEqual, RHS: rhs}
}

// roundedMIRZeroLower applies MIR to a row whose variables have already been
// shifted or complemented to nonnegative space. minDenom is deliberately
// configurable: legacy row-MIR keeps its conservative 0.25 floor, while cMIR
// delta search can safely explore down to 0.05 under the same coefficient cap.
func roundedMIRZeroLower(m *Model, coefs map[Var]float64, rhs, scale, minDenom float64) (Constraint, bool) {
	const fracTol = 1e-9
	const cutCoefCap = 1e6
	if scale <= 0 || math.IsInf(scale, 0) || math.IsNaN(scale) {
		return Constraint{}, false
	}
	scaledRHS := scale * rhs
	f0 := scaledRHS - math.Floor(scaledRHS)
	if f0 < fracTol || f0 > 1-fracTol {
		return Constraint{}, false
	}
	denom := 1 - f0
	if f0 < 0.05 || denom < minDenom || scale/denom > cutCoefCap {
		return Constraint{}, false
	}

	cutCoefs := make(map[Var]float64, len(coefs))
	for v, a := range coefs {
		scaled := scale * a
		var coef float64
		if t := m.vars[v].Type; t == Integer || t == Binary {
			fraction := scaled - math.Floor(scaled)
			coef = math.Floor(scaled)
			if fraction > f0+fracTol {
				coef += (fraction - f0) / denom
			}
		} else if scaled < -fracTol {
			coef = scaled / denom
		}
		if math.Abs(coef) > fracTol {
			if math.Abs(coef) > cutCoefCap {
				return Constraint{}, false
			}
			cutCoefs[v] = coef
		}
	}
	if len(cutCoefs) == 0 {
		return Constraint{}, false
	}
	cutRHS := math.Floor(scaledRHS)
	if math.Abs(cutRHS) > cutCoefCap {
		return Constraint{}, false
	}
	return Constraint{Coeffs: cutCoefs, Sense: LessOrEqual, RHS: cutRHS}, true
}

// singleRowMIR returns the MIR cut for `sum coefs[v] * x_v ≤ rhs` after
// scaling by k and rounding. Variable types are read from m: integer/binary
// vars take the φ_int formula, continuous vars take φ_cont. ok is true only
// when the cut is non-trivial AND violated by the supplied LP-relaxation
// point.
func singleRowMIR(m *Model, coefs map[Var]float64, rhs, k float64, lpValues []float64, violationTol float64) (Constraint, bool) {
	cut, ok := roundedMIRZeroLower(m, coefs, rhs, k, 0.25)
	if !ok {
		return Constraint{}, false
	}
	lhs := 0.0
	for _, e := range sortedCoeffs(cut.Coeffs) {
		if e.v >= len(lpValues) {
			return Constraint{}, false
		}
		lhs += e.coef * lpValues[e.v]
	}
	if lhs <= cut.RHS+violationTol {
		return Constraint{}, false
	}
	return cut, true
}
