package qopt

import "math"

const (
	cutRowprepMaxCoefficient = 1e4
	cutRowprepMinEfficacy    = 1e-4
	// cutAcceptMaxCoefficientRatio is SCIP's separating/maxcoefratio (1e4):
	// rowprep cleanup only improves a row's range toward 10/feastol, but the
	// aggregation-family separators additionally discard any cut whose final
	// dynamism exceeds this hard cap before it can reach the pool or LP.
	// Without it, iterated cut-on-cut separation accumulates rows with 1e8+
	// coefficient ranges that eventually poison the root LP factorization.
	cutAcceptMaxCoefficientRatio = 1e4
)

type cutRowprepConfig struct {
	maxCoefficientRatio float64
	maxAcceptRatio      float64
	maxCoefficient      float64
	minEfficacy         float64
	integralTolerance   float64
}

type cutRowprepResult struct {
	row                 Constraint
	success             bool
	reason              cutCandidateState
	modified            bool
	removedCoefficients int
	roundedCoefficients int
	referenceLoss       float64
	scale               float64
}

func rootCutRowprepConfig(opts Options) cutRowprepConfig {
	feasTol := opts.FeasibilityTol
	if feasTol <= 0 {
		feasTol = 1e-8
	}
	integralTol := opts.OptimalityTol
	if integralTol <= 0 {
		integralTol = 1e-9
	}
	return cutRowprepConfig{
		maxCoefficientRatio: 10 / feasTol,
		maxAcceptRatio:      cutAcceptMaxCoefficientRatio,
		maxCoefficient:      cutRowprepMaxCoefficient,
		minEfficacy:         cutRowprepMinEfficacy,
		integralTolerance:   integralTol,
	}
}

// cleanupCutRow applies the conservative subset of SCIPcleanupRowprep2 that
// q-opt can currently prove from global variable bounds. The row is converted
// to <= form, large coefficient ranges are improved by eliminating either the
// largest or smallest term with a bound, coefficients close to integers are
// rounded with a compensating side relaxation, and the row is power-of-two
// scaled into a moderate range. Every modification weakens or preserves the
// original inequality; no tolerance-only validity assumption is used.
func cleanupCutRow(m *Model, cut Constraint, values []float64, cfg cutRowprepConfig) cutRowprepResult {
	result := cutRowprepResult{row: cloneConstraint(cut), scale: 1}
	if m == nil || cut.Sense == Equal || cfg.maxCoefficientRatio < 1 || cfg.maxCoefficient < 1 {
		result.reason = cutCandidateRejectedRowprepInvalid
		return result
	}
	if cut.Sense == GreaterOrEqual {
		for v, coef := range result.row.Coeffs {
			result.row.Coeffs[v] = -coef
		}
		result.row.RHS = -result.row.RHS
		result.row.Sense = LessOrEqual
	} else if cut.Sense != LessOrEqual {
		result.reason = cutCandidateRejectedRowprepInvalid
		return result
	}
	if !cutRowFinite(result.row) {
		result.reason = cutCandidateRejectedRowprepInvalid
		return result
	}

	// Fixed variables are exact constants and should never contribute to a
	// cut's coefficient range or LP matrix work.
	for _, entry := range sortedCoeffs(result.row.Coeffs) {
		if entry.v < 0 || entry.v >= len(m.vars) {
			result.reason = cutCandidateRejectedRowprepInvalid
			return result
		}
		spec := m.vars[entry.v]
		lb, ub := *spec.Lower, *spec.Upper
		if !math.IsInf(lb, 0) && lb == ub {
			result.row.RHS -= entry.coef * lb
			delete(result.row.Coeffs, Var(entry.v))
			result.removedCoefficients++
			result.modified = true
		}
	}

	for {
		maxVar, minVar, maxCoef, minCoef := cutCoefficientExtremes(result.row.Coeffs)
		if minVar < 0 || maxCoef/minCoef <= cfg.maxCoefficientRatio {
			break
		}
		maxLoss, maxBound, maxOK := cutTermElimination(result.row.Coeffs[Var(maxVar)], m.vars[maxVar], valueAt(values, maxVar))
		minLoss, minBound, minOK := cutTermElimination(result.row.Coeffs[Var(minVar)], m.vars[minVar], valueAt(values, minVar))
		if !maxOK && !minOK {
			break
		}
		removeVar, loss, bound := maxVar, maxLoss, maxBound
		if !maxOK || minOK && minLoss < maxLoss {
			removeVar, loss, bound = minVar, minLoss, minBound
		}
		coef := result.row.Coeffs[Var(removeVar)]
		result.row.RHS -= coef * bound
		delete(result.row.Coeffs, Var(removeVar))
		result.removedCoefficients++
		result.referenceLoss += loss
		result.modified = true
	}

	if len(result.row.Coeffs) == 0 {
		result.reason = cutCandidateRejectedRowprepEmpty
		return restoreCutRowSense(result, cut.Sense)
	}

	_, _, maxCoef, _ := cutCoefficientExtremes(result.row.Coeffs)
	scale := 1.0
	if maxCoef > cfg.maxCoefficient {
		desired := cfg.maxCoefficient / maxCoef
		scale = math.Ldexp(1, int(math.Floor(math.Log2(desired))))
	} else if maxCoef < 1/cfg.maxCoefficient {
		desired := (1 / cfg.maxCoefficient) / maxCoef
		scale = math.Ldexp(1, int(math.Ceil(math.Log2(desired))))
	}
	if scale != 1 && scale > 0 && !math.IsInf(scale, 0) && !math.IsNaN(scale) {
		for v, coef := range result.row.Coeffs {
			result.row.Coeffs[v] = coef * scale
		}
		result.row.RHS *= scale
		result.scale = scale
		result.modified = true
	}

	for _, entry := range sortedCoeffs(result.row.Coeffs) {
		rounded := math.Round(entry.coef)
		delta := rounded - entry.coef
		if delta == 0 || math.Abs(delta) > cfg.integralTolerance*math.Max(1, math.Abs(entry.coef)) {
			continue
		}
		bound, ok := cutDeltaUpperBound(delta, m.vars[entry.v])
		if !ok {
			continue
		}
		ref := clampToBounds(valueAt(values, entry.v), m.vars[entry.v])
		maxDelta := delta * bound
		result.row.RHS += maxDelta
		result.referenceLoss += math.Max(0, maxDelta-delta*ref)
		if rounded == 0 {
			delete(result.row.Coeffs, Var(entry.v))
			result.removedCoefficients++
		} else {
			result.row.Coeffs[Var(entry.v)] = rounded
			result.roundedCoefficients++
		}
		result.modified = true
	}

	if len(result.row.Coeffs) == 0 {
		result.reason = cutCandidateRejectedRowprepEmpty
		return restoreCutRowSense(result, cut.Sense)
	}
	if !cutRowFinite(result.row) {
		result.reason = cutCandidateRejectedRowprepInvalid
		return restoreCutRowSense(result, cut.Sense)
	}
	_, _, maxCoef, minCoef := cutCoefficientExtremes(result.row.Coeffs)
	if maxCoef/minCoef > cfg.maxCoefficientRatio {
		result.reason = cutCandidateRejectedRowprepCoefficientRange
		return restoreCutRowSense(result, cut.Sense)
	}
	if cfg.maxAcceptRatio > 0 && maxCoef/minCoef > cfg.maxAcceptRatio {
		result.reason = cutCandidateRejectedRowprepCoefficientRange
		return restoreCutRowSense(result, cut.Sense)
	}

	result = restoreCutRowSense(result, cut.Sense)
	norm := coeffNorm(result.row.Coeffs)
	if norm < 1e-12 || cutEfficacy(result.row, values, norm) < cfg.minEfficacy {
		result.reason = cutCandidateRejectedRowprepInefficacious
		return result
	}
	result.success = true
	result.reason = cutCandidateGenerated
	return result
}

func cloneConstraint(c Constraint) Constraint {
	clone := c
	clone.Coeffs = make(map[Var]float64, len(c.Coeffs))
	for v, coef := range c.Coeffs {
		clone.Coeffs[v] = coef
	}
	return clone
}

func restoreCutRowSense(result cutRowprepResult, original Sense) cutRowprepResult {
	if original != GreaterOrEqual {
		return result
	}
	for v, coef := range result.row.Coeffs {
		result.row.Coeffs[v] = -coef
	}
	result.row.RHS = -result.row.RHS
	result.row.Sense = GreaterOrEqual
	return result
}

func cutRowFinite(row Constraint) bool {
	if math.IsNaN(row.RHS) || math.IsInf(row.RHS, 0) {
		return false
	}
	for _, coef := range row.Coeffs {
		if coef == 0 || math.IsNaN(coef) || math.IsInf(coef, 0) {
			return false
		}
	}
	return true
}

func cutCoefficientExtremes(coefs map[Var]float64) (maxVar, minVar int, maxCoef, minCoef float64) {
	maxVar, minVar = -1, -1
	minCoef = math.Inf(1)
	for _, entry := range sortedCoeffs(coefs) {
		absCoef := math.Abs(entry.coef)
		if absCoef > maxCoef {
			maxCoef = absCoef
			maxVar = entry.v
		}
		if absCoef < minCoef {
			minCoef = absCoef
			minVar = entry.v
		}
	}
	return maxVar, minVar, maxCoef, minCoef
}

// cutTermElimination returns the efficacy loss at ref and the bound used to
// eliminate coef*x from a <= row. The new side is rhs - coef*bound.
func cutTermElimination(coef float64, spec VarSpec, ref float64) (loss, bound float64, ok bool) {
	ref = clampToBounds(ref, spec)
	if coef > 0 {
		bound = *spec.Lower
		if math.IsInf(bound, -1) {
			return 0, 0, false
		}
		return math.Max(0, coef*(ref-bound)), bound, true
	}
	bound = *spec.Upper
	if math.IsInf(bound, 1) {
		return 0, 0, false
	}
	return math.Max(0, -coef*(bound-ref)), bound, true
}

func cutDeltaUpperBound(delta float64, spec VarSpec) (float64, bool) {
	if delta > 0 {
		bound := *spec.Upper
		return bound, !math.IsInf(bound, 1)
	}
	bound := *spec.Lower
	return bound, !math.IsInf(bound, -1)
}

func clampToBounds(value float64, spec VarSpec) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		value = 0
	}
	return math.Max(*spec.Lower, math.Min(*spec.Upper, value))
}

func valueAt(values []float64, idx int) float64 {
	if idx < 0 || idx >= len(values) {
		return 0
	}
	return values[idx]
}
