package qopt

import (
	"math"
	"strings"
)

type cutCandidateState string

const (
	cutCandidateGenerated                       cutCandidateState = "generated"
	cutCandidateSelected                        cutCandidateState = "selected"
	cutCandidateApplied                         cutCandidateState = "applied"
	cutCandidateRejectedZeroNorm                cutCandidateState = "zero_norm"
	cutCandidateRejectedParallel                cutCandidateState = "parallel"
	cutCandidateRejectedCap                     cutCandidateState = "selection_cap"
	cutCandidateRejectedValidation              cutCandidateState = "validation"
	cutCandidateRejectedPoolDuplicate           cutCandidateState = "pool_duplicate"
	cutCandidateRejectedRowprepInvalid          cutCandidateState = "rowprep_invalid"
	cutCandidateRejectedRowprepEmpty            cutCandidateState = "rowprep_empty"
	cutCandidateRejectedRowprepCoefficientRange cutCandidateState = "rowprep_coefficient_range"
	cutCandidateRejectedRowprepInefficacious    cutCandidateState = "rowprep_inefficacious"
)

type cutCandidate struct {
	row                 Constraint
	family              string
	creationRound       int
	norm                float64
	efficacy            float64
	density             float64
	maxCoefficient      float64
	coefficientRatio    float64
	rowprepModified     bool
	rowprepRemoved      int
	rowprepRounded      int
	rowprepLoss         float64
	rowprepScale        float64
	preNonzeros         int
	preEfficacy         float64
	preMaxCoefficient   float64
	preCoefficientRatio float64
	poolIndex           int
	source              string
	state               cutCandidateState
}

func cutFamilyName(c Constraint) string {
	switch {
	case strings.HasPrefix(c.Name, "gomory"):
		return "gomory"
	case strings.HasPrefix(c.Name, "cmir"):
		return "cmir"
	case strings.HasPrefix(c.Name, "mir"):
		return "mir"
	default:
		return "other"
	}
}

func makeCutCandidates(cuts []Constraint, values []float64, nvars, round int) []cutCandidate {
	candidates := make([]cutCandidate, len(cuts))
	for i, cut := range cuts {
		candidates[i] = cutCandidate{
			row:           cut,
			family:        cutFamilyName(cut),
			creationRound: round,
			rowprepScale:  1,
			poolIndex:     -1,
			source:        "generator",
			state:         cutCandidateGenerated,
		}
		refreshCutCandidateMetrics(&candidates[i], values, nvars)
		candidates[i].preNonzeros = len(cut.Coeffs)
		candidates[i].preEfficacy = candidates[i].efficacy
		candidates[i].preMaxCoefficient = candidates[i].maxCoefficient
		candidates[i].preCoefficientRatio = candidates[i].coefficientRatio
	}
	return candidates
}

func makePreparedCutCandidates(m *Model, cuts []Constraint, values []float64, opts Options, round int) []cutCandidate {
	candidates := makeCutCandidates(cuts, values, len(m.vars), round)
	cfg := rootCutRowprepConfig(opts)
	for i := range candidates {
		prepared := cleanupCutRow(m, candidates[i].row, values, cfg)
		candidates[i].row = prepared.row
		candidates[i].rowprepModified = prepared.modified
		candidates[i].rowprepRemoved = prepared.removedCoefficients
		candidates[i].rowprepRounded = prepared.roundedCoefficients
		candidates[i].rowprepLoss = prepared.referenceLoss
		candidates[i].rowprepScale = prepared.scale
		if !prepared.success {
			candidates[i].state = prepared.reason
		}
		refreshCutCandidateMetrics(&candidates[i], values, len(m.vars))
	}
	return candidates
}

func refreshCutCandidateMetrics(candidate *cutCandidate, values []float64, nvars int) {
	if candidate == nil {
		return
	}
	candidate.norm = coeffNorm(candidate.row.Coeffs)
	minNZ := math.Inf(1)
	candidate.maxCoefficient = 0
	for _, coef := range candidate.row.Coeffs {
		absCoef := math.Abs(coef)
		if absCoef == 0 {
			continue
		}
		minNZ = math.Min(minNZ, absCoef)
		candidate.maxCoefficient = math.Max(candidate.maxCoefficient, absCoef)
	}
	candidate.coefficientRatio = 0
	if minNZ < math.Inf(1) {
		candidate.coefficientRatio = candidate.maxCoefficient / minNZ
	}
	candidate.density = 0
	if nvars > 0 {
		candidate.density = float64(len(candidate.row.Coeffs)) / float64(nvars)
	}
	candidate.efficacy = 0
	if candidate.norm >= 1e-12 {
		candidate.efficacy = cutEfficacy(candidate.row, values, candidate.norm)
	}
}

func selectCutCandidates(candidates []cutCandidate, cap int) []int {
	if len(candidates) <= cap {
		selected := make([]int, 0, len(candidates))
		for i := range candidates {
			if candidates[i].state != cutCandidateGenerated {
				continue
			}
			candidates[i].state = cutCandidateSelected
			selected = append(selected, i)
		}
		return selected
	}
	if cap <= 0 {
		for i := range candidates {
			if candidates[i].state == cutCandidateGenerated {
				candidates[i].state = cutCandidateRejectedCap
			}
		}
		return nil
	}

	used := make([]bool, len(candidates))
	for i := range candidates {
		if candidates[i].state != cutCandidateGenerated {
			used[i] = true
			continue
		}
		if candidates[i].norm < 1e-12 {
			used[i] = true
			candidates[i].state = cutCandidateRejectedZeroNorm
		}
	}

	const minOrtho = 0.90
	selected := make([]int, 0, cap)
	for len(selected) < cap {
		best := -1
		for i := range candidates {
			if used[i] {
				continue
			}
			if best < 0 || candidates[i].efficacy > candidates[best].efficacy {
				best = i
			}
		}
		if best < 0 {
			break
		}
		used[best] = true
		candidates[best].state = cutCandidateSelected
		selected = append(selected, best)
		for i := range candidates {
			if used[i] {
				continue
			}
			if math.Abs(cosineSim(candidates[best].row.Coeffs, candidates[best].norm, candidates[i].row.Coeffs, candidates[i].norm)) >= minOrtho {
				used[i] = true
				candidates[i].state = cutCandidateRejectedParallel
			}
		}
	}
	for i := range candidates {
		if candidates[i].state == cutCandidateGenerated {
			candidates[i].state = cutCandidateRejectedCap
		}
	}
	return selected
}

func selectedCandidateCuts(candidates []cutCandidate, selected []int) []Constraint {
	cuts := make([]Constraint, 0, len(selected))
	for _, idx := range selected {
		cuts = append(cuts, candidates[idx].row)
	}
	return cuts
}

// selectCuts replaces naive "keep however many the generator happened to
// emit first" truncation with a greedy efficacy + orthogonality selection,
// the mechanism SCIP's cutsel_hybrid.c uses to keep a large pool of
// candidate cuts down to a small, mutually-useful set before they ever touch
// the LP. Candidates are scored by efficacy (Euclidean distance from the
// current LP point to the cut's hyperplane — scale-invariant, unlike raw
// violation), picked greedily highest-efficacy first, and any remaining
// candidate whose cosine similarity to an already-picked cut exceeds
// minOrtho is dropped (near-duplicate cuts stacking the same direction add
// LP rows without adding pruning power). Returns at most cap cuts.
func selectCuts(cuts []Constraint, values []float64, cap int) []Constraint {
	candidates := makeCutCandidates(cuts, values, len(values), 0)
	return selectedCandidateCuts(candidates, selectCutCandidates(candidates, cap))
}

func coeffNorm(coeffs map[Var]float64) float64 {
	sum := 0.0
	for _, e := range sortedCoeffs(coeffs) {
		sum += e.coef * e.coef
	}
	return math.Sqrt(sum)
}

// cutEfficacy is the signed distance from values to the cut's hyperplane,
// normalized by the coefficient L2 norm — the standard efficacy measure
// (higher is a more-violated, more-useful cut). Constraints reaching this
// function are already known to be violated by their generator (both
// gomoryCuts and revisedGomoryCuts check `lhs < rhs` before emitting), so
// this is always the magnitude of that violation, but it is computed
// directly rather than assumed so a cut list from any future generator
// stays correct even without that precondition.
func cutEfficacy(c Constraint, values []float64, norm float64) float64 {
	lhs := 0.0
	for _, e := range sortedCoeffs(c.Coeffs) {
		if e.v < len(values) {
			lhs += e.coef * values[e.v]
		}
	}
	var violation float64
	switch c.Sense {
	case LessOrEqual:
		violation = lhs - c.RHS
	case GreaterOrEqual:
		violation = c.RHS - lhs
	case Equal:
		violation = math.Abs(lhs - c.RHS)
	}
	if violation < 0 {
		violation = 0
	}
	return violation / norm
}

// cosineSim returns the cosine similarity between two cuts' coefficient
// vectors, treated as sparse vectors over the shared Var index space.
func cosineSim(a map[Var]float64, normA float64, b map[Var]float64, normB float64) float64 {
	if normA < 1e-12 || normB < 1e-12 {
		return 0
	}
	// Iterate the smaller map (in sorted order, for a deterministic sum)
	// for the dot product.
	dot := 0.0
	if len(a) <= len(b) {
		for _, e := range sortedCoeffs(a) {
			if bv, ok := b[Var(e.v)]; ok {
				dot += e.coef * bv
			}
		}
	} else {
		for _, e := range sortedCoeffs(b) {
			if av, ok := a[Var(e.v)]; ok {
				dot += e.coef * av
			}
		}
	}
	return dot / (normA * normB)
}
