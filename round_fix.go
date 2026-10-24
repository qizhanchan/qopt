package qopt

import (
	"context"
	"math"
)

const roundFixMaxLPSolves = 8

// runRoundAndFix rounds every discrete variable using several deterministic
// seeds, fixes that assignment, and lets one LP recover all continuous
// variables simultaneously. This is especially useful for equality-heavy
// models where coordinate repair repeatedly breaks a neighboring row.
func runRoundAndFix(
	ctx context.Context,
	m *Model,
	lpValues []float64,
	opts Options,
	cache *canonCache,
) ([]float64, bool, int) {
	if len(lpValues) != len(m.vars) {
		return nil, false, 0
	}
	lower, upper := effectiveBounds(m, nil)
	lpSolves := 0
	for seed := 0; seed < roundFixMaxLPSolves && ctx.Err() == nil; seed++ {
		bounds := make([]boundConstraint, 0, 2*len(m.vars))
		for i, spec := range m.vars {
			if spec.Type != Integer && spec.Type != Binary {
				continue
			}
			v := roundedSeedValue(m, lpValues[i], i, seed)
			v = math.Max(lower[i], math.Min(upper[i], v))
			bounds = append(bounds,
				boundConstraint{varIndex: i, sense: GreaterOrEqual, rhs: v},
				boundConstraint{varIndex: i, sense: LessOrEqual, rhs: v},
			)
		}
		res, _, err := solveNodeOrFull(ctx, cache, m, nil, bounds, opts, nil)
		lpSolves++
		if err != nil || res.Status != Optimal {
			continue
		}
		if _, feasible := feasibleIncumbent(m, res.Values, opts); feasible {
			return append([]float64(nil), res.Values...), true, lpSolves
		}
	}
	return nil, false, lpSolves
}

func roundedSeedValue(m *Model, value float64, variable, seed int) float64 {
	floor := math.Floor(value)
	switch seed {
	case 0: // nearest
		return math.Round(value)
	case 1: // objective-aligned
		coef := m.objective[Var(variable)]
		if (m.direction == Maximize && coef > 0) || (m.direction == Minimize && coef < 0) {
			return floor + 1
		}
		return floor
	case 2:
		return floor
	case 3:
		return floor + 1
	default:
		if deterministicUnit(seed-4, variable) < value-floor {
			return floor + 1
		}
		return floor
	}
}
