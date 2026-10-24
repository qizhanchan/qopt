package qopt

import (
	"context"
	"math"
)

const divingMaxLPSolves = 20

// runPseudocostDiving follows a single warm-started path from an LP solution.
// At each step it chooses a fractional variable by pseudocost and dives in the
// direction with lower predicted objective degradation; an infeasible first
// choice gets one opposite-direction retry. It never changes the main tree.
func runPseudocostDiving(
	ctx context.Context,
	original, nodeModel *Model,
	nodeExtras []Constraint,
	baseBounds []boundConstraint,
	values []float64,
	warm *lpWarmStart,
	opts Options,
	pcosts *pseudocostState,
	cache *canonCache,
) ([]float64, bool, int) {
	if len(values) != len(original.vars) {
		return nil, false, 0
	}
	current := append([]float64(nil), values...)
	bounds := append([]boundConstraint(nil), baseBounds...)
	lpSolves := 0

	for lpSolves < divingMaxLPSolves && ctx.Err() == nil {
		// Repair only after at least one actual dive step; the caller already
		// ran the root repair, so doing it again here would report a zero-LP
		// "diving" success and never exercise the warm-started path.
		if lpSolves > 0 {
			if _, repaired, ok := repairRoundedIncumbent(original, current, opts); ok {
				return repaired, true, lpSolves
			}
		}
		j, ok := pickBranchVar(original.vars, current, Options{
			IntegralityTol: opts.IntegralityTol,
			MIPBranching:   MIPBranchPseudocost,
		}, pcosts)
		if !ok {
			if _, feasible := feasibleIncumbent(original, current, opts); feasible {
				return append([]float64(nil), current...), true, lpSolves
			}
			return nil, false, lpSolves
		}

		v := current[j]
		floor, ceil := math.Floor(v), math.Ceil(v)
		downFirst := divingPreferDown(original, pcosts, j, v-floor)
		choices := []bool{downFirst, !downFirst}
		advanced := false
		for _, down := range choices {
			if lpSolves >= divingMaxLPSolves || ctx.Err() != nil {
				break
			}
			bc := boundConstraint{varIndex: j, sense: GreaterOrEqual, rhs: ceil}
			if down {
				bc.sense, bc.rhs = LessOrEqual, floor
			}
			trialBounds := appendBounds(bounds, bc)
			res, resp, err := solveNodeOrFull(ctx, cache, nodeModel, nodeExtras, trialBounds, opts, warm)
			lpSolves++
			if err != nil || res.Status != Optimal {
				continue
			}
			bounds = trialBounds
			current = res.Values
			warm = resp.warm
			advanced = true
			break
		}
		if !advanced {
			return nil, false, lpSolves
		}
	}
	return nil, false, lpSolves
}

func divingPreferDown(m *Model, p *pseudocostState, j int, frac float64) bool {
	if p != nil {
		down := frac * p.downGain(j)
		up := (1 - frac) * p.upGain(j)
		if down > 0 || up > 0 {
			return down <= up
		}
	}
	coef := m.objective[Var(j)]
	if m.direction == Maximize {
		return coef <= 0
	}
	return coef >= 0
}
