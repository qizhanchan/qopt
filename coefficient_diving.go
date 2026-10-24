package qopt

import (
	"context"
	"math"
)

const coefficientDivingMaxLPSolves = 80

const coefficientDivingTailFractionals = 24

type diveLocks struct {
	down int
	up   int
}

type diveDecision struct {
	variable int
	down     bool
	rhs      float64
}

// runCoefficientDiving follows SCIP/CBC's coefficient-diving idea: variables
// are scored by the number of row locks in the chosen rounding direction,
// with unroundable and binary candidates preferred. Several bound changes are
// accumulated before each LP re-solve so a shallow LP budget can traverse a
// much deeper integer path.
func runCoefficientDiving(
	ctx context.Context,
	original, nodeModel *Model,
	nodeExtras []Constraint,
	baseBounds []boundConstraint,
	values []float64,
	warm *lpWarmStart,
	opts Options,
	cache *canonCache,
) ([]float64, bool, int) {
	if len(values) != len(original.vars) {
		return nil, false, 0
	}
	locks := modelDiveLocks(original)
	current := append([]float64(nil), values...)
	bounds := append([]boundConstraint(nil), baseBounds...)
	lpSolves := 0
	var checkpointBounds []boundConstraint
	var checkpointValues []float64
	var checkpointWarm *lpWarmStart
	var checkpointDecision diveDecision
	haveCheckpoint := false
	tailEntryBatchUsed := false

	for lpSolves < coefficientDivingMaxLPSolves && ctx.Err() == nil {
		if obj, rounded, ok := roundedIncumbent(original, current, opts); ok {
			_ = obj
			return rounded, true, lpSolves
		}
		fractional := countFractionalDiscrete(original, current, opts.IntegralityTol)
		if fractional == 0 {
			if _, ok := feasibleIncumbent(original, current, opts); ok {
				return append([]float64(nil), current...), true, lpSolves
			}
			return nil, false, lpSolves
		}
		// Use one conservative decision exactly once when the dive first enters
		// the tail. This avoids committing a whole small batch to a numerically
		// feasible but integer-dead path, then restores batching for progress.
		batchSize := coefficientDiveBatchSize(fractional, !tailEntryBatchUsed)
		if fractional <= coefficientDivingTailFractionals {
			tailEntryBatchUsed = true
		}
		used := make(map[int]bool, batchSize)
		decisions := make([]diveDecision, 0, batchSize)
		for len(decisions) < batchSize {
			decision, ok := pickCoefficientDiveDecision(original, current, locks, used, opts.IntegralityTol)
			if !ok {
				break
			}
			used[decision.variable] = true
			decisions = append(decisions, decision)
		}
		if len(decisions) == 0 {
			return nil, false, lpSolves
		}

		accepted := false
		for n := len(decisions); n >= 1 && lpSolves < coefficientDivingMaxLPSolves && ctx.Err() == nil; n /= 2 {
			trial := append([]boundConstraint(nil), bounds...)
			for _, d := range decisions[:n] {
				sense := GreaterOrEqual
				if d.down {
					sense = LessOrEqual
				}
				trial = append(trial, boundConstraint{varIndex: d.variable, sense: sense, rhs: d.rhs})
			}
			res, resp, err := solveNodeOrFull(ctx, cache, nodeModel, nodeExtras, trial, opts, warm)
			lpSolves++
			if err == nil && res.Status == Optimal {
				checkpointBounds = append(checkpointBounds[:0], bounds...)
				checkpointValues = append(checkpointValues[:0], current...)
				checkpointWarm = warm
				checkpointDecision = decisions[n-1]
				haveCheckpoint = true
				bounds, current, warm = trial, res.Values, resp.warm
				debugf("coefdiving: accepted batch=%d remaining_fractional=%d lp_solves=%d", n,
					countFractionalDiscrete(original, current, opts.IntegralityTol), lpSolves)
				accepted = true
				break
			}
			if n == 1 {
				break
			}
		}
		if !accepted {
			// One-level backtrack: restore the state before the previous
			// successful batch and reverse that batch's last decision.
			if !haveCheckpoint || lpSolves >= coefficientDivingMaxLPSolves {
				return nil, false, lpSolves
			}
			d := checkpointDecision
			sense, rhs := LessOrEqual, math.Floor(checkpointValues[d.variable])
			if d.down {
				sense, rhs = GreaterOrEqual, math.Ceil(checkpointValues[d.variable])
			}
			trial := appendBounds(checkpointBounds, boundConstraint{varIndex: d.variable, sense: sense, rhs: rhs})
			res, resp, err := solveNodeOrFull(ctx, cache, nodeModel, nodeExtras, trial, opts, checkpointWarm)
			lpSolves++
			if err != nil || res.Status != Optimal {
				debugf("coefdiving: backtrack failed after %d LP solves", lpSolves)
				return nil, false, lpSolves
			}
			bounds, current, warm = trial, res.Values, resp.warm
			haveCheckpoint = false
			debugf("coefdiving: backtracked remaining_fractional=%d lp_solves=%d",
				countFractionalDiscrete(original, current, opts.IntegralityTol), lpSolves)
		}
	}
	return nil, false, lpSolves
}

func coefficientDiveBatchSize(fractional int, useTailEntryBatch bool) int {
	if useTailEntryBatch && fractional <= coefficientDivingTailFractionals {
		return 1
	}
	batchSize := int(math.Ceil(0.15 * float64(fractional)))
	if batchSize < 1 {
		return 1
	}
	return batchSize
}

func modelDiveLocks(m *Model) []diveLocks {
	locks := make([]diveLocks, len(m.vars))
	for _, c := range m.constraints {
		for variable, coef := range c.Coeffs {
			i := int(variable)
			switch c.Sense {
			case LessOrEqual:
				if coef > 0 {
					locks[i].up++
				} else {
					locks[i].down++
				}
			case GreaterOrEqual:
				if coef > 0 {
					locks[i].down++
				} else {
					locks[i].up++
				}
			case Equal:
				locks[i].down++
				locks[i].up++
			}
		}
	}
	return locks
}

func countFractionalDiscrete(m *Model, values []float64, tol float64) int {
	n := 0
	for i, spec := range m.vars {
		if (spec.Type == Integer || spec.Type == Binary) && math.Abs(values[i]-math.Round(values[i])) > tol {
			n++
		}
	}
	return n
}

func pickCoefficientDiveDecision(m *Model, values []float64, locks []diveLocks, used map[int]bool, tol float64) (diveDecision, bool) {
	best := diveDecision{}
	bestScore := math.Inf(-1)
	found := false
	for i, spec := range m.vars {
		if used[i] || (spec.Type != Integer && spec.Type != Binary) {
			continue
		}
		value := values[i]
		frac := value - math.Floor(value)
		if frac <= tol || 1-frac <= tol {
			continue
		}
		mayDown := locks[i].down == 0
		mayUp := locks[i].up == 0
		roundUp := false
		if mayDown || mayUp {
			if mayDown && mayUp {
				roundUp = frac > 0.5
			} else {
				// Coefficient diving deliberately selects the direction that is
				// not already guaranteed safe by the row locks.
				roundUp = mayDown
			}
		} else {
			roundUp = locks[i].down > locks[i].up || (locks[i].down == locks[i].up && frac > 0.5)
		}
		distance := frac
		score := float64(locks[i].down)
		if roundUp {
			distance = 1 - frac
			score = float64(locks[i].up)
		}
		if distance < 0.01 {
			score *= 0.01
		}
		if spec.Type != Binary {
			score *= 0.1
		}
		if mayDown || mayUp {
			score -= float64(len(m.constraints))
		}
		if !found || score > bestScore+1e-15 || (math.Abs(score-bestScore) <= 1e-15 && i < best.variable) {
			found, bestScore = true, score
			best = diveDecision{variable: i, down: !roundUp, rhs: math.Floor(value)}
			if roundUp {
				best.rhs = math.Ceil(value)
			}
		}
	}
	return best, found
}
