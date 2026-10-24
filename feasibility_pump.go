package qopt

import (
	"context"
	"math"
	"sort"
	"time"
)

// Feasibility pump (Fischetti / Glover / Lodi, 2005) for mixed/general-
// integer models.
//
// On MIPLIB instances where the LP relaxation never naturally rounds to an
// integer-feasible solution (glass4, ic97_potential, neos-3046615-murg, ...)
// pure branching can run the full time budget without ever finding an
// incumbent. The pump alternates between:
//
//   1. Rounding the current LP fractional solution to integers.
//   2. Re-solving the LP relaxation with the objective replaced by the
//      L1 distance from the LP variable to the rounded point.
//
// The new LP optimum is by construction "as close as possible" (under L1) to
// the integer point, then we round that, and so on. If a rounded point
// satisfies the original constraints, we have a feasible incumbent. If we
// keep producing the same rounded point, we perturb the most-fractional
// variables and continue.
//
// Binary distance is represented directly in the objective. General integer
// distance uses one continuous deviation variable d_j and the two rows
// d_j >= x_j-z_j, d_j >= z_j-x_j, so the pump now targets mixed/general-
// integer models such as timtab1 as well.

const (
	// fpMaxIters is the default LP-iteration cap for the pump. Each iter is
	// one LP solve; tens of solves are typically enough when feasibility is
	// reachable, and once we hit this cap further attempts are unlikely.
	fpMaxIters = 50
	// fpPerturbBase is the base number of binary-variable flips on cycle.
	// The actual count grows as sqrt(numBinary) so problems with hundreds
	// of binaries (glass4-class) get meaningful perturbations rather than
	// the 3%-flip that's not enough to escape a cycle.
	fpPerturbBase = 10
	// fpHistory is the size of the recent-rounded-points window we look
	// against to detect cycles.
	fpHistory = 5
	// fpStallEscalations is the maximum number of cycles seen before the
	// pump gives up. The rotating perturbation window and the 50-iteration
	// outer cap bound the work; allowing more than five repeats is important
	// on equality-heavy general-integer models where short cycles are common.
	fpStallEscalations = 15
)

// runFeasibilityPump tries to find a feasible integer incumbent for m
// starting from the root LP relaxation values. work is the model used for
// LP solves (may carry root cuts on top of m); m is the user model used
// for the original-constraint feasibility check.
//
// Returns the integer-feasible values vector if found, or (nil, false).
// Mutates and restores work.objective and work.direction.
func runFeasibilityPump(ctx context.Context, m, work *Model, rootValues []float64, opts Options, start time.Time) ([]float64, bool, int) {
	if len(rootValues) != len(m.vars) || len(work.vars) != len(m.vars) {
		return nil, false, 0
	}
	hasDiscrete := false
	for _, v := range m.vars {
		if v.Type == Binary || v.Type == Integer {
			hasDiscrete = true
			break
		}
	}
	if !hasDiscrete {
		return nil, false, 0
	}

	numDiscrete := countDiscrete(m)
	basePerturb := int(math.Sqrt(float64(numDiscrete))) + fpPerturbBase
	if basePerturb > numDiscrete {
		basePerturb = numDiscrete
	}
	current := append([]float64(nil), rootValues...)
	history := make([][]float64, 0, fpHistory)
	escalation := 0
	lpSolves := 0
	debugf("fp: starting (discrete vars=%d perturb=%d)", numDiscrete, basePerturb)

	for iter := 0; iter < fpMaxIters; iter++ {
		if err := ctx.Err(); err != nil {
			return nil, false, lpSolves
		}
		if opts.TimeLimit > 0 && time.Since(start) >= opts.TimeLimit {
			return nil, false, lpSolves
		}

		// 1. Round current LP solution to integers (binaries clamped to [0,1],
		// general integers clamped to [Lower, Upper]).
		rounded := append([]float64(nil), current...)
		for i, v := range m.vars {
			switch v.Type {
			case Binary:
				rounded[i] = math.Round(current[i])
				if rounded[i] < 0 {
					rounded[i] = 0
				} else if rounded[i] > 1 {
					rounded[i] = 1
				}
			case Integer:
				rounded[i] = math.Round(current[i])
				lb := math.Ceil(*v.Lower)
				ub := math.Floor(*v.Upper)
				if rounded[i] < lb {
					rounded[i] = lb
				} else if !math.IsInf(ub, 1) && rounded[i] > ub {
					rounded[i] = ub
				}
			}
		}

		// 2. Feasibility check against the user's original model.
		if obj, ok := feasibleIncumbent(m, rounded, opts); ok {
			debugf("fp: iter %d found feasible incumbent obj=%g", iter, obj)
			return rounded, true, lpSolves
		}

		// 3. Cycle detection: if we've seen this rounded vector recently,
		// perturb. Perturbation flips the K binary variables whose LP
		// values are most-fractional (closest to 0.5), giving the next
		// pump LP a different starting point.
		if seenRoundedPoint(history, rounded) {
			escalation++
			if escalation > fpStallEscalations {
				debugf("fp: cycle persisted past %d escalations, giving up at iter %d",
					fpStallEscalations, iter)
				return nil, false, lpSolves
			}
			count := basePerturb * escalation
			if count > numDiscrete {
				count = numDiscrete
			}
			debugf("fp: iter %d cycle #%d, perturbing %d vars", iter, escalation, count)
			perturbDiscrete(current, rounded, m, count, escalation)
		}
		history = appendHistory(history, rounded, fpHistory)

		// 4. Build and solve the L1-distance LP relaxation. A fresh clone keeps
		// the caller's model/cache immutable and gives general integers explicit
		// deviation columns.
		pump := buildPumpLP(work, m, rounded)
		lpRes, _, _, err := solveLPModelWithBackend(ctx, pump, nil, opts, nil)
		lpSolves++
		if err != nil || lpRes.Status != Optimal {
			debugf("fp: iter %d LP solve failed (status=%s err=%v)", iter, lpRes.Status, err)
			return nil, false, lpSolves
		}
		current = append(current[:0], lpRes.Values[:len(m.vars)]...)
	}

	debugf("fp: budget exhausted (%d iters) without finding incumbent", fpMaxIters)
	return nil, false, lpSolves
}

func countDiscrete(m *Model) int {
	c := 0
	for _, v := range m.vars {
		if v.Type == Binary || v.Type == Integer {
			c++
		}
	}
	return c
}

func buildPumpLP(work, original *Model, rounded []float64) *Model {
	pump := cloneModel(work)
	pump.direction = Minimize
	obj := make(map[Var]float64)
	for j, spec := range original.vars {
		switch spec.Type {
		case Binary:
			if rounded[j] < 0.5 {
				obj[Var(j)] = 1
			} else {
				obj[Var(j)] = -1 // constant +1 is irrelevant
			}
		case Integer:
			d := pump.AddVar(VarSpec{Name: "fp_dev"})
			obj[d] = 1
			_ = pump.AddNamedConstraint("fp_dev_pos", map[Var]float64{Var(j): 1, d: -1}, LessOrEqual, rounded[j])
			_ = pump.AddNamedConstraint("fp_dev_neg", map[Var]float64{Var(j): -1, d: -1}, LessOrEqual, -rounded[j])
		}
	}
	_ = pump.SetObjective(obj)
	return pump
}

func seenRoundedPoint(history [][]float64, x []float64) bool {
	const eps = 1e-9
	for _, prev := range history {
		if len(prev) != len(x) {
			continue
		}
		same := true
		for i, v := range prev {
			if math.Abs(v-x[i]) > eps {
				same = false
				break
			}
		}
		if same {
			return true
		}
	}
	return false
}

func appendHistory(history [][]float64, x []float64, cap int) [][]float64 {
	history = append(history, append([]float64(nil), x...))
	if len(history) > cap {
		history = history[len(history)-cap:]
	}
	return history
}

// perturbDiscrete changes the targets of the most fractional discrete
// variables. Binaries flip; general integers move to the adjacent integer on
// the opposite side of the current LP point.
func perturbDiscrete(currentLP, rounded []float64, m *Model, count, seed int) {
	type fracIdx struct {
		idx  int
		dist float64 // 0.5 - |x - 0.5|; larger = more fractional
	}
	cands := make([]fracIdx, 0, count*2)
	for i, v := range m.vars {
		if v.Type != Binary && v.Type != Integer {
			continue
		}
		d := math.Abs(currentLP[i] - math.Round(currentLP[i]))
		if d > 0 {
			cands = append(cands, fracIdx{i, d})
		}
	}
	sort.Slice(cands, func(i, j int) bool {
		return cands[i].dist > cands[j].dist
	})
	flips := count
	if flips > len(cands) {
		flips = len(cands)
	}
	start := 0
	if len(cands) > 0 {
		start = (seed * 17) % len(cands)
	}
	for k := 0; k < flips; k++ {
		i := cands[(start+k)%len(cands)].idx
		if m.vars[i].Type == Binary {
			if rounded[i] < 0.5 {
				rounded[i] = 1
			} else {
				rounded[i] = 0
			}
			continue
		}
		lb := math.Ceil(*m.vars[i].Lower)
		ub := math.Floor(*m.vars[i].Upper)
		if rounded[i] <= currentLP[i] {
			rounded[i] = math.Min(ub, rounded[i]+1)
		} else {
			rounded[i] = math.Max(lb, rounded[i]-1)
		}
	}
}
