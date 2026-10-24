package qopt

import "math"

// repairRoundedIncumbent is a deterministic shift-and-propagate heuristic.
// It first propagates the model domain, rounds discrete variables, then repairs
// the most violated row by shifting the variable that buys the most violation
// reduction per unit of objective damage and collateral row violation.
func repairRoundedIncumbent(m *Model, lpValues []float64, opts Options) (float64, []float64, bool) {
	if len(lpValues) != len(m.vars) {
		return 0, nil, false
	}
	lower, upper := effectiveBounds(m, nil)
	domain := newDomainState(m, lower, upper, opts.FeasibilityTol)
	if !domain.propagate(5) {
		return 0, nil, false
	}

	type seedMode int
	const (
		seedNearest seedMode = iota
		seedObjective
		seedDown
		seedUp
		seedRandom
	)
	var bestObj float64
	var best []float64
	found := false
	modes := []seedMode{seedNearest, seedObjective, seedDown, seedUp}
	for i := 0; i < 8; i++ {
		modes = append(modes, seedRandom)
	}
	for seed, mode := range modes {
		cand := append([]float64(nil), lpValues...)
		for i, spec := range m.vars {
			if cand[i] < lower[i] {
				cand[i] = lower[i]
			}
			if cand[i] > upper[i] {
				cand[i] = upper[i]
			}
			if spec.Type != Integer && spec.Type != Binary {
				continue
			}
			switch mode {
			case seedNearest:
				cand[i] = math.Round(cand[i])
			case seedObjective:
				coef := m.objective[Var(i)]
				if (m.direction == Maximize && coef > 0) || (m.direction == Minimize && coef < 0) {
					cand[i] = math.Ceil(cand[i])
				} else {
					cand[i] = math.Floor(cand[i])
				}
			case seedDown:
				cand[i] = math.Floor(cand[i])
			case seedUp:
				cand[i] = math.Ceil(cand[i])
			case seedRandom:
				floor := math.Floor(cand[i])
				frac := cand[i] - floor
				if deterministicUnit(seed, i) < frac {
					cand[i] = floor + 1
				} else {
					cand[i] = floor
				}
			}
			cand[i] = math.Max(lower[i], math.Min(upper[i], cand[i]))
		}
		if !repairPoint(m, cand, lower, upper, opts) {
			continue
		}
		obj, ok := feasibleIncumbent(m, cand, opts)
		if !ok {
			continue
		}
		if !found || betterObjective(m.direction, obj, bestObj, opts.FeasibilityTol) {
			found, bestObj, best = true, obj, append([]float64(nil), cand...)
		}
	}
	return bestObj, best, found
}

func deterministicUnit(seed, variable int) float64 {
	// SplitMix64 gives deterministic, well-dispersed rounding choices without
	// introducing run-to-run randomness into the solver.
	x := uint64(seed+1)*0x9e3779b97f4a7c15 ^ uint64(variable+1)*0xbf58476d1ce4e5b9
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return float64(x>>11) * (1.0 / (1 << 53))
}

func repairPoint(m *Model, x, lower, upper []float64, opts Options) bool {
	rows := m.sortedRows()
	data := m.propagationData()
	activity := make([]float64, len(rows))
	rowNorm := make([]float64, len(rows))
	for r, row := range rows {
		for _, e := range row {
			activity[r] += e.coef * x[e.v]
			rowNorm[r] += math.Abs(e.coef)
		}
		if rowNorm[r] < 1 {
			rowNorm[r] = 1
		}
	}
	tol := opts.FeasibilityTol
	if tol <= 0 {
		tol = 1e-8
	}
	objScale := 1.0
	for i, v := range x {
		objScale += math.Abs(m.objective[Var(i)] * v)
	}
	maxIters := 10*len(m.vars) + 100
	if maxIters > 5000 {
		maxIters = 5000
	}

	for iter := 0; iter < maxIters; iter++ {
		row := -1
		want := 0.0
		worst := tol
		violation := 0.0
		for r, c := range m.constraints {
			var v, dir float64
			switch c.Sense {
			case LessOrEqual:
				v, dir = activity[r]-c.RHS, -1
			case GreaterOrEqual:
				v, dir = c.RHS-activity[r], 1
			case Equal:
				if activity[r] > c.RHS {
					v, dir = activity[r]-c.RHS, -1
				} else {
					v, dir = c.RHS-activity[r], 1
				}
			}
			scaled := v / rowNorm[r]
			if v > tol && (row < 0 || scaled > worst+tol) {
				row, want, worst, violation = r, dir, scaled, v
			}
		}
		if row < 0 {
			return true
		}

		bestVar := -1
		bestDelta := 0.0
		bestScore := -1.0
		for _, e := range rows[row] {
			v := e.v
			direction := want
			if e.coef < 0 {
				direction = -direction
			}
			available := upper[v] - x[v]
			if direction < 0 {
				available = x[v] - lower[v]
			}
			if available <= tol || math.IsNaN(available) {
				continue
			}
			move := violation / math.Abs(e.coef)
			if m.vars[v].Type == Integer || m.vars[v].Type == Binary {
				move = math.Ceil(move - tol)
				if move < 1 {
					move = 1
				}
			}
			if move > available {
				move = available
			}
			delta := direction * move
			gain := math.Min(violation, math.Abs(e.coef*delta))
			if gain <= tol {
				continue
			}

			harm := 0.0
			for _, rr := range data.varRows[v] {
				if rr == row {
					continue
				}
				before := constraintViolation(m.constraints[rr], activity[rr])
				after := constraintViolation(m.constraints[rr], activity[rr]+m.constraints[rr].Coeffs[Var(v)]*delta)
				if after > before {
					harm += (after - before) / rowNorm[rr]
				}
			}
			objDelta := m.objective[Var(v)] * delta
			if m.direction == Maximize {
				objDelta = -objDelta
			}
			if objDelta < 0 {
				objDelta = 0
			}
			score := gain / (1 + harm + objDelta/objScale)
			if score > bestScore+1e-15 || (math.Abs(score-bestScore) <= 1e-15 && (bestVar < 0 || v < bestVar)) {
				bestVar, bestDelta, bestScore = v, delta, score
			}
		}
		if bestVar < 0 {
			return false
		}
		x[bestVar] += bestDelta
		if m.vars[bestVar].Type == Integer || m.vars[bestVar].Type == Binary {
			x[bestVar] = math.Round(x[bestVar])
		}
		for _, r := range data.varRows[bestVar] {
			activity[r] += m.constraints[r].Coeffs[Var(bestVar)] * bestDelta
		}
	}
	return false
}

func constraintViolation(c Constraint, lhs float64) float64 {
	switch c.Sense {
	case LessOrEqual:
		return math.Max(0, lhs-c.RHS)
	case GreaterOrEqual:
		return math.Max(0, c.RHS-lhs)
	case Equal:
		return math.Abs(lhs - c.RHS)
	}
	return 0
}
