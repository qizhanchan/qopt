package qopt

import (
	"context"
	"math"
	"sort"
)

// runRENS solves a small relaxation-induced neighborhood: discrete variables
// closest to integral in the root LP are fixed, while the remaining variables
// are searched by a node-capped plain B&B sub-solve. It is a bounded root
// neighborhood fallback after repair, pump, and diving.
func runRENS(ctx context.Context, m *Model, lpValues []float64, opts Options) ([]float64, bool, int) {
	if len(lpValues) != len(m.vars) {
		return nil, false, 0
	}
	type candidate struct {
		idx  int
		dist float64
	}
	var cands []candidate
	for i, spec := range m.vars {
		if spec.Type == Integer || spec.Type == Binary {
			cands = append(cands, candidate{i, math.Abs(lpValues[i] - math.Round(lpValues[i]))})
		}
	}
	if len(cands) < 2 {
		return nil, false, 0
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].dist != cands[j].dist {
			return cands[i].dist < cands[j].dist
		}
		return cands[i].idx < cands[j].idx
	})
	// True RENS fixes only variables already integral in the LP relaxation.
	// Rounding an arbitrary nearest 20% can make the neighborhood infeasible
	// before its first LP solve, which defeats the point of a feasibility-
	// preserving root neighborhood.
	fixCount := 0
	for fixCount < len(cands) && cands[fixCount].dist <= opts.IntegralityTol {
		fixCount++
	}
	maxFix := int(math.Floor(0.80 * float64(len(cands))))
	if fixCount > maxFix {
		fixCount = maxFix
	}
	if fixCount <= 0 || fixCount >= len(cands) {
		return nil, false, 0
	}

	sub := cloneModel(m)
	for _, c := range cands[:fixCount] {
		v := math.Round(lpValues[c.idx])
		v = math.Max(math.Ceil(*m.vars[c.idx].Lower), v)
		v = math.Min(math.Floor(*m.vars[c.idx].Upper), v)
		sub.vars[c.idx].Lower = Float64(v)
		sub.vars[c.idx].Upper = Float64(v)
	}
	subOpts := opts
	subOpts.TimeLimit = 0 // ctx carries the scheduler's per-call deadline
	subOpts.MaxNodes = 20
	subOpts.MaxCutRounds = 0
	subOpts.MaxCutDepth = 0
	subOpts.UseMIRCuts = false
	subOpts.DetectSymmetry = false
	subOpts.PseudoObjPropagation = false
	subOpts.ReducedCostFixing = false
	subOpts.UseRINS = false
	subOpts.DisablePrimalHeuristics = true
	subOpts.MIPBranching = MIPBranchMostFractional
	res, err := sub.Solve(ctx, subOpts)
	lpSolves := 0
	if res.Stats != nil {
		lpSolves = res.Stats.Revised.LPSolves
	}
	if err != nil || res.Values == nil {
		return nil, false, lpSolves
	}
	if _, feasible := feasibleIncumbent(m, res.Values, opts); !feasible {
		return nil, false, lpSolves
	}
	return res.Values, true, lpSolves
}
