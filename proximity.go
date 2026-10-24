package qopt

import (
	"context"
	"math"
	"time"
)

// Proximity search keeps the original objective as an improvement constraint
// and minimizes binary Hamming distance to the incumbent. This is the
// Fischetti-Monaci neighborhood used by SCIP's heur_proximity.c.
const (
	proximityPeriodNodes       = 200
	proximityWaitingNodes      = 100
	proximityMinNodesSeen      = 4000
	proximityDefaultMinImprove = 0.02
	proximityBinaryQuot        = 0.10
	proximityNodesOffset       = 50
	proximityNodesQuot         = 0.10
	proximityMinNodes          = 1
	proximityMaxNodes          = 10000
	proximityMaxCallTime       = time.Second
)

func proximityMinImprove(opts Options) float64 {
	if opts.ProximityMinImprove > 0 {
		return opts.ProximityMinImprove
	}
	return proximityDefaultMinImprove
}

type proximityState struct {
	haveReference bool
	referenceNode int
}

func (s *proximityState) eligible(referenceNode int) bool {
	return !s.haveReference || s.referenceNode != referenceNode
}

func (s *proximityState) record(referenceNode int) {
	s.haveReference = true
	s.referenceNode = referenceNode
}

func neighborhoodHeuristicCount(opts Options) int {
	n := 0
	if opts.UseRINS {
		n++
	}
	if opts.UseLocalBranching {
		n++
	}
	if opts.UseProximitySearch {
		n++
	}
	return n
}

func shouldTryProximity(opts Options, m *Model, haveIncumbent bool, nodes, lastImprovedNode int) bool {
	if !opts.UseProximitySearch || !haveIncumbent || len(m.objective) == 0 ||
		nodes < proximityMinNodesSeen || nodes%proximityPeriodNodes != 0 ||
		nodes-lastImprovedNode < proximityWaitingNodes || len(m.vars) == 0 {
		return false
	}
	return float64(countBinaryVars(m)) >= proximityBinaryQuot*float64(len(m.vars))
}

func proximityModel(m *Model, incumbent []float64, incumbentObj, lpBound float64, opts Options) (*Model, bool) {
	if len(incumbent) != len(m.vars) || len(m.objective) == 0 ||
		!betterObjective(m.direction, lpBound, incumbentObj, opts.FeasibilityTol) {
		return nil, false
	}

	sub := cloneModel(m)
	minImprove := proximityMinImprove(opts)
	cutoff := (1-minImprove)*incumbentObj + minImprove*lpBound
	sense := LessOrEqual
	if m.direction == Minimize {
		cutoff = math.Min(cutoff, incumbentObj-opts.FeasibilityTol)
	} else {
		sense = GreaterOrEqual
		cutoff = math.Max(cutoff, incumbentObj+opts.FeasibilityTol)
	}
	coeffs := make(map[Var]float64, len(m.objective))
	for v, coef := range m.objective {
		coeffs[v] = coef
	}
	if err := sub.AddNamedConstraint("proximity_objective_cutoff", coeffs, sense, cutoff); err != nil {
		return nil, false
	}

	distance := make(map[Var]float64)
	binaries := 0
	for i, spec := range m.vars {
		if spec.Type != Binary {
			continue
		}
		binaries++
		value := incumbent[i]
		if math.Abs(value-math.Round(value)) > opts.IntegralityTol {
			return nil, false
		}
		if math.Round(value) == 1 {
			distance[Var(i)] = -1
		} else {
			distance[Var(i)] = 1
		}
	}
	if binaries == 0 || float64(binaries) < proximityBinaryQuot*float64(len(m.vars)) {
		return nil, false
	}
	sub.direction = Minimize
	if err := sub.SetObjective(distance); err != nil {
		return nil, false
	}
	return sub, true
}

func proximityNodeLimit(outerNodes int) int {
	return clampInt(proximityNodesOffset+int(proximityNodesQuot*float64(outerNodes)),
		proximityMinNodes, proximityMaxNodes)
}

func proximitySubOptions(opts Options, nodeLimit int) Options {
	sub := localBranchingSubOptions(opts, nodeLimit)
	sub.UseProximitySearch = false
	return sub
}

type proximityRunResult struct {
	objective float64
	values    []float64
	improved  bool
	nodes     int
	lpSolves  int
	nodeLimit int
}

func runProximity(ctx context.Context, m *Model, incumbent []float64, incumbentObj, lpBound float64, outerNodes int, opts Options) proximityRunResult {
	nodeLimit := proximityNodeLimit(outerNodes)
	run := proximityRunResult{nodeLimit: nodeLimit}
	sub, ok := proximityModel(m, incumbent, incumbentObj, lpBound, opts)
	if !ok {
		return run
	}
	res, err := sub.Solve(ctx, proximitySubOptions(opts, nodeLimit))
	run.nodes = res.Nodes
	if res.Stats != nil {
		run.lpSolves = res.Stats.Revised.LPSolves
	}
	if err != nil || res.Values == nil {
		return run
	}
	obj, feasible := feasibleIncumbent(m, res.Values, opts)
	if !feasible || !betterObjective(m.direction, obj, incumbentObj, opts.FeasibilityTol) {
		return run
	}
	run.objective = obj
	run.values = res.Values
	run.improved = true
	return run
}
