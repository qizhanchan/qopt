package qopt

import (
	"context"
	"math"
	"time"
)

const (
	localBranchingDefaultRadius = 18
	localBranchingPeriodNodes   = 200
	localBranchingWaitNodes     = 200
	localBranchingMinNodesSeen  = 4000
	localBranchingNodesOffset   = 1000
	localBranchingNodesQuot     = 0.05
	localBranchingMinNodes      = 1000
	localBranchingMaxNodes      = 10000
	localBranchingMaxCallTime   = time.Second
)

type localBranchingState struct {
	haveReference bool
	referenceNode int
}

func (s *localBranchingState) eligible(referenceNode int) bool {
	return !s.haveReference || s.referenceNode != referenceNode
}

func (s *localBranchingState) record(referenceNode int) {
	s.haveReference = true
	s.referenceNode = referenceNode
}

func localBranchingRadius(opts Options) int {
	if opts.LocalBranchingRadius > 0 {
		return opts.LocalBranchingRadius
	}
	return localBranchingDefaultRadius
}

func countBinaryVars(m *Model) int {
	n := 0
	for _, spec := range m.vars {
		if spec.Type == Binary {
			n++
		}
	}
	return n
}

func shouldTryLocalBranching(opts Options, m *Model, haveIncumbent bool, nodes, lastImprovedNode int) bool {
	if !opts.UseLocalBranching || !haveIncumbent || nodes < localBranchingMinNodesSeen ||
		nodes%localBranchingPeriodNodes != 0 || nodes-lastImprovedNode < localBranchingWaitNodes {
		return false
	}
	radius := localBranchingRadius(opts)
	return radius > 0 && countBinaryVars(m) >= 2*radius
}

func localBranchingModel(m *Model, incumbent []float64, radius int, tol float64) (*Model, bool) {
	if len(incumbent) != len(m.vars) || radius <= 0 {
		return nil, false
	}
	coeffs := make(map[Var]float64)
	ones := 0
	binaries := 0
	for i, spec := range m.vars {
		if spec.Type != Binary {
			continue
		}
		binaries++
		value := incumbent[i]
		if math.Abs(value-math.Round(value)) > tol {
			return nil, false
		}
		if math.Round(value) == 1 {
			coeffs[Var(i)] = -1
			ones++
		} else {
			coeffs[Var(i)] = 1
		}
	}
	if binaries < 2*radius {
		return nil, false
	}
	sub := cloneModel(m)
	if err := sub.AddNamedConstraint("local_branching_distance", coeffs, LessOrEqual, float64(radius-ones)); err != nil {
		return nil, false
	}
	return sub, true
}

func addStrictObjectiveCutoff(sub *Model, direction Direction, incumbent, tol float64) bool {
	if len(sub.objective) == 0 {
		return false
	}
	coeffs := make(map[Var]float64, len(sub.objective))
	for v, coef := range sub.objective {
		coeffs[v] = coef
	}
	sense := LessOrEqual
	rhs := incumbent - tol
	if direction == Maximize {
		sense = GreaterOrEqual
		rhs = incumbent + tol
	}
	return sub.AddNamedConstraint("local_branching_objective_cutoff", coeffs, sense, rhs) == nil
}

func localBranchingNodeLimit(outerNodes int) int {
	return clampInt(localBranchingNodesOffset+int(localBranchingNodesQuot*float64(outerNodes)),
		localBranchingMinNodes, localBranchingMaxNodes)
}

func localBranchingSubOptions(opts Options, nodeLimit int) Options {
	sub := opts
	sub.TimeLimit = 0
	sub.MaxNodes = nodeLimit
	sub.MaxCutRounds = 0
	sub.MaxCutDepth = 0
	sub.UseMIRCuts = false
	sub.DetectSymmetry = false
	sub.PseudoObjPropagation = false
	sub.ReducedCostFixing = false
	sub.CutRowAgeLimit = 0
	sub.UseRINS = false
	sub.UseLocalBranching = false
	sub.DisablePrimalHeuristics = true
	sub.MIPGap = 0
	sub.MIPNodeSelection = MIPBestEstimate
	sub.BestEstimateBestBoundFreq = 10
	sub.MIPBranching = MIPBranchMostFractional
	return sub
}

type localBranchingRunResult struct {
	objective float64
	values    []float64
	improved  bool
	nodes     int
	lpSolves  int
	radius    int
	nodeLimit int
}

func runLocalBranching(ctx context.Context, m *Model, direction Direction, incumbent []float64, incumbentObj float64, outerNodes int, opts Options) localBranchingRunResult {
	radius := localBranchingRadius(opts)
	nodeLimit := localBranchingNodeLimit(outerNodes)
	run := localBranchingRunResult{radius: radius, nodeLimit: nodeLimit}
	sub, ok := localBranchingModel(m, incumbent, radius, opts.IntegralityTol)
	if !ok || !addStrictObjectiveCutoff(sub, direction, incumbentObj, opts.FeasibilityTol) {
		return run
	}
	res, err := sub.Solve(ctx, localBranchingSubOptions(opts, nodeLimit))
	run.nodes = res.Nodes
	if res.Stats != nil {
		run.lpSolves = res.Stats.Revised.LPSolves
	}
	if err != nil || res.Values == nil || !betterObjective(direction, res.Objective, incumbentObj, opts.FeasibilityTol) {
		return run
	}
	obj, feasible := feasibleIncumbent(m, res.Values, opts)
	if !feasible || !betterObjective(direction, obj, incumbentObj, opts.FeasibilityTol) {
		return run
	}
	run.objective = obj
	run.values = res.Values
	run.improved = true
	return run
}
