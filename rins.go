package qopt

import (
	"context"
	"math"
	"time"
)

// Relaxation Induced Neighborhood Search (RINS, Danna/Rothberg/Le Pape):
// once an incumbent exists, periodically fix every integer/binary variable
// whose current LP relaxation value already agrees with the incumbent's
// value, then re-solve the (usually much smaller) remaining sub-problem
// with its own small node/time budget. q-opt has a rounding heuristic and a
// feasibility pump to find a FIRST incumbent but nothing that searches a
// neighborhood of an EXISTING one — this is the gap documented against
// HiGHS/SCIP on mad / neos-3046615-murg / neos-3754480-nidda (20-40% worse
// incumbents, same ballpark of feasibility). See SCIP's heur_rins.c for the
// reference design this ports a simplified version of.
//
// The opt-in adaptive mode follows the useful parts of SCIP's heur_rins.c:
// a success/usage-adjusted node budget, a wait after incumbent improvements,
// an objective cutoff, and failure backoff. The neighborhood also inherits
// the current B&B node bounds. Each call is additionally capped at 500ms, and
// a first built-but-unsuccessful neighborhood exhausts adaptive RINS for that
// solve; q-opt's sub-MIP LP solves are still much more expensive than SCIP's.
// Only integer/binary variables get fixed; continuous variables remain free in
// the reduced model.

const (
	rinsPeriodNodes    = 200 // check cadence: only ever consider firing on node counts that are multiples of this
	rinsWaitNodes      = 200 // additionally require at least this many nodes since the last incumbent improvement
	rinsMinFixFraction = 0.3 // abort if fewer than this fraction of discrete vars agree with the incumbent
	rinsMinNodes       = 50
	rinsMaxSeconds     = 5.0 // absolute cap on the sub-solve's own time budget

	rinsLegacyNodesOffset = 100
	rinsLegacyNodesQuot   = 0.05
	rinsLegacyMaxNodes    = 2000

	rinsAdaptiveNodesOffset  = 500
	rinsAdaptiveNodesQuot    = 0.30
	rinsAdaptiveMaxNodes     = 5000
	rinsAdaptiveSetupCost    = 100
	rinsMinImprove           = 0.01
	rinsAdaptiveMaxCallTime  = 500 * time.Millisecond
	rinsAdaptiveMinNodesSeen = 4000
	rinsAdaptiveMinPrimalGap = 0.50
)

type rinsState struct {
	calls            int
	successes        int
	usedNodes        int
	consecutiveFails int
	nextEligibleNode int
	exhausted        bool
}

func (s *rinsState) nodeBudget(outerNodes int, adaptive bool) int {
	if !adaptive {
		return clampInt(rinsLegacyNodesOffset+int(rinsLegacyNodesQuot*float64(outerNodes)), rinsMinNodes, rinsLegacyMaxNodes)
	}
	nodes := int(rinsAdaptiveNodesQuot * float64(outerNodes))
	nodes = int(float64(nodes) * float64(s.successes+1) / float64(s.calls+1))
	nodes -= rinsAdaptiveSetupCost * s.calls
	nodes += rinsAdaptiveNodesOffset
	nodes -= s.usedNodes
	if nodes > rinsAdaptiveMaxNodes {
		nodes = rinsAdaptiveMaxNodes
	}
	if nodes < rinsMinNodes {
		return 0
	}
	return nodes
}

func (s *rinsState) eligible(node int) bool {
	return !s.exhausted && node >= s.nextEligibleNode
}

func (s *rinsState) record(node int, run rinsRunResult) {
	if !run.built {
		return
	}
	s.calls++
	s.usedNodes += run.nodes
	if run.improved {
		s.successes++
		s.consecutiveFails = 0
		s.nextEligibleNode = node + rinsWaitNodes
		return
	}
	if s.successes == 0 {
		// A failed first real neighborhood is a strong predictor of repeated
		// failure on the measured nidda class; do not keep displacing the main
		// search with equivalent sub-MIPs.
		s.exhausted = true
		return
	}
	s.consecutiveFails++
	shift := s.consecutiveFails
	if shift > 4 {
		shift = 4
	}
	s.nextEligibleNode = node + rinsWaitNodes*(1<<shift)
}

// shouldTryRINS reports whether the RINS heuristic should fire at this
// point in the outer search. Gating on lastImprovedNode (mirroring SCIP's
// nwaitingnodes) matters a lot in practice: on instances where DFS keeps
// finding strictly better incumbents on its own (markshare_4_0, pk1 — both
// hundreds of thousands to millions of nodes), a plain "every N nodes"
// cadence fires thousands of times over a run; even a small fraction of
// those triggering an actual sub-solve (up to rinsMaxSeconds each) burns
// enough of the time budget to cost those instances their proofs. Without
// this gate RINS should only ever fire once the search has been stuck.
func shouldTryRINS(opts Options, haveIncumbent bool, nodes, lastImprovedNode int) bool {
	return opts.UseRINS && haveIncumbent && nodes > 0 && nodes%rinsPeriodNodes == 0 && nodes-lastImprovedNode >= rinsWaitNodes
}

func adaptiveRINSCandidate(direction Direction, lpBound, incumbent float64, nodes int, tol float64) bool {
	if nodes < rinsAdaptiveMinNodesSeen || !betterObjective(direction, lpBound, incumbent, tol) {
		return false
	}
	return mipRelativeGap(lpBound, incumbent, true) >= rinsAdaptiveMinPrimalGap
}

// rinsFixedModel builds the model RINS re-solves: a clone of the original
// (uncut) model with every integer/binary variable whose LP relaxation
// value agrees with the incumbent's value fixed to that value. Returns
// ok=false if fewer than rinsMinFixFraction of the discrete variables agree
// (not enough of a reduction to be worth a sub-solve) or if every discrete
// variable would be fixed (nothing left to search).
func rinsFixedModel(m *Model, lpValues, incumbentValues []float64, tol float64) (*Model, bool) {
	return rinsFixedNodeModel(m, nil, lpValues, incumbentValues, tol)
}

func rinsFixedNodeModel(m *Model, nodeBounds []boundConstraint, lpValues, incumbentValues []float64, tol float64) (*Model, bool) {
	sub, _, ok := rinsFixedNodeModelWithInfo(m, nodeBounds, lpValues, incumbentValues, tol)
	return sub, ok
}

type rinsFixingInfo struct {
	fixed int
	total int
}

func rinsFixedNodeModelWithInfo(m *Model, nodeBounds []boundConstraint, lpValues, incumbentValues []float64, tol float64) (*Model, rinsFixingInfo, bool) {
	if len(lpValues) != len(m.vars) || len(incumbentValues) != len(m.vars) {
		return nil, rinsFixingInfo{}, false
	}
	discrete := 0
	agree := 0
	for i, spec := range m.vars {
		if spec.Type != Integer && spec.Type != Binary {
			continue
		}
		discrete++
		if math.Abs(lpValues[i]-incumbentValues[i]) <= tol {
			agree++
		}
	}
	info := rinsFixingInfo{fixed: agree, total: discrete}
	if discrete == 0 || agree == 0 || agree == discrete || float64(agree) < rinsMinFixFraction*float64(discrete) {
		return nil, info, false
	}

	sub := cloneModel(m)
	lower, upper := effectiveBounds(m, nodeBounds)
	for i := range sub.vars {
		sub.vars[i].Lower = Float64(lower[i])
		sub.vars[i].Upper = Float64(upper[i])
	}
	for i, spec := range m.vars {
		if spec.Type != Integer && spec.Type != Binary {
			continue
		}
		if math.Abs(lpValues[i]-incumbentValues[i]) <= tol {
			v := math.Round(incumbentValues[i])
			sub.vars[i].Lower = Float64(v)
			sub.vars[i].Upper = Float64(v)
		}
	}
	return sub, info, true
}

// rinsSubOptions derives the sub-solve's Options from the outer opts: same
// numerical tolerances/backend/branching rule, its own small node and time
// budget, and every other opt-in search-shaping mechanism (cuts, symmetry,
// the propagators above, and RINS itself) turned off so the sub-solve is a
// plain, fast, non-recursive B&B search.
func rinsSubOptions(opts Options, outerNodes int) Options {
	return rinsSubOptionsForNodes(opts, clampInt(
		rinsLegacyNodesOffset+int(rinsLegacyNodesQuot*float64(outerNodes)),
		rinsMinNodes, rinsLegacyMaxNodes,
	))
}

func rinsSubOptionsForNodes(opts Options, nodeLimit int) Options {
	sub := opts
	sub.MaxNodes = nodeLimit
	maxBudget := time.Duration(rinsMaxSeconds * float64(time.Second))
	budget := opts.TimeLimit
	if budget <= 0 || budget > maxBudget {
		budget = maxBudget
	}
	sub.TimeLimit = budget
	sub.MaxCutRounds = 0
	sub.MaxCutDepth = 0
	sub.UseMIRCuts = false
	sub.DetectSymmetry = false
	sub.PseudoObjPropagation = false
	sub.ReducedCostFixing = false
	sub.CutRowAgeLimit = 0
	sub.UseRINS = false
	sub.DisablePrimalHeuristics = true
	sub.MIPGap = 0
	return sub
}

type rinsRunResult struct {
	objective float64
	values    []float64
	built     bool
	improved  bool
	nodes     int
	lpSolves  int
	fixed     int
	discrete  int
}

// runRINS attempts one RINS call: fix variables agreeing with the
// incumbent, re-solve the reduced model with a small budget, and report the
// result if it improves on bestObj. ok=false whenever fixing wasn't
// possible or the sub-solve didn't produce a better feasible solution — the
// caller's search continues exactly as if RINS had not been attempted.
func runRINS(ctx context.Context, m *Model, direction Direction, lpValues, bestValues []float64, bestObj float64, outerNodes int, opts Options) (objective float64, values []float64, ok bool) {
	run := runRINSAtNode(ctx, m, direction, nil, lpValues, bestValues, bestObj,
		clampInt(rinsLegacyNodesOffset+int(rinsLegacyNodesQuot*float64(outerNodes)), rinsMinNodes, rinsLegacyMaxNodes), opts)
	return run.objective, run.values, run.improved
}

func runRINSAtNode(ctx context.Context, m *Model, direction Direction, nodeBounds []boundConstraint, lpValues, bestValues []float64, bestObj float64, nodeLimit int, opts Options) rinsRunResult {
	sub, fixing, built := rinsFixedNodeModelWithInfo(m, nodeBounds, lpValues, bestValues, opts.IntegralityTol)
	run := rinsRunResult{built: built, fixed: fixing.fixed, discrete: fixing.total}
	if !built {
		return run
	}
	if !addRINSObjectiveCutoff(sub, direction, objectiveAt(m, lpValues), bestObj, opts.FeasibilityTol) {
		run.built = false
		return run
	}
	res, err := sub.Solve(ctx, rinsSubOptionsForNodes(opts, nodeLimit))
	run.nodes = res.Nodes
	if res.Stats != nil {
		run.lpSolves = res.Stats.Revised.LPSolves
	}
	if err != nil || res.Values == nil {
		return run
	}
	if !betterObjective(direction, res.Objective, bestObj, opts.FeasibilityTol) {
		return run
	}
	obj, feasible := feasibleIncumbent(m, res.Values, opts)
	if !feasible {
		return run
	}
	run.objective = obj
	run.values = res.Values
	run.improved = true
	return run
}

func objectiveAt(m *Model, values []float64) float64 {
	obj := 0.0
	for _, term := range sortedCoeffs(m.objective) {
		obj += term.coef * values[term.v]
	}
	return obj
}

func addRINSObjectiveCutoff(sub *Model, direction Direction, lpBound, incumbent, tol float64) bool {
	if len(sub.objective) == 0 || !betterObjective(direction, lpBound, incumbent, tol) {
		return false
	}
	cutoff := (1-rinsMinImprove)*incumbent + rinsMinImprove*lpBound
	sense := LessOrEqual
	if direction == Minimize {
		cutoff = math.Min(cutoff, incumbent-tol)
	} else {
		sense = GreaterOrEqual
		cutoff = math.Max(cutoff, incumbent+tol)
	}
	coeffs := make(map[Var]float64, len(sub.objective))
	for v, coef := range sub.objective {
		coeffs[v] = coef
	}
	return sub.AddNamedConstraint("rins_objective_cutoff", coeffs, sense, cutoff) == nil
}

// lastIncumbentNode returns the B&B node count at which the most recent
// incumbent was recorded (rounding, feasibility pump, integer-feasible
// branch, or a prior RINS call all call stats.recordIncumbent), or 0 if
// none has been recorded yet.
func lastIncumbentNode(stats *SolverStats) int {
	if stats == nil || len(stats.Incumbents) == 0 {
		return 0
	}
	return stats.Incumbents[len(stats.Incumbents)-1].Node
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
