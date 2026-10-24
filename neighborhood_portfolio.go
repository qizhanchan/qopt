package qopt

import (
	"context"
	"time"
)

func (sc *searchContext) tryIncumbentNeighborhood(node mipNode, lpValues []float64) {
	if !sc.haveIncumbent || neighborhoodHeuristicCount(sc.opts) == 0 {
		return
	}
	opts := sc.opts
	incumbentNode := lastIncumbentNode(sc.stats)
	lpBound := objectiveAt(sc.original, lpValues)
	primalGap := mipRelativeGap(lpBound, sc.bestObj, true)
	multi := neighborhoodHeuristicCount(opts) > 1
	if multi && sc.neighborhoods.exhausted {
		return
	}
	candidates := make([]neighborhoodCandidate, 0, 3)

	if shouldTryProximity(opts, sc.original, true, sc.nodes, incumbentNode) &&
		sc.proximity.eligible(incumbentNode) &&
		(!multi || sc.neighborhoods.allow(neighborhoodProximity, primalGap)) {
		candidates = append(candidates, neighborhoodCandidate{
			kind:  neighborhoodProximity,
			prior: neighborhoodPrior(neighborhoodProximity, primalGap),
		})
	}

	rinsReady := shouldTryRINS(opts, true, sc.nodes, incumbentNode)
	if rinsReady && opts.RINSAdaptiveBudget {
		rinsReady = sc.rins.eligible(sc.nodes) &&
			adaptiveRINSCandidate(sc.work.direction, lpBound, sc.bestObj, sc.nodes, opts.FeasibilityTol)
	}
	if rinsReady && sc.rins.nodeBudget(sc.nodes, opts.RINSAdaptiveBudget) > 0 {
		candidates = append(candidates, neighborhoodCandidate{
			kind:  neighborhoodRINS,
			prior: neighborhoodPrior(neighborhoodRINS, primalGap),
		})
	}

	if shouldTryLocalBranching(opts, sc.original, true, sc.nodes, incumbentNode) &&
		sc.localBranching.eligible(incumbentNode) &&
		(!multi || sc.neighborhoods.allow(neighborhoodLocalBranching, primalGap)) {
		candidates = append(candidates, neighborhoodCandidate{
			kind:  neighborhoodLocalBranching,
			prior: neighborhoodPrior(neighborhoodLocalBranching, primalGap),
		})
	}

	if len(candidates) == 0 {
		return
	}
	selected := candidates[0].kind
	if multi {
		var ok bool
		selected, ok = sc.neighborhoods.selectCandidate(candidates)
		if !ok {
			return
		}
	}
	debugf("neighborhood portfolio: outer_nodes=%d incumbent_node=%d gap=%.6g selected=%s candidates=%d",
		sc.nodes, incumbentNode, primalGap, selected.name(), len(candidates))

	switch selected {
	case neighborhoodProximity:
		sc.runProximityNeighborhood(lpBound, incumbentNode, multi)
	case neighborhoodRINS:
		sc.runRINSNeighborhood(node, lpValues, lpBound, multi)
	case neighborhoodLocalBranching:
		sc.runLocalBranchingNeighborhood(lpBound, incumbentNode, multi)
	}
}

func (sc *searchContext) runProximityNeighborhood(lpBound float64, incumbentNode int, multi bool) {
	hctx, finish, allowed := sc.heuristics.beginNeighborhood(neighborhoodProximity.name())
	if !allowed {
		return
	}
	before := sc.bestObj
	started := time.Now()
	runCtx, cancel := context.WithTimeout(hctx, proximityMaxCallTime)
	run := runProximity(runCtx, sc.original, sc.bestValues, before, lpBound, sc.nodes, sc.opts)
	cancel()
	elapsed := time.Since(started)
	finish(run.improved, run.lpSolves)
	sc.proximity.record(incumbentNode)
	debugf("proximity: outer_nodes=%d node_limit=%d sub_nodes=%d lp_solves=%d improved=%t",
		sc.nodes, run.nodeLimit, run.nodes, run.lpSolves, run.improved)
	if run.improved {
		sc.bestObj = run.objective
		sc.bestValues = run.values
		sc.stats.recordIncumbent(time.Since(sc.start), sc.nodes, sc.bestObj, neighborhoodProximity.name())
	}
	sc.observeNeighborhood(neighborhoodProximity, before, sc.bestObj, lpBound, elapsed, run.improved, multi)
}

func (sc *searchContext) runRINSNeighborhood(node mipNode, lpValues []float64, lpBound float64, multi bool) {
	nodeBudget := sc.rins.nodeBudget(sc.nodes, sc.opts.RINSAdaptiveBudget)
	if nodeBudget <= 0 {
		return
	}
	hctx, finish, allowed := sc.heuristics.beginNeighborhood(neighborhoodRINS.name())
	if !allowed {
		return
	}
	before := sc.bestObj
	started := time.Now()
	runCtx := hctx
	cancel := func() {}
	if sc.opts.RINSAdaptiveBudget {
		runCtx, cancel = context.WithTimeout(hctx, rinsAdaptiveMaxCallTime)
	}
	run := runRINSAtNode(runCtx, sc.original, sc.work.direction, node.bounds,
		lpValues, sc.bestValues, before, nodeBudget, sc.opts)
	cancel()
	elapsed := time.Since(started)
	fixRate := 0.0
	if run.discrete > 0 {
		fixRate = float64(run.fixed) / float64(run.discrete)
	}
	debugf("rins: outer_nodes=%d fixed=%d/%d rate=%.3f built=%t node_limit=%d sub_nodes=%d lp_solves=%d improved=%t",
		sc.nodes, run.fixed, run.discrete, fixRate, run.built, nodeBudget, run.nodes, run.lpSolves, run.improved)
	finish(run.improved, run.lpSolves)
	sc.rins.record(sc.nodes, run)
	if run.improved {
		sc.bestObj = run.objective
		sc.bestValues = run.values
		sc.stats.recordIncumbent(time.Since(sc.start), sc.nodes, sc.bestObj, neighborhoodRINS.name())
	}
	sc.observeNeighborhood(neighborhoodRINS, before, sc.bestObj, lpBound, elapsed, run.improved, multi)
}

func (sc *searchContext) runLocalBranchingNeighborhood(lpBound float64, incumbentNode int, multi bool) {
	hctx, finish, allowed := sc.heuristics.beginNeighborhood(neighborhoodLocalBranching.name())
	if !allowed {
		return
	}
	before := sc.bestObj
	started := time.Now()
	runCtx, cancel := context.WithTimeout(hctx, localBranchingMaxCallTime)
	run := runLocalBranching(runCtx, sc.original, sc.work.direction, sc.bestValues,
		before, sc.nodes, sc.opts)
	cancel()
	elapsed := time.Since(started)
	finish(run.improved, run.lpSolves)
	sc.localBranching.record(incumbentNode)
	debugf("localbranching: outer_nodes=%d radius=%d node_limit=%d sub_nodes=%d lp_solves=%d improved=%t",
		sc.nodes, run.radius, run.nodeLimit, run.nodes, run.lpSolves, run.improved)
	if run.improved {
		sc.bestObj = run.objective
		sc.bestValues = run.values
		sc.stats.recordIncumbent(time.Since(sc.start), sc.nodes, sc.bestObj, neighborhoodLocalBranching.name())
	}
	sc.observeNeighborhood(neighborhoodLocalBranching, before, sc.bestObj, lpBound, elapsed, run.improved, multi)
}

func (sc *searchContext) observeNeighborhood(kind neighborhoodKind, before, after, lpBound float64, elapsed time.Duration, success, multi bool) {
	reward := neighborhoodReward(sc.work.direction, before, after, lpBound, elapsed, sc.opts.FeasibilityTol)
	sc.stats.recordHeuristicReward(kind.name(), reward)
	if multi {
		afterGap := mipRelativeGap(lpBound, after, true)
		sc.neighborhoods.observe(kind, reward, success, elapsed, afterGap)
	}
	debugf("neighborhood reward: name=%s success=%t reward=%.6g elapsed_ms=%.3f",
		kind.name(), success, reward, float64(elapsed.Microseconds())/1000)
}
