package qopt

import (
	"context"
	"math"
	"testing"
)

func TestLastIncumbentNode(t *testing.T) {
	if got := lastIncumbentNode(nil); got != 0 {
		t.Fatalf("nil stats: got %d, want 0", got)
	}
	s := &SolverStats{}
	if got := lastIncumbentNode(s); got != 0 {
		t.Fatalf("no incumbents yet: got %d, want 0", got)
	}
	s.recordIncumbent(0, 10, 5, "rounding")
	s.recordIncumbent(0, 37, 4, "rins")
	if got := lastIncumbentNode(s); got != 37 {
		t.Fatalf("got %d, want 37 (the most recent record)", got)
	}
}

func TestRinsFixedModel(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	z := m.AddVar(VarSpec{Name: "z", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1, z: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1, z: 1}, LessOrEqual, 2))

	t.Run("fixes variables agreeing with incumbent", func(t *testing.T) {
		lpValues := []float64{1, 0, 0.5}
		incumbent := []float64{1, 0, 1}
		sub, ok := rinsFixedModel(m, lpValues, incumbent, 1e-6)
		if !ok {
			t.Fatal("expected rinsFixedModel to build a sub-model")
		}
		if *sub.vars[x].Lower != 1 || *sub.vars[x].Upper != 1 {
			t.Fatalf("x should be fixed to 1, got [%v,%v]", *sub.vars[x].Lower, *sub.vars[x].Upper)
		}
		if *sub.vars[y].Lower != 0 || *sub.vars[y].Upper != 0 {
			t.Fatalf("y should be fixed to 0, got [%v,%v]", *sub.vars[y].Lower, *sub.vars[y].Upper)
		}
		// z disagrees (0.5 vs 1) -> must stay free.
		if *sub.vars[z].Lower != 0 || *sub.vars[z].Upper != 1 {
			t.Fatalf("z should stay free [0,1], got [%v,%v]", *sub.vars[z].Lower, *sub.vars[z].Upper)
		}
		// Original model's bounds must be untouched (fresh pointers, not shared).
		if *m.vars[x].Upper != 1 || *m.vars[x].Lower != 0 {
			t.Fatalf("original model x bounds mutated: [%v,%v]", *m.vars[x].Lower, *m.vars[x].Upper)
		}
	})

	t.Run("aborts when agreement is below the minimum fraction", func(t *testing.T) {
		// Only 1/3 variables agree, below rinsMinFixFraction (0.3 needs >=1
		// for 3 vars via float comparison; use values that clearly fail).
		lpValues := []float64{0.5, 0.5, 0.5}
		incumbent := []float64{1, 1, 1}
		if _, ok := rinsFixedModel(m, lpValues, incumbent, 1e-6); ok {
			t.Fatal("expected rinsFixedModel to abort with zero agreement")
		}
	})

	t.Run("aborts when every discrete var would be fixed", func(t *testing.T) {
		lpValues := []float64{1, 0, 1}
		incumbent := []float64{1, 0, 1}
		if _, ok := rinsFixedModel(m, lpValues, incumbent, 1e-6); ok {
			t.Fatal("expected rinsFixedModel to abort when nothing is left to search")
		}
	})
}

func TestShouldTryRINS(t *testing.T) {
	off := Options{}
	on := Options{UseRINS: true}
	if shouldTryRINS(off, true, rinsPeriodNodes, 0) {
		t.Fatal("RINS must not fire when UseRINS is false")
	}
	if shouldTryRINS(on, false, rinsPeriodNodes, 0) {
		t.Fatal("RINS must not fire without an incumbent")
	}
	if shouldTryRINS(on, true, 0, 0) {
		t.Fatal("RINS must not fire at node 0")
	}
	if shouldTryRINS(on, true, rinsPeriodNodes-1, 0) {
		t.Fatal("RINS must not fire off-cadence")
	}
	if !shouldTryRINS(on, true, rinsPeriodNodes, 0) {
		t.Fatal("RINS must fire exactly on cadence when stalled since node 0")
	}
	if !shouldTryRINS(on, true, rinsPeriodNodes*3, 0) {
		t.Fatal("RINS must fire on every cadence multiple")
	}
	// The stall gate: even on-cadence, RINS must not fire if the incumbent
	// improved too recently.
	if shouldTryRINS(on, true, rinsPeriodNodes*3, rinsPeriodNodes*3-1) {
		t.Fatal("RINS must not fire right after an incumbent improvement")
	}
	if !shouldTryRINS(on, true, rinsPeriodNodes*3, rinsPeriodNodes*3-rinsWaitNodes) {
		t.Fatal("RINS must fire once exactly rinsWaitNodes have passed since the last improvement")
	}
}

func TestAdaptiveRINSCandidateRequiresMatureLargeGapNode(t *testing.T) {
	if adaptiveRINSCandidate(Minimize, 0, 100, rinsAdaptiveMinNodesSeen-1, 1e-8) {
		t.Fatal("adaptive RINS must wait for enough outer search nodes")
	}
	if adaptiveRINSCandidate(Minimize, 60, 100, rinsAdaptiveMinNodesSeen, 1e-8) {
		t.Fatal("40% minimization gap must not trigger adaptive RINS")
	}
	if !adaptiveRINSCandidate(Minimize, 50, 100, rinsAdaptiveMinNodesSeen, 1e-8) {
		t.Fatal("50% minimization gap should trigger adaptive RINS")
	}
	if !adaptiveRINSCandidate(Maximize, 100, 50, rinsAdaptiveMinNodesSeen, 1e-8) {
		t.Fatal("100% maximization gap should trigger adaptive RINS")
	}
	if adaptiveRINSCandidate(Maximize, 40, 50, rinsAdaptiveMinNodesSeen, 1e-8) {
		t.Fatal("LP bound worse than the maximization incumbent must not trigger")
	}
}

func TestRinsSubOptionsDisablesRecursionAndOtherHeuristics(t *testing.T) {
	outer := Options{
		UseRINS: true, MaxCutRounds: 5, MaxCutDepth: 5, UseMIRCuts: true,
		DetectSymmetry: true, PseudoObjPropagation: true, ReducedCostFixing: true,
		CutRowAgeLimit: 10, LPBackend: LPBackendRevised, MIPBranching: MIPBranchReliability,
	}
	sub := rinsSubOptions(outer, 1000)
	if sub.UseRINS {
		t.Fatal("sub-solve must not re-trigger RINS (infinite recursion risk)")
	}
	if sub.MaxCutRounds != 0 || sub.MaxCutDepth != 0 || sub.UseMIRCuts {
		t.Fatal("sub-solve must not run cuts")
	}
	if sub.DetectSymmetry || sub.PseudoObjPropagation || sub.ReducedCostFixing || sub.CutRowAgeLimit != 0 {
		t.Fatal("sub-solve must not run the other opt-in mechanisms")
	}
	if sub.LPBackend != LPBackendRevised || sub.MIPBranching != MIPBranchReliability {
		t.Fatal("sub-solve should inherit backend/branching from the outer solve")
	}
	if sub.MaxNodes < rinsMinNodes || sub.MaxNodes > rinsLegacyMaxNodes {
		t.Fatalf("sub-solve node budget %d out of [%d,%d]", sub.MaxNodes, rinsMinNodes, rinsLegacyMaxNodes)
	}
	if sub.TimeLimit <= 0 {
		t.Fatal("sub-solve must have a bounded time limit")
	}
}

func TestRinsAdaptiveNodeBudgetAccountsForHistory(t *testing.T) {
	var state rinsState
	if got := state.nodeBudget(200, true); got != 560 {
		t.Fatalf("first adaptive budget = %d, want 560", got)
	}
	state.record(200, rinsRunResult{built: true, nodes: 100})
	if got := state.nodeBudget(400, true); got != 360 {
		t.Fatalf("budget after failed 100-node call = %d, want 360", got)
	}

	rewarded := rinsState{calls: 1, successes: 1, usedNodes: 100}
	if got := rewarded.nodeBudget(400, true); got != 420 {
		t.Fatalf("budget after successful call = %d, want 420", got)
	}
}

func TestRinsAdaptiveFirstBuiltFailureExhaustsNeighborhood(t *testing.T) {
	var state rinsState
	state.record(200, rinsRunResult{built: true})
	if !state.exhausted || state.eligible(100000) {
		t.Fatal("first built failure should exhaust adaptive RINS for this solve")
	}
}

func TestRinsAdaptiveFailureBackoffAfterSuccess(t *testing.T) {
	state := rinsState{calls: 1, successes: 1}
	state.record(600, rinsRunResult{built: true})
	if state.exhausted {
		t.Fatal("a failure after an earlier success must not exhaust RINS")
	}
	if state.eligible(999) || !state.eligible(1000) {
		t.Fatalf("failure next eligible node = %d, want 1000", state.nextEligibleNode)
	}
	state.record(1000, rinsRunResult{built: true, improved: true})
	if state.consecutiveFails != 0 || state.nextEligibleNode != 1200 {
		t.Fatalf("success did not reset backoff: fails=%d next=%d", state.consecutiveFails, state.nextEligibleNode)
	}
}

func TestRinsFixedModelInheritsNodeBounds(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer, Lower: Float64(0), Upper: Float64(10)})
	_ = m.AddVar(VarSpec{Name: "y", Type: Integer, Lower: Float64(0), Upper: Float64(10)})
	z := m.AddVar(VarSpec{Name: "z", Type: Integer, Lower: Float64(0), Upper: Float64(10)})
	bounds := []boundConstraint{{varIndex: int(z), sense: GreaterOrEqual, rhs: 4}}
	sub, ok := rinsFixedNodeModel(m, bounds, []float64{2, 3.5, 4.5}, []float64{2, 4, 5}, 1e-6)
	if !ok {
		t.Fatal("expected a RINS neighborhood")
	}
	if *sub.vars[x].Lower != 2 || *sub.vars[x].Upper != 2 {
		t.Fatalf("agreeing x not fixed: [%g,%g]", *sub.vars[x].Lower, *sub.vars[x].Upper)
	}
	if *sub.vars[z].Lower != 4 || *sub.vars[z].Upper != 10 {
		t.Fatalf("node lower bound not inherited: [%g,%g]", *sub.vars[z].Lower, *sub.vars[z].Upper)
	}
	if *m.vars[z].Lower != 0 {
		t.Fatalf("original bound mutated to %g", *m.vars[z].Lower)
	}
}

func TestRinsFixingInfoCountsAgreement(t *testing.T) {
	m := NewModel(Minimize)
	_ = m.AddVar(VarSpec{Name: "x", Type: Binary})
	_ = m.AddVar(VarSpec{Name: "y", Type: Integer, Lower: Float64(0), Upper: Float64(10)})
	_ = m.AddVar(VarSpec{Name: "z", Type: Integer, Lower: Float64(0), Upper: Float64(10)})
	_, info, ok := rinsFixedNodeModelWithInfo(m, nil, []float64{1, 2, 3.5}, []float64{1, 2, 4}, 1e-6)
	if !ok {
		t.Fatal("expected a valid RINS neighborhood")
	}
	if info.fixed != 2 || info.total != 3 {
		t.Fatalf("fixing info = %d/%d, want 2/3", info.fixed, info.total)
	}
}

func TestRinsObjectiveCutoffRequiresOnePercentGapImprovement(t *testing.T) {
	minModel := NewModel(Minimize)
	x := minModel.AddVar(VarSpec{Name: "x", Type: Continuous})
	must(t, minModel.SetObjective(map[Var]float64{x: 1}))
	if !addRINSObjectiveCutoff(minModel, Minimize, 0, 100, 1e-8) {
		t.Fatal("expected minimization cutoff")
	}
	minCut := minModel.constraints[len(minModel.constraints)-1]
	if minCut.Sense != LessOrEqual || math.Abs(minCut.RHS-99) > 1e-9 {
		t.Fatalf("min cutoff sense=%v rhs=%g, want <= 99", minCut.Sense, minCut.RHS)
	}

	maxModel := NewModel(Maximize)
	y := maxModel.AddVar(VarSpec{Name: "y", Type: Continuous})
	must(t, maxModel.SetObjective(map[Var]float64{y: 1}))
	if !addRINSObjectiveCutoff(maxModel, Maximize, 10, 0, 1e-8) {
		t.Fatal("expected maximization cutoff")
	}
	maxCut := maxModel.constraints[len(maxModel.constraints)-1]
	if maxCut.Sense != GreaterOrEqual || math.Abs(maxCut.RHS-0.1) > 1e-9 {
		t.Fatalf("max cutoff sense=%v rhs=%g, want >= 0.1", maxCut.Sense, maxCut.RHS)
	}
}

// TestRunRINSFindsBetterIncumbent exercises the whole path end to end: an
// outer incumbent that is NOT optimal, an LP relaxation whose fractional
// var already agrees with the incumbent on the rest, and a fixed sub-model
// that finds a strictly better solution.
func TestRunRINSFindsBetterIncumbent(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	z := m.AddVar(VarSpec{Name: "z", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 6, y: 5, z: 8}))
	must(t, m.AddConstraint(map[Var]float64{x: 4, y: 3, z: 5}, LessOrEqual, 8))

	// Suboptimal incumbent (x=1,y=0,z=0 -> obj=6); true optimum is x=0,y=0,z=1 (obj=8)
	// or x=1,y=1,z=0 (obj=11, infeasible: 4+3=7<=8 actually feasible, obj=11 is the
	// true optimum). Use bestValues/lpValues that agree except on y so RINS's
	// fix-and-resolve can discover the y=1 improvement.
	bestValues := []float64{1, 0, 0}
	lpValues := []float64{1, 0.5, 0} // x, z agree with incumbent; y is fractional
	opts := normalizeOptions(Options{})

	obj, values, ok := runRINS(context.Background(), m, Maximize, lpValues, bestValues, 6, 100, opts)
	if !ok {
		t.Fatal("expected runRINS to find an improving solution")
	}
	if obj <= 6+1e-6 {
		t.Fatalf("obj = %g, want > 6", obj)
	}
	gotObj, feasible := feasibleIncumbent(m, values, opts)
	if !feasible {
		t.Fatal("runRINS returned an infeasible point")
	}
	if math.Abs(gotObj-obj) > 1e-6 {
		t.Fatalf("returned objective %g doesn't match recomputed %g", obj, gotObj)
	}
}

// TestRINSNeverWorsensOptimum: enabling UseRINS end-to-end on small MIPs
// with known optima must never move the final objective away from the
// no-RINS baseline — RINS only ever installs a result that passed
// feasibleIncumbent and beat the previous best, so it can refine but never
// regress the reported answer.
func TestRINSNeverWorsensOptimum(t *testing.T) {
	cases := []struct {
		name  string
		build func() *Model
	}{
		{"binary-knapsack", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x", Type: Binary})
			y := m.AddVar(VarSpec{Name: "y", Type: Binary})
			z := m.AddVar(VarSpec{Name: "z", Type: Binary})
			must(t, m.SetObjective(map[Var]float64{x: 6, y: 5, z: 8}))
			must(t, m.AddConstraint(map[Var]float64{x: 4, y: 3, z: 5}, LessOrEqual, 8))
			return m
		}},
		{"integer-production", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x", Type: Integer})
			y := m.AddVar(VarSpec{Name: "y", Type: Integer})
			must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
			must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
			must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))
			return m
		}},
	}
	for _, tc := range cases {
		baseline := solveOK(t, tc.build(), Options{})
		for _, variant := range []Options{
			{UseRINS: true},
			{UseRINS: true, LPBackend: LPBackendRevised},
			{UseRINS: true, MIPNodeSelection: MIPBestBound},
			{UseRINS: true, MIPBranching: MIPBranchReliability},
		} {
			got := solveOK(t, tc.build(), variant)
			if math.Abs(got.Objective-baseline.Objective) > 1e-6 {
				t.Fatalf("%s: baseline=%.12g rins=%.12g (opts=%+v)", tc.name, baseline.Objective, got.Objective, variant)
			}
		}
	}
}
