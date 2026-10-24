package qopt

import (
	"container/heap"
	"context"
	"math"
	"testing"
	"time"
)

func TestPseudocostFindsSameOptimumOnSmallMIPs(t *testing.T) {
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
		{"binary-pruning", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x", Type: Binary})
			y := m.AddVar(VarSpec{Name: "y", Type: Binary})
			must(t, m.SetObjective(map[Var]float64{x: 10, y: 1}))
			must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 1))
			return m
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mf := solveOK(t, tc.build(), Options{MIPBranching: MIPBranchMostFractional})
			pc := solveOK(t, tc.build(), Options{MIPBranching: MIPBranchPseudocost})
			if math.Abs(mf.Objective-pc.Objective) > 1e-6 {
				t.Fatalf("objective: most-frac=%.12g pseudocost=%.12g", mf.Objective, pc.Objective)
			}
		})
	}
}

func TestReliabilityFindsSameOptimumOnSmallMIPs(t *testing.T) {
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
		t.Run(tc.name, func(t *testing.T) {
			mf := solveOK(t, tc.build(), Options{MIPBranching: MIPBranchMostFractional})
			// Reliability branching on both backends: the tableau path takes the
			// cold-probe branch (warm == nil), the revised path the warm-probe
			// branch — both must land on the same optimum as most-fractional.
			relTab := solveOK(t, tc.build(), Options{MIPBranching: MIPBranchReliability})
			relRev := solveOK(t, tc.build(), Options{MIPBranching: MIPBranchReliability, LPBackend: LPBackendRevised})
			if math.Abs(mf.Objective-relTab.Objective) > 1e-6 {
				t.Fatalf("objective (tableau): most-frac=%.12g reliability=%.12g", mf.Objective, relTab.Objective)
			}
			if math.Abs(mf.Objective-relRev.Objective) > 1e-6 {
				t.Fatalf("objective (revised): most-frac=%.12g reliability=%.12g", mf.Objective, relRev.Objective)
			}
		})
	}
}

func TestPseudocostWithRevisedBackend(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
	must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))
	res, err := m.Solve(context.Background(), Options{
		MIPBranching: MIPBranchPseudocost,
		LPBackend:    LPBackendRevised,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal {
		t.Fatalf("status = %s", res.Status)
	}
	closeTo(t, res.Objective, 20)
}

func TestPseudocostStateUpdates(t *testing.T) {
	p := newPseudocostState(3)
	p.observeDown(1, 4.0, 0.5) // gain per unit = 8
	p.observeDown(1, 6.0, 0.5) // gain per unit = 12; avg = 10
	if g := p.downGain(1); math.Abs(g-10) > 1e-9 {
		t.Fatalf("downGain(1) = %g, want 10", g)
	}
	// Var 0 has no own observations but a global average exists.
	if g := p.downGain(0); math.Abs(g-10) > 1e-9 {
		t.Fatalf("downGain(0) (global fallback) = %g, want 10", g)
	}
	// Up direction still cold globally.
	if g := p.upGain(0); g != 0 {
		t.Fatalf("upGain(0) cold = %g, want 0", g)
	}

	// Cold-start scoring uses most-fractional measure (f * (1-f)).
	cold := newPseudocostState(1)
	if s := cold.score(0, 0.5); math.Abs(s-0.25) > 1e-9 {
		t.Fatalf("cold score at f=0.5 = %g, want 0.25", s)
	}
}

func TestMIPCandidateEstimateUsesCheaperPseudocostDirection(t *testing.T) {
	vars := []VarSpec{{Type: Integer}, {Type: Binary}, {}}
	values := []float64{0.25, 0.75, 0.5}
	p := newPseudocostState(len(vars))
	p.observeDown(0, 8, 1)
	p.observeUp(0, 4, 1)
	p.observeDown(1, 6, 1)
	p.observeUp(1, 10, 1)

	// x contributes min(.25*8, .75*4)=2; y contributes
	// min(.75*6, .25*10)=2.5. The continuous variable is ignored.
	if got := mipCandidateEstimate(Minimize, 10, vars, values, p, 1e-8); math.Abs(got-14.5) > 1e-9 {
		t.Fatalf("minimize estimate = %g, want 14.5", got)
	}
	if got := mipCandidateEstimate(Maximize, 10, vars, values, p, 1e-8); math.Abs(got-5.5) > 1e-9 {
		t.Fatalf("maximize estimate = %g, want 5.5", got)
	}
}

func TestReliabilityProbeIterationBudget(t *testing.T) {
	tests := []struct {
		node, strong int
		urgent       bool
		want         int
	}{
		{node: 0, strong: 0, want: 1600},
		{node: 800, strong: 500, want: 1500},
		{node: 100, strong: 1650, want: 0},
		{node: 0, strong: 0, urgent: true, want: 1600},
	}
	for _, tc := range tests {
		if got := reliabilityProbeIterationBudget(tc.node, tc.strong, tc.urgent, 0); got != tc.want {
			t.Fatalf("budget(%d,%d) = %d, want %d", tc.node, tc.strong, got, tc.want)
		}
	}
	if got := reliabilityProbeIterationBudget(0, 0, true, 100000); got != 100000 {
		t.Fatalf("overridden urgent budget = %d, want 100000", got)
	}
}

func TestReliabilityThresholdForNode(t *testing.T) {
	tests := []struct {
		name    string
		opts    Options
		lpIters int
		strong  int
		nodes   int
		want    int
	}{
		{name: "default", opts: Options{}, lpIters: 100, nodes: 100, want: 5},
		{name: "adaptive warmup", opts: Options{ReliabilityAdaptiveThreshold: true}, lpIters: 100, nodes: 63, want: 5},
		{name: "adaptive cheap", opts: Options{ReliabilityAdaptiveThreshold: true}, lpIters: 1280, nodes: 64, want: 2},
		{name: "adaptive expensive", opts: Options{ReliabilityAdaptiveThreshold: true}, lpIters: 1281, nodes: 64, want: 5},
		{name: "explicit override", opts: Options{ReliabilityThreshold: 3, ReliabilityAdaptiveThreshold: true}, lpIters: 1, nodes: 1000, want: 3},
		{name: "budget exhausted", opts: Options{}, strong: 1600, nodes: 1, want: 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := reliabilityThresholdForNode(tc.opts, tc.lpIters, tc.strong, tc.nodes); got != tc.want {
				t.Fatalf("threshold = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestReliabilityProbeIterationLimit(t *testing.T) {
	tests := []struct {
		iters, nodes int
		want         int
	}{
		{iters: 0, nodes: 0, want: 500},
		{iters: 100, nodes: 1, want: 500},
		{iters: 1000, nodes: 100, want: 24},
		{iters: 100, nodes: 100, want: 10},
	}
	for _, tc := range tests {
		if got := reliabilityProbeIterationLimit(tc.iters, tc.nodes); got != tc.want {
			t.Fatalf("limit(%d,%d)=%d, want %d", tc.iters, tc.nodes, got, tc.want)
		}
	}
}

func TestAdaptiveRepairCadence(t *testing.T) {
	sc := &searchContext{
		opts:          Options{RepairAdaptiveCadence: true},
		stats:         &SolverStats{Incumbents: []IncumbentEvent{{Node: 1}}},
		haveIncumbent: true,
		nodes:         201,
	}
	if !sc.shouldRunIncumbentRepair() {
		t.Fatal("first incumbent repair should run after 200 nodes")
	}
	sc.recordRepairResult(false)
	if sc.repairInterval != 400 || sc.shouldRunIncumbentRepair() {
		t.Fatalf("first failure interval=%d due=%v", sc.repairInterval, sc.shouldRunIncumbentRepair())
	}
	sc.nodes += 400
	if !sc.shouldRunIncumbentRepair() {
		t.Fatal("repair should run after backed-off interval")
	}
	sc.recordRepairResult(false)
	sc.nodes += 800
	sc.recordRepairResult(false)
	sc.nodes += 1600
	sc.recordRepairResult(false)
	sc.nodes += 3200
	sc.recordRepairResult(false)
	if sc.repairInterval != 3200 {
		t.Fatalf("repair interval=%d, want cap 3200", sc.repairInterval)
	}
	sc.recordRepairResult(true)
	if sc.repairInterval != 200 {
		t.Fatalf("successful repair interval=%d, want reset 200", sc.repairInterval)
	}
}

func TestReliabilityReusesProbeProofAndChildLP(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Upper: Float64(1)})
	z := m.AddVar(VarSpec{Name: "z", Upper: Float64(1)})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{y: 1, x: -1}, LessOrEqual, 0))
	must(t, m.AddConstraint(map[Var]float64{z: 1, x: -1}, LessOrEqual, 0))
	must(t, m.AddConstraint(map[Var]float64{y: 1, z: 1}, GreaterOrEqual, 1))

	res := solveOK(t, m, Options{
		MIPBranching:            MIPBranchReliability,
		LPBackend:               LPBackendRevised,
		DisablePrimalHeuristics: true,
	})
	closeTo(t, res.Objective, 1)
	if res.Stats.Branching.ReliabilityInfeasibleSides == 0 {
		t.Fatal("expected reliability probing to prove the x<=0 side infeasible")
	}
	if res.Stats.Branching.ReliabilityChildrenSkipped == 0 {
		t.Fatal("expected the proved-infeasible child to be skipped")
	}
	if res.Stats.Branching.ReliabilityBoundReuses == 0 {
		t.Fatal("expected the feasible probe objective to be reused as a child bound")
	}
	if res.Stats.Branching.ReliabilityNativeProbes == 0 || res.Stats.Branching.ReliabilityFallbackProbes != 0 {
		t.Fatalf("native/fallback probes = %d/%d, want native-only", res.Stats.Branching.ReliabilityNativeProbes, res.Stats.Branching.ReliabilityFallbackProbes)
	}
	if res.Stats.Branching.ReliabilityWarmReuses != 0 {
		t.Fatalf("probe warm reuses = %d, SCIP-style probes must restore the parent basis", res.Stats.Branching.ReliabilityWarmReuses)
	}
	if res.Nodes > 2 {
		t.Fatalf("nodes = %d, want at most root plus the surviving child", res.Nodes)
	}
}

func TestBestEstimateQueuePlungesBeforeHeap(t *testing.T) {
	open := &mipBestBoundHeap{direction: Minimize, selection: MIPBestEstimate}
	heap.Init(open)
	heap.Push(open, mipCandidate{objective: 1, estimate: 1})
	q := &bestBoundQueue{
		open:   open,
		plunge: []mipCandidate{{objective: 10, estimate: 10}},
	}
	sc := &searchContext{ctx: context.Background(), start: time.Now(), opts: Options{}, hasGeneralInts: true}

	cand, status, err := q.next(sc)
	if err != nil || status != nextBranch {
		t.Fatalf("next status=%v err=%v", status, err)
	}
	if cand.objective != 10 {
		t.Fatalf("selected objective=%g, want plunge child 10", cand.objective)
	}
}
