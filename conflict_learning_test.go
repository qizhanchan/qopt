package qopt

import (
	"context"
	"testing"
)

func TestLPConflictPropagationAndPrune(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	sc := &searchContext{
		work:  m,
		opts:  Options{UseLPConflictLearning: true},
		stats: &SolverStats{},
	}
	if !sc.lpConflicts.add([]boundLiteral{
		{varIndex: int(x), sense: GreaterOrEqual, rhs: 1},
		{varIndex: int(y), sense: GreaterOrEqual, rhs: 1},
	}) {
		t.Fatal("failed to add conflict")
	}

	bounds, pruned := sc.applyLPConflicts([]boundConstraint{{varIndex: int(x), sense: GreaterOrEqual, rhs: 1}})
	if pruned {
		t.Fatal("unit conflict propagation pruned a feasible partial assignment")
	}
	lower, upper := effectiveBounds(m, bounds)
	if lower[y] != 0 || upper[y] != 0 {
		t.Fatalf("conflict did not force y=0: [%g,%g]", lower[y], upper[y])
	}
	if sc.stats.Branching.LPConflictPropagations != 1 {
		t.Fatalf("propagations=%d, want 1", sc.stats.Branching.LPConflictPropagations)
	}

	_, pruned = sc.applyLPConflicts([]boundConstraint{
		{varIndex: int(x), sense: GreaterOrEqual, rhs: 1},
		{varIndex: int(y), sense: GreaterOrEqual, rhs: 1},
	})
	if !pruned {
		t.Fatal("complete no-good assignment was not pruned")
	}
	if sc.stats.Branching.LPConflictPrunes != 1 {
		t.Fatalf("prunes=%d, want 1", sc.stats.Branching.LPConflictPrunes)
	}
}

func TestLearnLPConflictMinimizesIrrelevantLiteral(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	z := m.AddVar(VarSpec{Name: "z", Type: Binary})
	u := m.AddVar(VarSpec{Name: "u", Type: Continuous, Lower: Float64(-100), Upper: Float64(100)})
	v := m.AddVar(VarSpec{Name: "v", Type: Continuous, Lower: Float64(-100), Upper: Float64(100)})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1, z: 1}))
	// x=1 implies u-v=1; y=1 implies u-v=0. Each equality alone is
	// feasible over the free continuous variables, so row propagation cannot
	// prove the contradiction without the LP's combined Phase-I certificate.
	must(t, m.AddConstraint(map[Var]float64{u: 1, v: -1, x: -1}, Equal, 0))
	must(t, m.AddConstraint(map[Var]float64{u: 1, v: -1, y: 1}, Equal, 1))

	literals := []boundLiteral{
		{varIndex: int(x), sense: GreaterOrEqual, rhs: 1},
		{varIndex: int(y), sense: GreaterOrEqual, rhs: 1},
		{varIndex: int(z), sense: GreaterOrEqual, rhs: 1},
	}
	sc := &searchContext{
		ctx:  context.Background(),
		work: m,
		opts: Options{
			LPBackend:             LPBackendRevised,
			UseLPConflictLearning: true,
			FeasibilityTol:        1e-8,
			OptimalityTol:         1e-9,
			MaxIterations:         10000,
		},
		stats:     &SolverStats{},
		workCache: buildCanonCache(m),
	}
	node := mipNode{
		bounds:         literalBounds(literals),
		depth:          3,
		branchLiterals: literals,
	}
	res, resp, err := solveNodeOrFull(context.Background(), sc.workCache, m, nil, node.bounds, sc.opts, nil)
	if err != nil || res.Status != Infeasible || resp.infeasibility == nil {
		t.Fatalf("expected Phase-I infeasibility certificate, status=%s cert=%v err=%v", res.Status, resp.infeasibility != nil, err)
	}
	sc.learnLPConflict(node, resp.infeasibility)
	if len(sc.lpConflicts.conflicts) != 1 {
		t.Fatalf("learned conflicts=%d, want 1", len(sc.lpConflicts.conflicts))
	}
	got := sc.lpConflicts.conflicts[0].literals
	if len(got) != 2 || !conflictSubset([]boundLiteral{
		{varIndex: int(x), sense: GreaterOrEqual, rhs: 1},
		{varIndex: int(y), sense: GreaterOrEqual, rhs: 1},
	}, got) {
		t.Fatalf("learned conflict=%v, want x=1,y=1", got)
	}
	if sc.stats.Branching.LPConflictsLearned != 1 || sc.stats.Branching.LPConflictLiterals != 2 {
		t.Fatalf("unexpected conflict stats: %+v", sc.stats.Branching)
	}
	if sc.stats.Branching.LPConflictLPSolves != 0 || sc.stats.Branching.LPConflictCertificateProofs == 0 {
		t.Fatalf("certificate minimization used fallback LPs or proved nothing: %+v", sc.stats.Branching)
	}
}

func TestLearnLPConflictMinimizationReplaysPropagation(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	z := m.AddVar(VarSpec{Name: "z", Type: Binary})
	u := m.AddVar(VarSpec{Name: "u", Type: Continuous, Lower: Float64(-100), Upper: Float64(100)})
	v := m.AddVar(VarSpec{Name: "v", Type: Continuous, Lower: Float64(-100), Upper: Float64(100)})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1, z: 1}))
	// z=1 propagates x=1. The LP contradiction itself still needs Phase I:
	// x=1 imposes u-v=1 while y=1 imposes u-v=0.
	must(t, m.AddConstraint(map[Var]float64{x: 1, z: -1}, GreaterOrEqual, 0))
	must(t, m.AddConstraint(map[Var]float64{u: 1, v: -1, x: -1}, Equal, 0))
	must(t, m.AddConstraint(map[Var]float64{u: 1, v: -1, y: 1}, Equal, 1))

	literals := []boundLiteral{
		{varIndex: int(x), sense: GreaterOrEqual, rhs: 1},
		{varIndex: int(y), sense: GreaterOrEqual, rhs: 1},
		{varIndex: int(z), sense: GreaterOrEqual, rhs: 1},
	}
	sc := &searchContext{
		ctx:  context.Background(),
		work: m,
		opts: Options{
			LPBackend:             LPBackendRevised,
			UseLPConflictLearning: true,
			FeasibilityTol:        1e-8,
			OptimalityTol:         1e-9,
			MaxIterations:         10000,
		},
		stats:     &SolverStats{},
		workCache: buildCanonCache(m),
	}
	node := mipNode{bounds: literalBounds(literals), depth: 3, branchLiterals: literals}
	res, resp, err := solveNodeOrFull(context.Background(), sc.workCache, m, nil, node.bounds, sc.opts, nil)
	if err != nil || res.Status != Infeasible || resp.infeasibility == nil {
		t.Fatalf("expected Phase-I certificate, status=%s cert=%v err=%v", res.Status, resp.infeasibility != nil, err)
	}
	sc.learnLPConflict(node, resp.infeasibility)
	if len(sc.lpConflicts.conflicts) != 1 {
		t.Fatalf("learned conflicts=%d, want 1", len(sc.lpConflicts.conflicts))
	}
	got := sc.lpConflicts.conflicts[0].literals
	want := []boundLiteral{
		{varIndex: int(y), sense: GreaterOrEqual, rhs: 1},
		{varIndex: int(z), sense: GreaterOrEqual, rhs: 1},
	}
	if len(got) != 2 || !conflictSubset(want, got) {
		t.Fatalf("learned conflict=%v, want propagated reasons y=1,z=1", got)
	}
}

func TestLPConflictPropagatesGeneralIntegerBound(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer, Upper: Float64(10)})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer, Upper: Float64(10)})
	sc := &searchContext{work: m, opts: Options{UseLPConflictLearning: true}, stats: &SolverStats{}}
	if !sc.lpConflicts.add([]boundLiteral{
		{varIndex: int(x), sense: LessOrEqual, rhs: 3},
		{varIndex: int(y), sense: GreaterOrEqual, rhs: 7},
	}) {
		t.Fatal("failed to add general-integer conflict")
	}
	bounds, pruned := sc.applyLPConflicts([]boundConstraint{{varIndex: int(x), sense: LessOrEqual, rhs: 3}})
	if pruned {
		t.Fatal("general-integer unit propagation pruned the node")
	}
	lower, upper := effectiveBounds(m, bounds)
	if lower[y] != 0 || upper[y] != 6 {
		t.Fatalf("negated y>=7 literal did not force y<=6: [%g,%g]", lower[y], upper[y])
	}
}
