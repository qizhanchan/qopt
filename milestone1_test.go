package qopt

import (
	"context"
	"math"
	"testing"
	"time"
)

func TestDomainStateQueuedPropagationAndReason(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Upper: Float64(10)})
	y := m.AddVar(VarSpec{Name: "y", Upper: Float64(2)})
	must(t, m.AddNamedConstraint("demand", map[Var]float64{x: 1, y: 1}, GreaterOrEqual, 5))
	lower, upper := effectiveBounds(m, nil)
	d := newDomainState(m, lower, upper, 1e-9)
	if !d.propagate(5) {
		t.Fatal("propagation unexpectedly proved infeasible")
	}
	if math.Abs(lower[x]-3) > 1e-9 {
		t.Fatalf("x lower=%g, want 3", lower[x])
	}
	r := d.lowerReason[x]
	if r.Row != 0 || r.Kind != reasonRowLower || math.Abs(r.Bound-3) > 1e-9 {
		t.Fatalf("unexpected reason: %+v", r)
	}
}

func TestShiftAndPropagateRepairsRoundedEquality(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, Equal, 1))
	opts := normalizeOptions(Options{})
	obj, values, ok := repairRoundedIncumbent(m, []float64{0.6, 0.6}, opts)
	if !ok {
		t.Fatal("repair did not find a feasible point")
	}
	if obj != 1 || math.Abs(values[x]+values[y]-1) > opts.FeasibilityTol {
		t.Fatalf("obj=%g values=%v, want one selected binary", obj, values)
	}
}

func TestGeneralIntegerFeasibilityPump(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer, Upper: Float64(1)})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer, Upper: Float64(1)})
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, Equal, 1))
	opts := normalizeOptions(Options{TimeLimit: time.Second})
	values, ok, solves := runFeasibilityPump(context.Background(), m, m, []float64{0.5, 0.5}, opts, time.Now())
	if !ok || solves == 0 {
		t.Fatalf("pump ok=%v solves=%d values=%v", ok, solves, values)
	}
	if _, feasible := feasibleIncumbent(m, values, opts); !feasible {
		t.Fatalf("pump returned infeasible values %v", values)
	}
}

func TestPseudocostDivingReturnsFeasiblePoint(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 2, y: 2}, LessOrEqual, 3))
	opts := normalizeOptions(Options{})
	root, resp, _, err := solveLPModelWithBackend(context.Background(), m, nil, opts, nil)
	if err != nil || root.Status != Optimal {
		t.Fatalf("root status=%s err=%v", root.Status, err)
	}
	values, ok, _ := runPseudocostDiving(context.Background(), m, m, nil, nil, root.Values, resp.warm, opts, newPseudocostState(2), nil)
	if !ok {
		t.Fatal("diving did not return a feasible point")
	}
	if _, feasible := feasibleIncumbent(m, values, opts); !feasible {
		t.Fatalf("diving returned infeasible values %v", values)
	}
}

func TestHeuristicSchedulerRecordsBudgetedAttempt(t *testing.T) {
	stats := &SolverStats{}
	h := newHeuristicScheduler(context.Background(), Options{TimeLimit: time.Second}, stats)
	ctx, finish, ok := h.begin("test", 1)
	if !ok || ctx == nil {
		t.Fatal("scheduler rejected an in-budget attempt")
	}
	finish(true, 1)
	got := stats.Heuristics["test"]
	if got.Attempts != 1 || got.Successes != 1 || got.LPSolves != 1 {
		t.Fatalf("unexpected heuristic stats: %+v", got)
	}
}

func TestHeuristicSchedulerUsesSeparateNeighborhoodTimeBudget(t *testing.T) {
	stats := &SolverStats{}
	h := newHeuristicScheduler(context.Background(), Options{TimeLimit: time.Minute}, stats)
	h.lpUsed = h.lpBudget
	ctx, finish, ok := h.beginNeighborhood("proximity")
	if !ok || ctx == nil {
		t.Fatal("LP quota should not block a time-budgeted neighborhood")
	}
	finish(false, 500)
	if h.lpUsed != h.lpBudget+500 || stats.Heuristics["proximity"].LPSolves != 500 {
		t.Fatalf("neighborhood LP usage was not recorded: used=%d stats=%+v", h.lpUsed, stats.Heuristics["proximity"])
	}
}

func TestRoundAndFixUsesContinuousRecourse(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Continuous, Lower: Float64(0), Upper: Float64(1)})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, Equal, 0.5))
	opts := normalizeOptions(Options{})
	root, _, _, err := solveLPModelWithBackend(context.Background(), m, nil, opts, nil)
	if err != nil || root.Status != Optimal {
		t.Fatalf("root status=%s err=%v", root.Status, err)
	}
	values, ok, solves := runRoundAndFix(context.Background(), m, root.Values, opts, nil)
	if !ok || solves < 1 {
		t.Fatalf("round-and-fix failed: ok=%v solves=%d", ok, solves)
	}
	if _, feasible := feasibleIncumbent(m, values, opts); !feasible {
		t.Fatalf("round-and-fix returned infeasible values %v", values)
	}
}

func TestCoefficientDivingUsesRowLocks(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, GreaterOrEqual, 0.5))
	opts := normalizeOptions(Options{})
	root, resp, _, err := solveLPModelWithBackend(context.Background(), m, nil, opts, nil)
	if err != nil || root.Status != Optimal {
		t.Fatalf("root status=%s err=%v", root.Status, err)
	}
	values, ok, _ := runCoefficientDiving(context.Background(), m, m, nil, nil, root.Values, resp.warm, opts, nil)
	if !ok {
		t.Fatal("coefficient diving did not find a feasible point")
	}
	if _, feasible := feasibleIncumbent(m, values, opts); !feasible {
		t.Fatalf("coefficient diving returned infeasible values %v", values)
	}
}

func TestCoefficientDivingUsesSingleVariableTailBatches(t *testing.T) {
	for _, fractional := range []int{1, 6, coefficientDivingTailFractionals} {
		if got := coefficientDiveBatchSize(fractional, true); got != 1 {
			t.Fatalf("batch size for %d fractional variables = %d, want 1", fractional, got)
		}
	}
	if got := coefficientDiveBatchSize(coefficientDivingTailFractionals, false); got != 4 {
		t.Fatalf("batch size after tail entry = %d, want 4", got)
	}
	if got := coefficientDiveBatchSize(coefficientDivingTailFractionals+1, true); got != 4 {
		t.Fatalf("batch size above tail = %d, want 4", got)
	}
}

func TestRENSFindsFeasibleNeighborhoodSolution(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	z := m.AddVar(VarSpec{Name: "z", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 2, z: 3}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1, z: 1}, Equal, 1))
	opts := normalizeOptions(Options{})
	values, ok, _ := runRENS(context.Background(), m, []float64{0, 0.49, 0.51}, opts)
	if !ok {
		t.Fatal("RENS failed to find a feasible neighborhood solution")
	}
	if _, feasible := feasibleIncumbent(m, values, opts); !feasible {
		t.Fatalf("RENS returned infeasible values %v", values)
	}
}
