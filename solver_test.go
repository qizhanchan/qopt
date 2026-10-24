package qopt

import (
	"context"
	"math"
	"testing"
)

func TestLPSimpleFeasibleOptimum(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x"})
	y := m.AddVar(VarSpec{Name: "y"})
	must(t, m.SetObjective(map[Var]float64{x: 3, y: 2}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 4))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, LessOrEqual, 2))
	must(t, m.AddConstraint(map[Var]float64{y: 1}, LessOrEqual, 3))

	res := solveOK(t, m, Options{})
	closeTo(t, res.Objective, 10)
	closeTo(t, res.Values[x], 2)
	closeTo(t, res.Values[y], 2)
}

func TestLPEqualityConstraint(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x"})
	y := m.AddVar(VarSpec{Name: "y"})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, Equal, 5))

	res := solveOK(t, m, Options{})
	closeTo(t, res.Objective, 5)
	closeTo(t, res.Values[x]+res.Values[y], 5)
}

func TestLPGreaterOrEqualConstraint(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x"})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, GreaterOrEqual, 3))

	res := solveOK(t, m, Options{})
	closeTo(t, res.Objective, 3)
	closeTo(t, res.Values[x], 3)
}

func TestLPFiniteBounds(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Lower: Float64(2), Upper: Float64(5)})
	must(t, m.SetObjective(map[Var]float64{x: 1}))

	res := solveOK(t, m, Options{})
	closeTo(t, res.Objective, 2)
	closeTo(t, res.Values[x], 2)
}

func TestBinaryExplicitBoundsAreIntersected(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary, Lower: Float64(1), Upper: Float64(1)})
	must(t, m.SetObjective(map[Var]float64{x: -1}))
	res := solveOK(t, m, Options{})
	closeTo(t, res.Values[x], 1)
	closeTo(t, res.Objective, -1)
}

func TestNarrowContinuousIntervalIsNotFixedAway(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x"})
	must(t, m.SetObjective(map[Var]float64{x: 1e12}))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, LessOrEqual, 1e-10))
	res := solveOK(t, m, Options{})
	if math.Abs(res.Values[x]-1e-10) > 1e-14 {
		t.Fatalf("x=%g, want 1e-10", res.Values[x])
	}
	if math.Abs(res.Objective-100) > 1e-6 {
		t.Fatalf("objective=%g, want 100", res.Objective)
	}
}

func TestLPInfeasible(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x"})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, GreaterOrEqual, 2))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, LessOrEqual, 1))

	res, err := m.Solve(context.Background(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Infeasible {
		t.Fatalf("status = %s, want infeasible", res.Status)
	}
}

func TestLPUnbounded(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x"})
	must(t, m.SetObjective(map[Var]float64{x: 1}))

	res, err := m.Solve(context.Background(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Unbounded {
		t.Fatalf("status = %s, want unbounded", res.Status)
	}
}

func TestLPDirectionConsistency(t *testing.T) {
	max := NewModel(Maximize)
	x1 := max.AddVar(VarSpec{Name: "x"})
	must(t, max.SetObjective(map[Var]float64{x1: 1}))
	must(t, max.AddConstraint(map[Var]float64{x1: 1}, LessOrEqual, 7))

	min := NewModel(Minimize)
	x2 := min.AddVar(VarSpec{Name: "x"})
	must(t, min.SetObjective(map[Var]float64{x2: -1}))
	must(t, min.AddConstraint(map[Var]float64{x2: 1}, LessOrEqual, 7))

	r1 := solveOK(t, max, Options{})
	r2 := solveOK(t, min, Options{})
	closeTo(t, r1.Objective, -r2.Objective)
	closeTo(t, r1.Values[x1], r2.Values[x2])
}

func TestMIPBinaryKnapsack(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	z := m.AddVar(VarSpec{Name: "z", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 6, y: 5, z: 8}))
	must(t, m.AddConstraint(map[Var]float64{x: 4, y: 3, z: 5}, LessOrEqual, 8))

	res := solveOK(t, m, Options{})
	closeTo(t, res.Objective, 13)
	closeTo(t, res.Values[y], 1)
	closeTo(t, res.Values[z], 1)
}

func TestMIPIntegerProduction(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
	must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))

	res := solveOK(t, m, Options{})
	closeTo(t, res.Objective, 20)
	closeTo(t, res.Values[x], 4)
	closeTo(t, res.Values[y], 0)
}

func TestMIPRequiresBranching(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 2, y: 2}, LessOrEqual, 3))

	res := solveOK(t, m, Options{})
	closeTo(t, res.Objective, 1)
	if math.Round(res.Values[x]+res.Values[y]) != 1 {
		t.Fatalf("values = %v, want one selected integer unit", res.Values)
	}
}

func TestMIPInfeasibleInteger(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, GreaterOrEqual, 0.2))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, LessOrEqual, 0.8))

	res, err := m.Solve(context.Background(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Infeasible {
		t.Fatalf("status = %s, want infeasible", res.Status)
	}
}

func TestMIPNodeLimit(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 2, y: 2}, LessOrEqual, 3))

	res, err := m.Solve(context.Background(), Options{MaxNodes: 1})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != NodeLimit {
		t.Fatalf("status = %s, want node_limit", res.Status)
	}
}

func TestMIPIncumbentPruning(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 10, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 1))

	res := solveOK(t, m, Options{})
	closeTo(t, res.Objective, 10)
	closeTo(t, res.Values[x], 1)
	closeTo(t, res.Values[y], 0)
	if res.Nodes > 3 {
		t.Fatalf("nodes = %d, expected pruning to keep search small", res.Nodes)
	}
}

func TestMIPPriorityNodeSelectionSolvesSameModel(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
	must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))

	for _, selection := range []MIPNodeSelection{MIPBestBound, MIPBestEstimate} {
		res := solveOK(t, m, Options{MIPNodeSelection: selection})
		closeTo(t, res.Objective, 20)
		closeTo(t, res.Values[x], 4)
		closeTo(t, res.Values[y], 0)
		if res.NodeSelection != selection {
			t.Fatalf("node selection = %s, want %s", res.NodeSelection, selection)
		}
	}
}

func TestFeasibleIncumbentRejectsInvalidValues(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y"})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 2}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 1))

	if _, ok := feasibleIncumbent(m, []float64{1, 1}, normalizeOptions(Options{})); ok {
		t.Fatal("expected incumbent violating row constraint to be rejected")
	}
	if _, ok := feasibleIncumbent(m, []float64{0.5, 0}, normalizeOptions(Options{})); ok {
		t.Fatal("expected fractional binary incumbent to be rejected")
	}
	obj, ok := feasibleIncumbent(m, []float64{1, 0}, normalizeOptions(Options{}))
	if !ok {
		t.Fatal("expected feasible incumbent to be accepted")
	}
	closeTo(t, obj, 1)
}

func TestRoundedIncumbentAcceptsOnlyFeasibleCandidates(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 10, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 1))

	obj, values, ok := roundedIncumbent(m, []float64{0.9, 0.2}, normalizeOptions(Options{}))
	if !ok {
		t.Fatal("expected rounding heuristic to find a feasible incumbent")
	}
	closeTo(t, obj, 10)
	closeTo(t, values[x], 1)
	closeTo(t, values[y], 0)

	_, _, ok = roundedIncumbent(m, []float64{0.6, 0.6}, normalizeOptions(Options{}))
	if !ok {
		t.Fatal("expected at least one feasible rounded candidate")
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func solveOK(t *testing.T, m *Model, opts Options) Result {
	t.Helper()
	res, err := m.Solve(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal {
		t.Fatalf("status = %s, want optimal", res.Status)
	}
	return res
}

func closeTo(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-7 {
		t.Fatalf("got %.12g, want %.12g", got, want)
	}
}
