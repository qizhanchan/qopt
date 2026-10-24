package qopt

import (
	"context"
	"math"
	"testing"
)

func TestTightenBounds_SingletonRow(t *testing.T) {
	// x ≥ 0 by default; constraint 2*x ≤ 6 implies x ≤ 3.
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x"})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 2}, LessOrEqual, 6))

	lower := []float64{0}
	upper := []float64{math.Inf(1)}
	if tightenBounds(m, lower, upper, 5) {
		t.Fatal("unexpected infeasible")
	}
	if upper[0] != 3 {
		t.Fatalf("upper = %v, want [3]", upper)
	}
}

func TestTightenBounds_Cascade(t *testing.T) {
	// x + y ≤ 5; y ≥ 4 → forces x ≤ 1.
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x"})
	y := m.AddVar(VarSpec{Name: "y"})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 5))
	must(t, m.AddConstraint(map[Var]float64{y: 1}, GreaterOrEqual, 4))

	lower := []float64{0, 0}
	upper := []float64{math.Inf(1), math.Inf(1)}
	if tightenBounds(m, lower, upper, 5) {
		t.Fatal("unexpected infeasible")
	}
	if lower[1] < 4-1e-9 {
		t.Fatalf("y lower = %v, want ≥4", lower[1])
	}
	if upper[0] > 1+1e-9 {
		t.Fatalf("x upper = %v, want ≤1", upper[0])
	}
}

func TestTightenBounds_IntegerFloor(t *testing.T) {
	// 3*x ≤ 10, x integer → x ≤ floor(10/3) = 3.
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 3}, LessOrEqual, 10))

	lower := []float64{0}
	upper := []float64{math.Inf(1)}
	if tightenBounds(m, lower, upper, 5) {
		t.Fatal("unexpected infeasible")
	}
	if upper[0] != 3 {
		t.Fatalf("upper = %v, want 3", upper)
	}
}

func TestTightenBounds_NegativeCoeff(t *testing.T) {
	// -2*x ≤ -6 → x ≥ 3.
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x"})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: -2}, LessOrEqual, -6))

	lower := []float64{0}
	upper := []float64{math.Inf(1)}
	if tightenBounds(m, lower, upper, 5) {
		t.Fatal("unexpected infeasible")
	}
	if lower[0] < 3-1e-9 {
		t.Fatalf("lower = %v, want ≥3", lower)
	}
}

func TestTightenBounds_InfeasibleByConstraint(t *testing.T) {
	// x ∈ [2, 5]; constraint x ≤ 1 → infeasible.
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Lower: Float64(2), Upper: Float64(5)})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, LessOrEqual, 1))

	lower := []float64{2}
	upper := []float64{5}
	if !tightenBounds(m, lower, upper, 5) {
		t.Fatal("expected infeasible")
	}
}

func TestTightenBounds_EqualityForcesBoth(t *testing.T) {
	// x + y = 4; both in [0, +inf). For each var, the other can be 0..inf,
	// so we can only derive: 0 ≤ x ≤ 4, 0 ≤ y ≤ 4.
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x"})
	y := m.AddVar(VarSpec{Name: "y"})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, Equal, 4))

	lower := []float64{0, 0}
	upper := []float64{math.Inf(1), math.Inf(1)}
	if tightenBounds(m, lower, upper, 5) {
		t.Fatal("unexpected infeasible")
	}
	if upper[0] != 4 || upper[1] != 4 {
		t.Fatalf("upper = %v, want [4 4]", upper)
	}
}

func TestTightenBounds_ForcingLessEqualRow(t *testing.T) {
	// With nonnegative vars, x + y ≤ 0 is feasible only at x = y = 0.
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x"})
	y := m.AddVar(VarSpec{Name: "y"})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 0))

	lower := []float64{0, 0}
	upper := []float64{math.Inf(1), math.Inf(1)}
	if tightenBounds(m, lower, upper, 5) {
		t.Fatal("unexpected infeasible")
	}
	if upper[0] != 0 || upper[1] != 0 {
		t.Fatalf("upper = %v, want [0 0]", upper)
	}
}

func TestTightenBounds_ForcingGreaterEqualRow(t *testing.T) {
	// x + y ≥ 5 with x∈[0,2], y∈[0,3] is feasible only at upper bounds.
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x"})
	y := m.AddVar(VarSpec{Name: "y"})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, GreaterOrEqual, 5))

	lower := []float64{0, 0}
	upper := []float64{2, 3}
	if tightenBounds(m, lower, upper, 5) {
		t.Fatal("unexpected infeasible")
	}
	if lower[0] != 2 || lower[1] != 3 {
		t.Fatalf("lower = %v, want [2 3]", lower)
	}
}

func TestCanonicalize_DropsFixedVariables(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Lower: Float64(2), Upper: Float64(2)})
	y := m.AddVar(VarSpec{Name: "y", Upper: Float64(10)})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 3}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 5))

	lp := canonicalize(m, nil)
	if len(lp.c) != 1 || len(lp.colToVar) != 1 || lp.colToVar[0] != int(y) {
		t.Fatalf("reduced columns = c:%v colToVar:%v, want only y", lp.c, lp.colToVar)
	}
	if !lp.fixed[x] || lp.fixedVal[x] != 2 {
		t.Fatalf("x fixed = %v value %v, want true/2", lp.fixed[x], lp.fixedVal[x])
	}
	if math.Abs(lp.offset-2) > 1e-12 {
		t.Fatalf("offset = %v, want 2", lp.offset)
	}
	if len(lp.a) != 1 {
		t.Fatalf("rows = %d, want only y UB row after singleton reduction", len(lp.a))
	}
	if len(lp.a[0]) != 1 || lp.a[0][0] != 1 || lp.b[0] != 3 || !lp.rowKinds[0].isUpperBound {
		t.Fatalf("remaining row = %v <= %v kind=%+v, want UB [1] <= 3", lp.a[0], lp.b[0], lp.rowKinds[0])
	}

	res, err := m.Solve(context.Background(), Options{LPBackend: LPBackendRevised})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal {
		t.Fatalf("status = %s, want optimal", res.Status)
	}
	if len(res.Values) != 2 || math.Abs(res.Values[x]-2) > 1e-9 || math.Abs(res.Values[y]-3) > 1e-9 {
		t.Fatalf("values = %v, want x=2 y=3", res.Values)
	}
	if math.Abs(res.Objective-11) > 1e-9 {
		t.Fatalf("objective = %v, want 11", res.Objective)
	}
}

func TestCanonicalize_RemovesSingletonUpperRow(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x"})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 2}, LessOrEqual, 6))

	lp := canonicalize(m, nil)
	if len(lp.c) != 1 {
		t.Fatalf("columns = %d, want 1", len(lp.c))
	}
	if lp.upper[0] != 3 {
		t.Fatalf("upper = %v, want 3", lp.upper)
	}
	if len(lp.a) != 1 || !lp.rowKinds[0].isUpperBound {
		t.Fatalf("rows = %d kinds=%v, want only synthetic UB row", len(lp.a), lp.rowKinds)
	}

	for _, backend := range []LPBackend{LPBackendTableau, LPBackendRevised} {
		res, err := m.Solve(context.Background(), Options{LPBackend: backend})
		if err != nil {
			t.Fatal(err)
		}
		if res.Status != Optimal || len(res.Values) != 1 || math.Abs(res.Values[0]-3) > 1e-9 || math.Abs(res.Objective-3) > 1e-9 {
			t.Fatalf("%s result = status %s values %v obj %v, want optimal x=3 obj=3", backend, res.Status, res.Values, res.Objective)
		}
	}
}

func TestCanonicalize_RemovesSingletonLowerRow(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x"})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: -2}, LessOrEqual, -6))

	lp := canonicalize(m, nil)
	if len(lp.c) != 1 {
		t.Fatalf("columns = %d, want 1", len(lp.c))
	}
	if lp.lower[0] != 3 {
		t.Fatalf("lower = %v, want 3", lp.lower)
	}
	if len(lp.a) != 0 {
		t.Fatalf("rows = %d, want singleton lower row removed", len(lp.a))
	}

	for _, backend := range []LPBackend{LPBackendTableau, LPBackendRevised} {
		res, err := m.Solve(context.Background(), Options{LPBackend: backend})
		if err != nil {
			t.Fatal(err)
		}
		if res.Status != Optimal || len(res.Values) != 1 || math.Abs(res.Values[0]-3) > 1e-9 || math.Abs(res.Objective-3) > 1e-9 {
			t.Fatalf("%s result = status %s values %v obj %v, want optimal x=3 obj=3", backend, res.Status, res.Values, res.Objective)
		}
	}
}

func TestCanonicalize_KeepsIntegerSingletonRow(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 2}, LessOrEqual, 6))

	lp := canonicalize(m, nil)
	if len(lp.c) != 1 {
		t.Fatalf("columns = %d, want 1", len(lp.c))
	}
	if lp.upper[0] != 3 {
		t.Fatalf("upper = %v, want 3", lp.upper)
	}
	if len(lp.a) != 2 || lp.rowKinds[0].isUpperBound || !lp.rowKinds[1].isUpperBound {
		t.Fatalf("rows = %d kinds=%+v, want user singleton retained plus synthetic UB", len(lp.a), lp.rowKinds)
	}
}

func TestCanonicalize_RemovesRedundantFixedRows(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Lower: Float64(2), Upper: Float64(2)})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, LessOrEqual, 3))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, GreaterOrEqual, 1))

	lp := canonicalize(m, nil)
	if len(lp.c) != 0 {
		t.Fatalf("columns = %d, want 0", len(lp.c))
	}
	if len(lp.a) != 0 {
		t.Fatalf("rows = %d, want redundant fixed rows removed", len(lp.a))
	}

	res, err := m.Solve(context.Background(), Options{LPBackend: LPBackendRevised})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal || len(res.Values) != 1 || res.Values[0] != 2 || res.Objective != 2 {
		t.Fatalf("res = status %s values %v obj %v, want optimal [2] obj 2", res.Status, res.Values, res.Objective)
	}
}

func TestCanonicalize_FixedRowInfeasible(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Lower: Float64(2), Upper: Float64(2)})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, LessOrEqual, 1))

	lp := canonicalize(m, nil)
	if len(lp.c) != 0 {
		t.Fatalf("columns = %d, want 0", len(lp.c))
	}
	if len(lp.a) != 1 || len(lp.a[0]) != 0 || lp.b[0] != -1 {
		t.Fatalf("infeasible row = %v <= %v, want empty <= -1", lp.a, lp.b)
	}

	res, err := m.Solve(context.Background(), Options{LPBackend: LPBackendRevised})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Infeasible {
		t.Fatalf("status = %s, want infeasible", res.Status)
	}
}

// TestPresolve_BranchCollapse drives the integration: build a model where
// a single bound from branching plus the constraint should immediately
// prove infeasibility through presolve, before the simplex runs. With the
// node-bound mechanism this is the common B&B pruning scenario.
func TestPresolve_BranchCollapse(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 2, y: 2}, LessOrEqual, 3))

	res, err := m.Solve(context.Background(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal {
		t.Fatalf("status = %s, want optimal", res.Status)
	}
	if math.Abs(res.Objective-1) > 1e-9 {
		t.Fatalf("objective = %v, want 1", res.Objective)
	}
}
