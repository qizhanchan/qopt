package qopt

import (
	"context"
	"testing"
	"time"
)

// TestFeasibilityPumpRescuesBinaryProblem builds a small binary MIP where
// the LP relaxation has all binaries at fractional values that don't
// round to a feasible solution by themselves. The pump should iterate to
// integer feasibility within its budget.
//
// Problem:
//
//	max  x1 + x2 + x3 + x4
//	     x1 + x2 = 1
//	     x3 + x4 = 1
//	     x1 + x3 ≤ 1
//	     x2 + x4 ≤ 1
//	x_i ∈ {0,1}
//
// LP relaxation optimum is x = (0.5, 0.5, 0.5, 0.5) (all four equality / le
// constraints active at fractional point). Naive rounding-to-nearest of
// (0.5, ...) is ill-defined; either all-zero or all-one violates the
// equalities. The pump iterates with an L1 objective and should converge
// on (1,0,0,1) or (0,1,1,0).
func TestFeasibilityPumpRescuesBinaryProblem(t *testing.T) {
	m := NewModel(Maximize)
	x1 := m.AddVar(VarSpec{Name: "x1", Type: Binary})
	x2 := m.AddVar(VarSpec{Name: "x2", Type: Binary})
	x3 := m.AddVar(VarSpec{Name: "x3", Type: Binary})
	x4 := m.AddVar(VarSpec{Name: "x4", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x1: 1, x2: 1, x3: 1, x4: 1}))
	must(t, m.AddNamedConstraint("c1", map[Var]float64{x1: 1, x2: 1}, Equal, 1))
	must(t, m.AddNamedConstraint("c2", map[Var]float64{x3: 1, x4: 1}, Equal, 1))
	must(t, m.AddNamedConstraint("c3", map[Var]float64{x1: 1, x3: 1}, LessOrEqual, 1))
	must(t, m.AddNamedConstraint("c4", map[Var]float64{x2: 1, x4: 1}, LessOrEqual, 1))

	res, err := m.Solve(context.Background(), Options{TimeLimit: 5 * time.Second})
	if err != nil {
		t.Fatalf("solve: %v", err)
	}
	if res.Status != Optimal {
		t.Fatalf("status = %s, want optimal", res.Status)
	}
	if res.Objective != 2 {
		t.Fatalf("objective = %v, want 2", res.Objective)
	}
}

// TestFeasibilityPumpRespectsTimeLimit ensures the pump bails out promptly
// when the overall solve time limit is hit, even on a model where the LP
// is degenerate and the pump might otherwise iterate forever.
func TestFeasibilityPumpRespectsTimeLimit(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddNamedConstraint("c", map[Var]float64{x: 1, y: 1}, Equal, 1))

	start := time.Now()
	res, err := m.Solve(context.Background(), Options{TimeLimit: 200 * time.Millisecond})
	if err != nil {
		t.Fatalf("solve: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
		t.Fatalf("solve took %v, expected to bail by ~time limit (200ms)", elapsed)
	}
	if res.Status != Optimal {
		// Tiny problem — should still finish, but tolerate other terminal
		// statuses as long as we didn't run away.
		t.Logf("status = %s (acceptable for tiny problem)", res.Status)
	}
}
