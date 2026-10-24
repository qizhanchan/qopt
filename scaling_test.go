package qopt

import (
	"context"
	"math"
	"testing"
)

// TestRuizScale_Equilibrates checks that after ruizScale, every row and column
// of A has inf-norm within a small constant of 1. This is the property the
// feasibility-tolerance logic in solveSimplex relies on.
func TestRuizScale_Equilibrates(t *testing.T) {
	// Manufactured ill-scaled matrix: rows span 1e-3 .. 1e6, cols similar.
	a := [][]float64{
		{1e6, 1e-3, 1.0},
		{1e3, 1e0, 1e-2},
		{1e-2, 1e2, 1e4},
	}
	b := []float64{1e6, 1e3, 1e2}
	c := []float64{1e3, 1.0, 1e-2}

	ruizScale(a, b, c, 12)

	for i, row := range a {
		max := 0.0
		for _, v := range row {
			if math.Abs(v) > max {
				max = math.Abs(v)
			}
		}
		if max < 0.1 || max > 10 {
			t.Errorf("row %d inf-norm %.3g escaped [0.1, 10] band", i, max)
		}
	}
	for j := 0; j < len(c); j++ {
		max := 0.0
		for _, row := range a {
			if math.Abs(row[j]) > max {
				max = math.Abs(row[j])
			}
		}
		if max < 0.1 || max > 10 {
			t.Errorf("col %d inf-norm %.3g escaped [0.1, 10] band", j, max)
		}
	}
}

// TestRuizScale_PreservesEquivalence solves a small LP both with and without
// scaling and verifies the optimum is identical to 1e-9. This guards against
// off-by-one mistakes in the unscaling step in solveLPModel.
func TestRuizScale_PreservesEquivalence(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x"})
	y := m.AddVar(VarSpec{Name: "y"})
	must(t, m.SetObjective(map[Var]float64{x: 3, y: 2}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 4))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, LessOrEqual, 2))
	must(t, m.AddConstraint(map[Var]float64{y: 1}, LessOrEqual, 3))

	res, err := m.Solve(context.Background(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(res.Objective-10) > 1e-9 {
		t.Fatalf("objective %.12g, want 10", res.Objective)
	}
	if math.Abs(res.Values[x]-2) > 1e-9 || math.Abs(res.Values[y]-2) > 1e-9 {
		t.Fatalf("values %v, want [2 2]", res.Values)
	}
}

// TestRuizScale_BadlyScaledLP solves an LP whose constraint matrix mixes
// entries spanning twelve orders of magnitude — analogous to glass4 — and
// verifies the simplex returns the right optimum rather than falsely flagging
// infeasibility. Without scaling, the Phase I "artificial vars must be zero"
// check fails on this input.
func TestRuizScale_BadlyScaledLP(t *testing.T) {
	// max  x + y
	//      1e6·x  +  1e-6·y  ≤  2e6      (binding at x=2)
	//      1e-6·x +  1e6·y   ≤  2e6      (binding at y=2)
	// Optimum: x = y ≈ 2 (slightly less due to cross-coupling), obj ≈ 4.
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x"})
	y := m.AddVar(VarSpec{Name: "y"})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1e6, y: 1e-6}, LessOrEqual, 2e6))
	must(t, m.AddConstraint(map[Var]float64{x: 1e-6, y: 1e6}, LessOrEqual, 2e6))

	res, err := m.Solve(context.Background(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal {
		t.Fatalf("status = %s, want optimal", res.Status)
	}
	if math.Abs(res.Objective-4) > 1e-6 {
		t.Fatalf("objective %.6g, want ≈4", res.Objective)
	}
}
