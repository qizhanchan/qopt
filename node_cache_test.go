package qopt

import (
	"context"
	"math"
	"testing"
)

// TestCanonCacheMatchesFullSolve verifies the cross-node canonicalize cache
// (solveNode) produces the same LP status/objective/values as the generic
// canonicalize+scale builder (solveLPModelWithBackend) for a range of
// branching bound configurations on the revised backend. The cache skips
// fixed-variable elimination and singleton-row deletion, so this guards
// against those reductions changing the LP result.
func TestCanonCacheMatchesFullSolve(t *testing.T) {
	build := func() *Model {
		m := NewModel(Minimize)
		x := m.AddVar(VarSpec{Name: "x", Type: Integer, Lower: Float64(0), Upper: Float64(10)})
		y := m.AddVar(VarSpec{Name: "y", Type: Integer, Lower: Float64(0), Upper: Float64(10)})
		z := m.AddVar(VarSpec{Name: "z", Type: Continuous, Lower: Float64(0), Upper: Float64(8)})
		must(t, m.SetObjective(map[Var]float64{x: -3, y: -2, z: 1}))
		must(t, m.AddConstraint(map[Var]float64{x: 2, y: 1, z: 1}, LessOrEqual, 14))
		must(t, m.AddConstraint(map[Var]float64{x: 1, y: 3}, LessOrEqual, 18))
		must(t, m.AddConstraint(map[Var]float64{x: 1, z: 1}, GreaterOrEqual, 1))
		return m
	}

	opts := Options{LPBackend: LPBackendRevised, FeasibilityTol: 1e-8, OptimalityTol: 1e-9, IntegralityTol: 1e-8, MaxIterations: 100000}

	cases := [][]boundConstraint{
		nil,
		{{varIndex: 0, sense: LessOrEqual, rhs: 3}},
		{{varIndex: 0, sense: GreaterOrEqual, rhs: 4}},
		{{varIndex: 0, sense: GreaterOrEqual, rhs: 2}, {varIndex: 1, sense: LessOrEqual, rhs: 2}},
		{{varIndex: 1, sense: Equal, rhs: 5}},
	}

	m := build()
	cache := buildCanonCache(m)
	ctx := context.Background()

	for i, bounds := range cases {
		full, _, _, err := solveLPModelWithBackend(ctx, m, bounds, opts, nil)
		if err != nil {
			t.Fatalf("case %d: full solve error: %v", i, err)
		}
		cached, _ := cache.solveNode(ctx, bounds, opts, nil)
		if cached.Status != full.Status {
			t.Fatalf("case %d: status mismatch cached=%v full=%v", i, cached.Status, full.Status)
		}
		if full.Status != Optimal {
			continue
		}
		if math.Abs(cached.Objective-full.Objective) > 1e-6 {
			t.Fatalf("case %d: objective mismatch cached=%g full=%g", i, cached.Objective, full.Objective)
		}
		for j := range full.Values {
			if math.Abs(cached.Values[j]-full.Values[j]) > 1e-6 {
				t.Fatalf("case %d: var %d mismatch cached=%g full=%g", i, j, cached.Values[j], full.Values[j])
			}
		}
	}
}

// TestCanonCacheMIPSolves checks that a full MIP solve on the revised backend
// (which now routes node LPs through the cache) reaches the known optimum.
func TestCanonCacheMIPSolves(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer, Lower: Float64(0), Upper: Float64(100)})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer, Lower: Float64(0), Upper: Float64(100)})
	must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
	must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))

	res, err := m.Solve(context.Background(), Options{LPBackend: LPBackendRevised, MIPBranching: MIPBranchReliability})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal {
		t.Fatalf("status=%v, want Optimal", res.Status)
	}
	// max 5x+4y s.t. 6x+4y≤24, x+2y≤6, integer: optimum is x=4, y=0, obj=20.
	if math.Abs(res.Objective-20) > 1e-6 {
		t.Fatalf("objective=%g, want 20", res.Objective)
	}
}
