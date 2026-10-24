package qopt

import (
	"context"
	"math"
	"testing"
)

// dualProofTestModel builds a model whose LP infeasibility under the branch
// bound z <= 0 is invisible to interval bound propagation (the Gauss-Seidel
// sweeps shrink the [0,10] boxes by roughly the 0.2 conflict gap per pass,
// so tightenBounds' five passes cannot empty them), so detection must come
// from the LP's certified Farkas exit:
//
//	min x + y + 3z
//	x + y + z >= 10.2
//	x + y - z <= 10.0
//	x, y in [0,10] continuous, z in {0,1,2} integer
//
// The two rows force x+y in [10.2-z, 10.0+z]: empty for z < 0.1, so the LP
// optimum sits at fractional z = 0.1 and the z <= 0 child is infeasible while
// z >= 1 is feasible.
func dualProofTestModel(t *testing.T) (*Model, Var, Var, Var) {
	t.Helper()
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Type: Continuous, Lower: Float64(0), Upper: Float64(10)})
	y := m.AddVar(VarSpec{Name: "y", Type: Continuous, Lower: Float64(0), Upper: Float64(10)})
	z := m.AddVar(VarSpec{Name: "z", Type: Integer, Lower: Float64(0), Upper: Float64(2)})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1, z: 3}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1, z: 1}, GreaterOrEqual, 10.2))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1, z: -1}, LessOrEqual, 10.0))
	return m, x, y, z
}

func TestDualProofPoolLearnsAndPrunes(t *testing.T) {
	m, _, _, z := dualProofTestModel(t)
	opts := normalizeOptions(Options{LPBackend: LPBackendRevised, MaxIterations: 100000})
	cache := buildCanonCache(m)
	cache.dualProofs = newDualProofPool(m, opts.FeasibilityTol)
	ctx := context.Background()

	root, rootResp := cache.solveNode(ctx, nil, opts, nil)
	if root.Status != Optimal {
		t.Fatalf("root status = %v", root.Status)
	}
	if math.Abs(root.Values[int(z)]-0.1) > 1e-6 {
		t.Fatalf("root z = %g, want fractional 0.1", root.Values[int(z)])
	}

	// The z <= 0 child (with an extra x >= 1 bound to exercise the
	// node-lower shift in the model-space translation) must be caught by the
	// warm dual's certified exit and pooled.
	infBounds := []boundConstraint{
		{varIndex: int(z), sense: LessOrEqual, rhs: 0},
		{varIndex: 0, sense: GreaterOrEqual, rhs: 1},
	}
	inf, infResp := cache.solveNode(ctx, infBounds, opts, rootResp.warm)
	if inf.Status != Infeasible {
		t.Fatalf("z<=0 child status = %v, want infeasible", inf.Status)
	}
	if !infResp.infeasibilityCertified || infResp.farkasProof == nil {
		t.Fatalf("z<=0 child not certified with a proof (certified=%v proof=%v); the test model no longer exercises the warm dual exit",
			infResp.infeasibilityCertified, infResp.farkasProof)
	}
	if cache.dualProofs.stats.Pooled != 1 {
		t.Fatalf("pooled = %d, want 1", cache.dualProofs.stats.Pooled)
	}

	// A later node with the same z <= 0 box (different other bounds) must be
	// pruned by the pool before any LP work.
	lpSolvesBefore := cache.dualProofs.stats.Prunes
	pruned, prunedResp := cache.solveNode(ctx, []boundConstraint{{varIndex: int(z), sense: LessOrEqual, rhs: 0}}, opts, nil)
	if pruned.Status != Infeasible || !prunedResp.infeasibilityCertified {
		t.Fatalf("pool prune failed: status=%v certified=%v", pruned.Status, prunedResp.infeasibilityCertified)
	}
	if cache.dualProofs.stats.Prunes != lpSolvesBefore+1 {
		t.Fatalf("prunes = %d, want %d", cache.dualProofs.stats.Prunes, lpSolvesBefore+1)
	}
	if prunedResp.revised.LPSolves != 0 {
		t.Fatalf("pool prune still solved an LP (%d solves)", prunedResp.revised.LPSolves)
	}

	// The feasible z >= 1 sibling must NOT be pruned and must solve to its
	// true optimum (x+y = 10.2-1 = 9.2, cost 12.2).
	feas, _ := cache.solveNode(ctx, []boundConstraint{{varIndex: int(z), sense: GreaterOrEqual, rhs: 1}}, opts, nil)
	if feas.Status != Optimal {
		t.Fatalf("z>=1 child status = %v, want optimal", feas.Status)
	}
	if math.Abs(feas.Objective-12.2) > 1e-6 {
		t.Fatalf("z>=1 child objective = %g, want 12.2", feas.Objective)
	}
}

// TestDualProofsEndToEndSound solves MIPs with and without the pool and
// requires identical statuses and objectives.
func TestDualProofsEndToEndSound(t *testing.T) {
	builders := []func(t *testing.T) *Model{
		func(t *testing.T) *Model { m, _, _, _ := dualProofTestModel(t); return m },
		func(t *testing.T) *Model {
			// Small set-partitioning-ish model with equality rows: exercises
			// certified exits on both branching directions.
			m := NewModel(Maximize)
			a := m.AddVar(VarSpec{Name: "a", Type: Binary})
			b := m.AddVar(VarSpec{Name: "b", Type: Binary})
			c := m.AddVar(VarSpec{Name: "c", Type: Binary})
			d := m.AddVar(VarSpec{Name: "d", Type: Binary})
			must(t, m.SetObjective(map[Var]float64{a: 5, b: 4, c: 3, d: 6}))
			must(t, m.AddConstraint(map[Var]float64{a: 1, b: 1, c: 1}, Equal, 2))
			must(t, m.AddConstraint(map[Var]float64{b: 1, c: 1, d: 1}, Equal, 2))
			must(t, m.AddConstraint(map[Var]float64{a: 3, b: 2, c: 4, d: 1}, LessOrEqual, 7))
			return m
		},
	}
	for i, build := range builders {
		base := build(t)
		off, err := base.Solve(context.Background(), Options{LPBackend: LPBackendRevised})
		if err != nil {
			t.Fatalf("model %d control solve: %v", i, err)
		}
		on, err := build(t).Solve(context.Background(), Options{LPBackend: LPBackendRevised, UseDualProofs: true})
		if err != nil {
			t.Fatalf("model %d dual-proof solve: %v", i, err)
		}
		if on.Status != off.Status {
			t.Fatalf("model %d status: on=%v off=%v", i, on.Status, off.Status)
		}
		if off.Status == Optimal && math.Abs(on.Objective-off.Objective) > 1e-6 {
			t.Fatalf("model %d objective: on=%g off=%g", i, on.Objective, off.Objective)
		}
	}
}
