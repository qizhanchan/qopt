package qopt

import (
	"context"
	"math"
	"testing"
)

func TestPseudoObjectiveBound(t *testing.T) {
	t.Run("minimize picks lower for positive coeff", func(t *testing.T) {
		m := NewModel(Minimize)
		x := m.AddVar(VarSpec{Name: "x", Lower: Float64(2), Upper: Float64(9)})
		y := m.AddVar(VarSpec{Name: "y", Lower: Float64(-3), Upper: Float64(5)})
		must(t, m.SetObjective(map[Var]float64{x: 1, y: -2}))
		// x has positive coeff -> pick lower (2); y has negative coeff -> pick upper (5).
		// 1*2 + (-2)*5 = 2 - 10 = -8.
		got := pseudoObjectiveBound(m, nil)
		closeTo(t, got, -8)
	})

	t.Run("maximize picks upper for positive coeff", func(t *testing.T) {
		m := NewModel(Maximize)
		x := m.AddVar(VarSpec{Name: "x", Lower: Float64(2), Upper: Float64(9)})
		y := m.AddVar(VarSpec{Name: "y", Lower: Float64(-3), Upper: Float64(5)})
		must(t, m.SetObjective(map[Var]float64{x: 1, y: -2}))
		// x has positive coeff -> pick upper (9); y has negative coeff -> pick lower (-3).
		// 1*9 + (-2)*(-3) = 9 + 6 = 15.
		got := pseudoObjectiveBound(m, nil)
		closeTo(t, got, 15)
	})

	t.Run("node bounds override var spec bounds", func(t *testing.T) {
		m := NewModel(Minimize)
		x := m.AddVar(VarSpec{Name: "x", Lower: Float64(0), Upper: Float64(9)})
		must(t, m.SetObjective(map[Var]float64{x: 1}))
		got := pseudoObjectiveBound(m, []boundConstraint{{varIndex: int(x), sense: GreaterOrEqual, rhs: 3}})
		closeTo(t, got, 3)
	})

	t.Run("unbounded direction yields uninformative infinity, not NaN", func(t *testing.T) {
		m := NewModel(Minimize)
		x := m.AddVar(VarSpec{Name: "x"}) // default [0, +Inf)
		y := m.AddVar(VarSpec{Name: "y"})
		must(t, m.SetObjective(map[Var]float64{x: 1, y: -1})) // y's coeff wants +Inf upper -> -Inf contribution
		got := pseudoObjectiveBound(m, nil)
		if !math.IsInf(got, -1) {
			t.Fatalf("pseudoObjectiveBound = %v, want -Inf", got)
		}
		// Feeding this into the pruning check must never look like "prune":
		// boundCannotImprove(Minimize, -Inf, anyIncumbent) must be false.
		if boundCannotImprove(Minimize, got, 100, Options{}) {
			t.Fatalf("boundCannotImprove(-Inf, 100) = true, want false (uninformative bound must never prune)")
		}
	})
}

func TestPropagateObjectiveCutoff(t *testing.T) {
	t.Run("min integer bounds", func(t *testing.T) {
		m := NewModel(Minimize)
		x := m.AddVar(VarSpec{Name: "x", Type: Integer, Upper: Float64(10)})
		y := m.AddVar(VarSpec{Name: "y", Type: Integer, Upper: Float64(10)})
		must(t, m.SetObjective(map[Var]float64{x: -3, y: -2}))
		lower, upper := effectiveBounds(m, nil)
		n, infeasible := propagateObjectiveCutoff(m, lower, upper, -45, 1e-8)
		if infeasible || n != 2 {
			t.Fatalf("reductions=%d infeasible=%v, want 2,false", n, infeasible)
		}
		if lower[x] != 9 || lower[y] != 8 {
			t.Fatalf("lower=(%g,%g), want (9,8)", lower[x], lower[y])
		}
	})

	t.Run("single favorable infinity", func(t *testing.T) {
		m := NewModel(Minimize)
		x := m.AddVar(VarSpec{Name: "x", Type: Integer})
		must(t, m.SetObjective(map[Var]float64{x: -2}))
		lower, upper := effectiveBounds(m, nil)
		n, infeasible := propagateObjectiveCutoff(m, lower, upper, -9, 1e-8)
		if infeasible || n != 1 || lower[x] != 5 || !math.IsInf(upper[x], 1) {
			t.Fatalf("reductions=%d infeasible=%v bounds=[%g,%g], want 1,false,[5,+Inf]", n, infeasible, lower[x], upper[x])
		}
	})

	t.Run("box proves infeasible", func(t *testing.T) {
		m := NewModel(Minimize)
		x := m.AddVar(VarSpec{Name: "x", Type: Integer, Upper: Float64(10)})
		must(t, m.SetObjective(map[Var]float64{x: -5}))
		lower, upper := effectiveBounds(m, nil)
		_, infeasible := propagateObjectiveCutoff(m, lower, upper, -51, 1e-8)
		if !infeasible {
			t.Fatal("cutoff below box optimum should prove infeasible")
		}
	})

	t.Run("maximize direction", func(t *testing.T) {
		m := NewModel(Maximize)
		x := m.AddVar(VarSpec{Name: "x", Type: Integer, Upper: Float64(10)})
		y := m.AddVar(VarSpec{Name: "y", Type: Integer, Upper: Float64(10)})
		must(t, m.SetObjective(map[Var]float64{x: 3, y: 2}))
		lower, upper := effectiveBounds(m, nil)
		n, infeasible := propagateObjectiveCutoff(m, lower, upper, 45, 1e-8)
		if infeasible || n != 2 || lower[x] != 9 || lower[y] != 8 {
			t.Fatalf("reductions=%d infeasible=%v lower=(%g,%g), want 2,false,(9,8)", n, infeasible, lower[x], lower[y])
		}
	})
}

func TestObjectiveCutoffPropagationPreservesMIPOptimum(t *testing.T) {
	build := func() *Model {
		m := NewModel(Minimize)
		x := m.AddVar(VarSpec{Name: "x", Type: Integer, Upper: Float64(10)})
		y := m.AddVar(VarSpec{Name: "y", Type: Integer, Upper: Float64(10)})
		must(t, m.SetObjective(map[Var]float64{x: -3, y: -2}))
		must(t, m.AddConstraint(map[Var]float64{x: 2, y: 1}, LessOrEqual, 13))
		must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 14))
		return m
	}
	baseline := solveOK(t, build(), Options{MIPBranching: MIPBranchReliability})
	got := solveOK(t, build(), Options{
		MIPBranching:               MIPBranchReliability,
		ObjectiveCutoffPropagation: true,
	})
	if math.Abs(got.Objective-baseline.Objective) > 1e-9 || got.Status != baseline.Status {
		t.Fatalf("baseline=(%s,%g) cutoff=(%s,%g)", baseline.Status, baseline.Objective, got.Status, got.Objective)
	}
}

// TestReducedCostBoundsDerivesExpectedTightening hand-verifies the derived
// bound against an analytically known LP optimum. Minimize x + 2y s.t.
// x+y>=4, x,y in [0,10]. The optimum puts all weight on the cheaper
// variable: x=4, y=0 (obj=4), with y nonbasic at its lower bound and
// reduced cost d(y) = 2 - 1*1 = 1 (dual on the row is 1, since x is basic
// with cost 1). With an incumbent bestObj=5, gap=1, the derived upper bound
// on y is l_y + gap/d(y) = 0 + 1/1 = 1 — verified independently: fixing
// y=1 forces x>=3, giving obj=3+2=5, exactly tying (not beating) the
// incumbent, so nothing strictly better than the incumbent has y>1.
func TestReducedCostBoundsDerivesExpectedTightening(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Upper: Float64(10)})
	y := m.AddVar(VarSpec{Name: "y", Upper: Float64(10)})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 2}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, GreaterOrEqual, 4))

	res, resp, _, err := solveLPModelWithBackend(context.Background(), m, nil, normalizeOptions(Options{LPBackend: LPBackendRevised}), nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal {
		t.Fatalf("status = %s", res.Status)
	}
	closeTo(t, res.Objective, 4)
	if resp.revisedState == nil {
		t.Fatal("revisedState is nil")
	}

	bounds := reducedCostBounds(m, resp.revisedState, Minimize, res.Objective, 5, Options{FeasibilityTol: 1e-8})
	found := false
	for _, b := range bounds {
		if b.varIndex == int(y) {
			found = true
			if b.sense != LessOrEqual {
				t.Fatalf("y bound sense = %v, want LessOrEqual", b.sense)
			}
			closeTo(t, b.rhs, 1)
		}
	}
	if !found {
		t.Fatalf("no bound derived for y; bounds=%v", bounds)
	}
}

// TestReducedCostBoundsNoIncumbentGap returns nil when the LP bound already
// meets/exceeds the incumbent — that case is boundCannotImprove's job, not
// this function's; asking it to still derive a bound would be operating on
// a meaningless (non-positive) gap.
func TestReducedCostBoundsNoIncumbentGap(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Upper: Float64(10)})
	y := m.AddVar(VarSpec{Name: "y", Upper: Float64(10)})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 2}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, GreaterOrEqual, 4))

	res, resp, _, err := solveLPModelWithBackend(context.Background(), m, nil, normalizeOptions(Options{LPBackend: LPBackendRevised}), nil)
	if err != nil {
		t.Fatal(err)
	}
	// bestObj == lpObj: zero gap, nothing to derive.
	if b := reducedCostBounds(m, resp.revisedState, Minimize, res.Objective, res.Objective, Options{FeasibilityTol: 1e-8}); b != nil {
		t.Fatalf("expected nil bounds at zero gap, got %v", b)
	}
	// nil revisedState: always nil, regardless of gap.
	if b := reducedCostBounds(m, nil, Minimize, res.Objective, 1000, Options{}); b != nil {
		t.Fatalf("expected nil bounds with nil revisedState, got %v", b)
	}
}

// TestPseudoObjAndReducedCostNeverWorsenOptimum is the correctness backstop:
// on a battery of small MIPs with known optima, turning either or both new
// propagators on (on both LP backends, and paired with the LP-bound-tightest
// branching mode) must never move the objective away from the no-propagation
// baseline. Any bug that wrongly excludes the true optimum would show up
// here as a worse objective.
func TestPseudoObjAndReducedCostNeverWorsenOptimum(t *testing.T) {
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
		{"min-cover", func() *Model {
			m := NewModel(Minimize)
			x := m.AddVar(VarSpec{Name: "x", Type: Integer, Upper: Float64(10)})
			y := m.AddVar(VarSpec{Name: "y", Type: Integer, Upper: Float64(10)})
			must(t, m.SetObjective(map[Var]float64{x: 1, y: 2}))
			must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, GreaterOrEqual, 4))
			must(t, m.AddConstraint(map[Var]float64{x: 3, y: 1}, GreaterOrEqual, 6))
			return m
		}},
	}

	variants := []struct {
		name string
		opts Options
	}{
		{"pseudoObj/tableau", Options{PseudoObjPropagation: true}},
		{"pseudoObj/revised", Options{PseudoObjPropagation: true, LPBackend: LPBackendRevised}},
		{"redcost/revised", Options{ReducedCostFixing: true, LPBackend: LPBackendRevised}},
		{"both/revised", Options{PseudoObjPropagation: true, ReducedCostFixing: true, LPBackend: LPBackendRevised}},
		{"both/revised/bestbound", Options{PseudoObjPropagation: true, ReducedCostFixing: true, LPBackend: LPBackendRevised, MIPNodeSelection: MIPBestBound}},
		{"both/revised/reliability", Options{PseudoObjPropagation: true, ReducedCostFixing: true, LPBackend: LPBackendRevised, MIPBranching: MIPBranchReliability}},
	}

	for _, tc := range cases {
		baseline := solveOK(t, tc.build(), Options{})
		for _, v := range variants {
			t.Run(tc.name+"/"+v.name, func(t *testing.T) {
				got := solveOK(t, tc.build(), v.opts)
				if math.Abs(got.Objective-baseline.Objective) > 1e-6 {
					t.Fatalf("objective baseline=%.12g %s=%.12g", baseline.Objective, v.name, got.Objective)
				}
			})
		}
	}
}
