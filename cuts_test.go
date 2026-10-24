package qopt

import (
	"context"
	"math"
	"testing"
)

// TestCuts_OptionDefaultOff confirms the cut option is gated and that the
// default behavior (MaxCutRounds = 0) is unchanged from the no-cuts world.
func TestCuts_OptionDefaultOff(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
	must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))

	res, err := m.Solve(context.Background(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal || math.Abs(res.Objective-20) > 1e-9 {
		t.Fatalf("optimum mismatched: status=%s obj=%v", res.Status, res.Objective)
	}
}

// TestCuts_BinaryKnapsack runs a small binary problem with cuts ON and OFF
// and checks both reach the same optimum. The point isn't to demonstrate a
// speedup on a 3-variable problem — it's to detect derivation bugs that would
// cut off the feasible region.
func TestCuts_BinaryKnapsack(t *testing.T) {
	build := func() *Model {
		m := NewModel(Maximize)
		x := m.AddVar(VarSpec{Name: "x", Type: Binary})
		y := m.AddVar(VarSpec{Name: "y", Type: Binary})
		z := m.AddVar(VarSpec{Name: "z", Type: Binary})
		must(t, m.SetObjective(map[Var]float64{x: 6, y: 5, z: 8}))
		must(t, m.AddConstraint(map[Var]float64{x: 4, y: 3, z: 5}, LessOrEqual, 8))
		return m
	}
	off, errOff := build().Solve(context.Background(), Options{})
	on, errOn := build().Solve(context.Background(), Options{MaxCutRounds: 5})
	if errOff != nil || errOn != nil {
		t.Fatalf("solve errors: %v / %v", errOff, errOn)
	}
	if off.Status != Optimal || on.Status != Optimal {
		t.Fatalf("non-optimal: off=%s on=%s", off.Status, on.Status)
	}
	if math.Abs(off.Objective-on.Objective) > 1e-9 {
		t.Fatalf("objectives diverged with cuts: off=%v on=%v", off.Objective, on.Objective)
	}
}

// TestCuts_FractionalRootRelaxation builds a problem whose root LP relaxation
// is fractional. With cuts on, we should still reach the proven integer
// optimum (and not, for example, cut it off through a bad derivation).
func TestCuts_FractionalRootRelaxation(t *testing.T) {
	// max  x + y     s.t.  2x + 2y ≤ 3,  x, y ∈ {0,1}
	// LP relaxation optimum: x=y=0.75 → obj = 1.5
	// MIP optimum: pick exactly one of x or y → obj = 1
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 2, y: 2}, LessOrEqual, 3))

	res, err := m.Solve(context.Background(), Options{MaxCutRounds: 5})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal {
		t.Fatalf("status = %s, want optimal", res.Status)
	}
	if math.Abs(res.Objective-1) > 1e-9 {
		t.Fatalf("objective %v, want 1", res.Objective)
	}
}

// TestCuts_GlobalRootCutsDoNotResolveCleanLP guards the tree-integration
// behavior measured on timtab1. Global root cuts are part of the fixed work
// relaxation and should directly provide the branching LP; solving the
// original no-cut relaxation again at every node discards their tree-steering
// effect and doubles the node LP path. Root-cut validation itself uses one
// cold revised solve before the cut-augmented root solve.
func TestCuts_GlobalRootCutsDoNotResolveCleanLP(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 2, y: 2}, LessOrEqual, 3))

	res, err := m.Solve(context.Background(), Options{
		LPBackend:               LPBackendRevised,
		MaxCutRounds:            1,
		MaxNodes:                1,
		DisablePrimalHeuristics: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Cuts == 0 {
		t.Fatal("expected at least one global root cut")
	}
	if res.Stats == nil {
		t.Fatal("missing solver stats")
	}
	// Persistent root separation: one base LP, one warm cut-augmented
	// re-solve, one final cold validation, then the search's cut-augmented
	// root solve. The guarded property is that node solves do NOT re-solve
	// the clean no-cut LP on top of that.
	if got := res.Stats.Revised.LPSolves; got != 4 {
		t.Fatalf("revised LP solves = %d, want base + warm re-solve + cold validation + root node solve", got)
	}
}

func TestRootSeparationRoundsSeeAppliedCuts(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 2, y: 2}, LessOrEqual, 3))

	res, err := m.Solve(context.Background(), Options{
		LPBackend:               LPBackendRevised,
		MaxCutRounds:            2,
		MaxNodes:                1,
		DisablePrimalHeuristics: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Stats == nil || len(res.Stats.RootSeparation) < 2 {
		t.Fatalf("root separation stats = %+v, want two attempted rounds", res.Stats)
	}
	first := res.Stats.RootSeparation[0]
	second := res.Stats.RootSeparation[1]
	if first.Applied == 0 {
		t.Fatalf("first round = %+v, want an applied cut", first)
	}
	if second.InputRows != first.InputRows+first.Applied {
		t.Fatalf("round input rows = %d then %d, applied=%d", first.InputRows, second.InputRows, first.Applied)
	}
	if first.Families["gomory"].Applied == 0 {
		t.Fatalf("family stats = %+v, want applied Gomory cut", first.Families)
	}
	if first.PoolSize == 0 || first.ActiveRows != first.Applied {
		t.Fatalf("first round pool stats = %+v, want applied cuts active in the pool", first)
	}
	if first.LPIterations == 0 || first.ValidationIterations == 0 || first.ObjectiveGain <= 0 || first.Elapsed <= 0 {
		t.Fatalf("first round solve telemetry = %+v", first)
	}
	if second.PoolTested < second.PoolCandidates {
		t.Fatalf("second round pool stats = %+v, tested must cover violated candidates", second)
	}
}

func TestCuts_DoNotMutateModel(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 2, y: 2}, LessOrEqual, 3))

	before := len(m.constraints)
	res, err := m.Solve(context.Background(), Options{MaxCutRounds: 5})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal {
		t.Fatalf("status = %s, want optimal", res.Status)
	}
	if len(m.constraints) != before {
		t.Fatalf("Solve with cuts mutated model constraints: before=%d after=%d", before, len(m.constraints))
	}

	again, err := m.Solve(context.Background(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if again.Status != Optimal || math.Abs(again.Objective-1) > 1e-9 {
		t.Fatalf("second solve changed after cuts solve: status=%s obj=%v", again.Status, again.Objective)
	}
}

// TestCuts_PerNodeDepth confirms MaxCutDepth lets cuts derive inside the B&B
// tree without changing the optimum on a small fractional-LP MIP.
func TestCuts_PerNodeDepth(t *testing.T) {
	build := func() *Model {
		// LP relaxation: x=y=z=0.6 (obj 9). MIP optimum: any single bit set
		// alone won't satisfy the 5-coverage budget; pick exactly two of {x,y,z}
		// → obj = 5+3+? actually let me reason in head: x+y+z=2 saturates the
		// constraint, obj = sum of two coefs. Best pair: y+z → 3+4=7. We don't
		// rely on the exact optimum, only that cut-on matches cut-off.
		m := NewModel(Maximize)
		x := m.AddVar(VarSpec{Name: "x", Type: Binary})
		y := m.AddVar(VarSpec{Name: "y", Type: Binary})
		z := m.AddVar(VarSpec{Name: "z", Type: Binary})
		must(t, m.SetObjective(map[Var]float64{x: 5, y: 3, z: 4}))
		must(t, m.AddConstraint(map[Var]float64{x: 5, y: 3, z: 4}, LessOrEqual, 7))
		return m
	}
	off, errOff := build().Solve(context.Background(), Options{})
	on, errOn := build().Solve(context.Background(), Options{MaxCutDepth: 3})
	if errOff != nil || errOn != nil {
		t.Fatalf("solve errors: off=%v on=%v", errOff, errOn)
	}
	if off.Status != Optimal || on.Status != Optimal {
		t.Fatalf("non-optimal: off=%s on=%s", off.Status, on.Status)
	}
	if math.Abs(off.Objective-on.Objective) > 1e-9 {
		t.Fatalf("objectives diverged with per-node cuts: off=%v on=%v", off.Objective, on.Objective)
	}
}

// TestCuts_PerNodeRevisedBackend exercises per-node cuts under the revised
// backend, which produces no simplex state of its own — generatePerNodeCuts
// must re-solve the node on the tableau backend to obtain the tableau and
// reuse gomoryCuts. The cut-on revised result must match the cut-off optimum;
// a divergence would mean the on-demand state path emitted an invalid cut.
func TestCuts_PerNodeRevisedBackend(t *testing.T) {
	build := func() *Model {
		m := NewModel(Maximize)
		x := m.AddVar(VarSpec{Name: "x", Type: Integer})
		y := m.AddVar(VarSpec{Name: "y", Type: Integer})
		must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
		must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
		must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))
		return m
	}
	off, errOff := build().Solve(context.Background(), Options{LPBackend: LPBackendRevised})
	on, errOn := build().Solve(context.Background(), Options{LPBackend: LPBackendRevised, MaxCutDepth: 4})
	if errOff != nil || errOn != nil {
		t.Fatalf("solve errors: off=%v on=%v", errOff, errOn)
	}
	if off.Status != Optimal || on.Status != Optimal {
		t.Fatalf("non-optimal: off=%s on=%s", off.Status, on.Status)
	}
	if math.Abs(off.Objective-on.Objective) > 1e-9 {
		t.Fatalf("per-node cuts on revised backend changed the optimum: off=%v on=%v", off.Objective, on.Objective)
	}
}

// TestCuts_PerNodeNoMutation guards against cut accumulation leaking into the
// caller's Model. Per-node cuts live in the cutBundle chain on each B&B node;
// they must never get AddConstraint'd into the user's model.
func TestCuts_PerNodeNoMutation(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	z := m.AddVar(VarSpec{Name: "z", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 5, y: 3, z: 4}))
	must(t, m.AddConstraint(map[Var]float64{x: 5, y: 3, z: 4}, LessOrEqual, 7))

	before := len(m.constraints)
	res, err := m.Solve(context.Background(), Options{MaxCutDepth: 3})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal {
		t.Fatalf("status = %s, want optimal", res.Status)
	}
	if len(m.constraints) != before {
		t.Fatalf("Solve with per-node cuts mutated model constraints: before=%d after=%d", before, len(m.constraints))
	}
}

// TestCuts_PerNodeDepthStride confirms cut-depth-stride sparsifies firing
// without disturbing the optimum. With stride=5 cuts only fire at depths
// {0,5,10,...}; with depth=3 that's effectively root-only. Optimum must
// still match the cut-off run.
func TestCuts_PerNodeDepthStride(t *testing.T) {
	build := func() *Model {
		m := NewModel(Maximize)
		x := m.AddVar(VarSpec{Name: "x", Type: Binary})
		y := m.AddVar(VarSpec{Name: "y", Type: Binary})
		z := m.AddVar(VarSpec{Name: "z", Type: Binary})
		must(t, m.SetObjective(map[Var]float64{x: 5, y: 3, z: 4}))
		must(t, m.AddConstraint(map[Var]float64{x: 5, y: 3, z: 4}, LessOrEqual, 7))
		return m
	}
	off, _ := build().Solve(context.Background(), Options{})
	sparse, _ := build().Solve(context.Background(), Options{MaxCutDepth: 3, CutDepthStride: 5})
	if math.Abs(off.Objective-sparse.Objective) > 1e-9 {
		t.Fatalf("stride changed optimum: off=%v sparse=%v", off.Objective, sparse.Objective)
	}
}

// TestCuts_ShouldCutAtDepth pins down the predicate so future cadence work
// can't silently drift. Empty/zero stride is treated as stride=1.
func TestCuts_ShouldCutAtDepth(t *testing.T) {
	cases := []struct {
		depth, max, stride int
		want               bool
	}{
		{depth: 0, max: 0, stride: 0, want: false}, // disabled
		{depth: 0, max: 3, stride: 0, want: true},  // stride 0 → 1, root fires
		{depth: 5, max: 3, stride: 0, want: false}, // beyond max
		{depth: 0, max: 10, stride: 5, want: true}, // depth 0 fires for any stride
		{depth: 3, max: 10, stride: 5, want: false},
		{depth: 5, max: 10, stride: 5, want: true},
		{depth: 7, max: 10, stride: 5, want: false},
		{depth: 10, max: 10, stride: 5, want: true},
	}
	for _, c := range cases {
		got := shouldCutAtDepth(c.depth, Options{MaxCutDepth: c.max, CutDepthStride: c.stride}, nil)
		if got != c.want {
			t.Errorf("shouldCutAtDepth(depth=%d max=%d stride=%d) = %v, want %v", c.depth, c.max, c.stride, got, c.want)
		}
	}
}

// TestCuts_AdaptiveCadenceDecays asserts the CBC-style cadence kicks in:
// after cutMissesToDecay consecutive cuts that didn't move the LP objective,
// the effective stride doubles. The predicate then skips depths that would
// have fired under the user-set stride.
func TestCuts_AdaptiveCadenceDecays(t *testing.T) {
	cadence := &cutCadenceState{}
	opts := Options{MaxCutDepth: 100, CutDepthStride: 1}

	// Baseline: stride 1, depth 1 fires.
	if !shouldCutAtDepth(1, opts, cadence) {
		t.Fatalf("baseline: depth 1 should fire under stride 1")
	}

	// cutMissesToDecay-1 misses must NOT trigger decay yet.
	for i := 0; i < cutMissesToDecay-1; i++ {
		cadence.observe(100.0, 100.0)
	}
	if got := cadence.effectiveStrideMultiplier(); got != 1 {
		t.Fatalf("before decay threshold reached, extraStride=%d, want 1", got)
	}

	// One more miss reaches the threshold → stride doubles to 2.
	cadence.observe(100.0, 100.0)
	if got, want := cadence.effectiveStrideMultiplier(), 2; got != want {
		t.Fatalf("after %d misses extraStride=%d, want %d", cutMissesToDecay, got, want)
	}
	if shouldCutAtDepth(1, opts, cadence) {
		t.Fatalf("after decay: depth 1 should NOT fire (effective stride 2)")
	}
	if !shouldCutAtDepth(2, opts, cadence) {
		t.Fatalf("after decay: depth 2 should fire (effective stride 2)")
	}

	// A real improvement resets the miss counter but keeps the multiplier.
	cadence.observe(100.0, 50.0)
	if cadence.misses != 0 {
		t.Fatalf("observed improvement should reset misses, got %d", cadence.misses)
	}
	if got, want := cadence.effectiveStrideMultiplier(), 2; got != want {
		t.Fatalf("multiplier should NOT decay back after improvement, got %d", got)
	}

	// Sustained misses double further, then cap at 1<<maxExtraStrideShift.
	for i := 0; i < cutMissesToDecay*(maxExtraStrideShift+2); i++ {
		cadence.observe(100.0, 100.0)
	}
	if got, max := cadence.effectiveStrideMultiplier(), 1<<maxExtraStrideShift; got != max {
		t.Fatalf("multiplier should cap at %d, got %d", max, got)
	}
}

// TestCuts_PerNodeCombinesWithRootCuts verifies the two cut switches compose:
// global root cuts (MaxCutRounds) layered with subtree-local cuts
// (MaxCutDepth) reach the same optimum as either knob alone.
func TestCuts_PerNodeCombinesWithRootCuts(t *testing.T) {
	build := func() *Model {
		m := NewModel(Maximize)
		x := m.AddVar(VarSpec{Name: "x", Type: Binary})
		y := m.AddVar(VarSpec{Name: "y", Type: Binary})
		z := m.AddVar(VarSpec{Name: "z", Type: Binary})
		w := m.AddVar(VarSpec{Name: "w", Type: Binary})
		must(t, m.SetObjective(map[Var]float64{x: 7, y: 5, z: 4, w: 3}))
		must(t, m.AddConstraint(map[Var]float64{x: 8, y: 5, z: 4, w: 3}, LessOrEqual, 11))
		return m
	}
	baseline, errA := build().Solve(context.Background(), Options{})
	combo, errB := build().Solve(context.Background(), Options{MaxCutRounds: 3, MaxCutDepth: 2})
	if errA != nil || errB != nil {
		t.Fatalf("solve errors: a=%v b=%v", errA, errB)
	}
	if math.Abs(baseline.Objective-combo.Objective) > 1e-9 {
		t.Fatalf("objectives diverged: baseline=%v combo=%v", baseline.Objective, combo.Objective)
	}
}

// TestCuts_MixedIntegerSkipsContinuousRows runs a mixed-integer model. The
// pure-fractional Gomory cut isn't valid for continuous non-basics so the
// generator must SKIP those rows. We're checking we still reach the right
// optimum (i.e. didn't emit an invalid cut).
func TestCuts_MixedIntegerSkipsContinuousRows(t *testing.T) {
	// max  3x + 2y
	//      x  + y  ≤ 5      (mixed: x integer, y continuous)
	//      x       ≤ 3
	//                 x int, y ≥ 0
	// Integer optimum: x=3, y=2 (continuous fills the slack), obj=13
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	y := m.AddVar(VarSpec{Name: "y"})
	must(t, m.SetObjective(map[Var]float64{x: 3, y: 2}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 5))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, LessOrEqual, 3))

	res, err := m.Solve(context.Background(), Options{MaxCutRounds: 5})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal {
		t.Fatalf("status = %s, want optimal", res.Status)
	}
	if math.Abs(res.Objective-13) > 1e-9 {
		t.Fatalf("objective %v, want 13", res.Objective)
	}
}
