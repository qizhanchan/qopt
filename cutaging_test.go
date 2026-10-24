package qopt

import (
	"context"
	"math"
	"testing"
)

func TestCutIsBinding(t *testing.T) {
	c := Constraint{Coeffs: map[Var]float64{Var(0): 1, Var(1): 1}, Sense: LessOrEqual, RHS: 5}
	if !cutIsBinding(c, []float64{2, 3}, 1e-6) {
		t.Fatal("2+3=5 should be binding against RHS=5")
	}
	if cutIsBinding(c, []float64{1, 1}, 1e-6) {
		t.Fatal("1+1=2 should not be binding against RHS=5")
	}

	ge := Constraint{Coeffs: map[Var]float64{Var(0): 1}, Sense: GreaterOrEqual, RHS: 4}
	if !cutIsBinding(ge, []float64{4}, 1e-6) {
		t.Fatal("x=4 should be binding against x>=4")
	}
	if cutIsBinding(ge, []float64{10}, 1e-6) {
		t.Fatal("x=10 should not be binding against x>=4")
	}

	eq := Constraint{Coeffs: map[Var]float64{Var(0): 1}, Sense: Equal, RHS: 3}
	if !cutIsBinding(eq, []float64{3}, 1e-6) {
		t.Fatal("x=3 should be binding against x==3")
	}
}

func TestRelaxAgedCutsKeepsStableSlots(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Upper: Float64(10)})
	y := m.AddVar(VarSpec{Name: "y", Lower: Float64(-2), Upper: Float64(4)})
	cuts := []Constraint{
		{Name: "active", Coeffs: map[Var]float64{x: 1}, Sense: GreaterOrEqual, RHS: 3},
		{Name: "aged_ge", Coeffs: map[Var]float64{x: 2, y: -1}, Sense: GreaterOrEqual, RHS: 5},
		{Name: "aged_le", Coeffs: map[Var]float64{x: -1, y: 3}, Sense: LessOrEqual, RHS: 7},
	}

	if got := relaxAgedCuts(m, cuts, nil, 5); &got[0] != &cuts[0] {
		t.Fatal("no age data should return the original row slice")
	}
	got := relaxAgedCuts(m, cuts, []int{0, 10, 10}, 5)
	if len(got) != len(cuts) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(cuts))
	}
	if got[0].RHS != cuts[0].RHS || got[0].Sense != cuts[0].Sense {
		t.Fatalf("active row changed: got %+v want %+v", got[0], cuts[0])
	}
	for i := 1; i < len(cuts); i++ {
		if got[i].Sense != cuts[i].Sense || got[i].RHS != 0 || len(got[i].Coeffs) != 0 {
			t.Fatalf("aged row %d = %+v, want empty same-sense slot", i, got[i])
		}
	}
}

func TestAgedCutSlotReusesWarmBasis(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Upper: Float64(10)})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	cache := buildCanonCache(m)
	structure := []Constraint{{Name: "cut", Coeffs: map[Var]float64{x: 1}, Sense: GreaterOrEqual, RHS: 5}}
	activeModel := modelWithExtraConstraints(m, structure)
	opts := normalizeOptions(Options{LPBackend: LPBackendRevised})
	active, activeResp, err := solveNodeOrFullWithStructure(
		context.Background(), cache, activeModel, structure, structure, nil, opts, nil,
	)
	if err != nil || active.Status != Optimal {
		t.Fatalf("active solve: status=%s err=%v", active.Status, err)
	}
	if math.Abs(active.Objective-5) > 1e-6 {
		t.Fatalf("active objective = %g, want 5", active.Objective)
	}

	relaxed := relaxAgedCuts(m, structure, []int{2}, 1)
	relaxedModel := modelWithExtraConstraints(m, relaxed)
	loose, looseResp, err := solveNodeOrFullWithStructure(
		context.Background(), cache, relaxedModel, relaxed, structure, nil, opts, activeResp.warm,
	)
	if err != nil || loose.Status != Optimal {
		t.Fatalf("relaxed solve: status=%s err=%v", loose.Status, err)
	}
	if math.Abs(loose.Objective) > 1e-6 {
		t.Fatalf("relaxed objective = %g, want 0", loose.Objective)
	}
	if looseResp.revised.WarmReopts == 0 || looseResp.revised.WarmRejected != 0 {
		t.Fatalf("empty-slot deactivation did not reuse warm basis: %+v", looseResp.revised)
	}

	reactivatedModel := modelWithExtraConstraints(m, structure)
	reactivated, reactivatedResp, err := solveNodeOrFullWithStructure(
		context.Background(), cache, reactivatedModel, structure, structure, nil, opts, looseResp.warm,
	)
	if err != nil || reactivated.Status != Optimal {
		t.Fatalf("reactivated solve: status=%s err=%v", reactivated.Status, err)
	}
	if math.Abs(reactivated.Objective-5) > 1e-6 {
		t.Fatalf("reactivated objective = %g, want 5", reactivated.Objective)
	}
	if reactivatedResp.revised.WarmReopts == 0 || reactivatedResp.revised.WarmRejected != 0 {
		t.Fatalf("slot reactivation did not reuse warm basis: %+v", reactivatedResp.revised)
	}
}

func TestRootCutSlotScalingMatchesBakedRootLP(t *testing.T) {
	base := NewModel(Maximize)
	x := base.AddVar(VarSpec{Name: "x", Upper: Float64(10)})
	y := base.AddVar(VarSpec{Name: "y", Upper: Float64(10)})
	must(t, base.SetObjective(map[Var]float64{x: 3, y: 2}))
	must(t, base.AddConstraint(map[Var]float64{x: 2, y: 1}, LessOrEqual, 12))
	cut := Constraint{Name: "root_cut", Coeffs: map[Var]float64{x: 1, y: 4}, Sense: LessOrEqual, RHS: 16}
	full := modelWithExtraConstraints(base, []Constraint{cut})
	opts := normalizeOptions(Options{LPBackend: LPBackendRevised})

	bakedCache := buildCanonCache(full)
	baked, _ := bakedCache.solveNode(context.Background(), nil, opts, nil)
	dynamicCache := buildCanonCacheScaledLike(base, full)
	dynamicModel := modelWithExtraConstraints(base, []Constraint{cut})
	dynamic, _, err := solveNodeOrFullWithStructure(
		context.Background(), dynamicCache, dynamicModel,
		[]Constraint{cut}, []Constraint{cut}, nil, opts, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if baked.Status != Optimal || dynamic.Status != Optimal {
		t.Fatalf("statuses baked=%s dynamic=%s", baked.Status, dynamic.Status)
	}
	if math.Abs(baked.Objective-dynamic.Objective) > 1e-9 {
		t.Fatalf("objectives baked=%g dynamic=%g", baked.Objective, dynamic.Objective)
	}
	for i := range baked.Values {
		if math.Abs(baked.Values[i]-dynamic.Values[i]) > 1e-9 {
			t.Fatalf("value[%d] baked=%g dynamic=%g", i, baked.Values[i], dynamic.Values[i])
		}
	}
	for i := range dynamicCache.colScale {
		if dynamicCache.colScale[i] != bakedCache.colScale[i] {
			t.Fatalf("column scale[%d] dynamic=%g baked=%g", i, dynamicCache.colScale[i], bakedCache.colScale[i])
		}
	}
	if len(dynamicCache.extraRowScale) != 1 || dynamicCache.extraRowScale[0] != bakedCache.rowScale[len(dynamicCache.rowScale)] {
		t.Fatalf("extra row scales dynamic=%v baked=%v", dynamicCache.extraRowScale, bakedCache.rowScale)
	}
}

func TestUpdateCutAges(t *testing.T) {
	tight := Constraint{Coeffs: map[Var]float64{Var(0): 1}, Sense: LessOrEqual, RHS: 5}
	loose := Constraint{Coeffs: map[Var]float64{Var(0): 1}, Sense: LessOrEqual, RHS: 100}
	extras := []Constraint{tight, loose}

	// First check: tight resets to 0, loose increments from its inherited age.
	ages := updateCutAges(extras, []int{3, 3}, []float64{5}, 1e-6)
	if ages[0] != 0 {
		t.Fatalf("tight cut age = %d, want 0", ages[0])
	}
	if ages[1] != 4 {
		t.Fatalf("loose cut age = %d, want 4", ages[1])
	}

	// Reactivation: a previously-aged cut that becomes tight again resets.
	ages2 := updateCutAges(extras, []int{20, 20}, []float64{5}, 1e-6)
	if ages2[0] != 0 {
		t.Fatalf("reactivated cut age = %d, want 0", ages2[0])
	}

	// Extras longer than the inherited ages slice (freshly-appended cuts):
	// missing entries default to age 0 before being evaluated.
	ages3 := updateCutAges(extras, []int{}, []float64{5}, 1e-6)
	if ages3[0] != 0 || ages3[1] != 1 {
		t.Fatalf("ages3 = %v, want [0, 1]", ages3)
	}
}

// TestCutAgingNeverWorsensOptimum is the correctness backstop: relaxing an
// aged-out cut only ever loosens the LP, so enabling CutRowAgeLimit must
// never exclude a solution that cuts-without-aging (or no cuts at all)
// would have found — on a small MIP where all three configs run to
// completion, they must agree on the objective.
func TestCutAgingNeverWorsensOptimum(t *testing.T) {
	build := func() *Model {
		m := NewModel(Maximize)
		x := m.AddVar(VarSpec{Name: "x", Type: Integer, Upper: Float64(20)})
		y := m.AddVar(VarSpec{Name: "y", Type: Integer, Upper: Float64(20)})
		must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
		must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
		must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))
		return m
	}
	noCuts := solveOK(t, build(), Options{})
	withCuts := solveOK(t, build(), Options{MaxCutRounds: 5, MaxCutDepth: 5})
	withAging := solveOK(t, build(), Options{MaxCutRounds: 5, MaxCutDepth: 5, CutRowAgeLimit: 2})
	withAgingRevised := solveOK(t, build(), Options{MaxCutRounds: 5, MaxCutDepth: 5, CutRowAgeLimit: 2, LPBackend: LPBackendRevised})

	for _, tc := range []struct {
		name string
		res  Result
	}{{"cuts", withCuts}, {"aging", withAging}, {"aging/revised", withAgingRevised}} {
		if math.Abs(tc.res.Objective-noCuts.Objective) > 1e-6 {
			t.Fatalf("%s objective = %.12g, want %.12g (no-cuts baseline)", tc.name, tc.res.Objective, noCuts.Objective)
		}
	}
	if withAgingRevised.Stats == nil || withAgingRevised.Stats.CutLifecycle.RootRows == 0 {
		t.Fatalf("revised root-cut lifecycle telemetry = %+v, want inherited root rows", withAgingRevised.Stats)
	}
	if withAgingRevised.Stats.CutLifecycle.NodeSamples == 0 || withAgingRevised.Stats.CutLifecycle.PeakActiveRows == 0 {
		t.Fatalf("revised root-cut lifecycle samples = %+v", withAgingRevised.Stats.CutLifecycle)
	}
}
