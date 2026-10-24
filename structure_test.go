package qopt

import (
	"math"
	"testing"
)

func TestDetectBinaryPackingConflicts(t *testing.T) {
	m := NewModel(Maximize)
	a := m.AddVar(VarSpec{Type: Binary})
	b := m.AddVar(VarSpec{Type: Binary})
	c := m.AddVar(VarSpec{Type: Binary})
	cont := m.AddVar(VarSpec{Type: Continuous})
	must(t, m.AddConstraint(map[Var]float64{a: 2, b: 2, c: 1}, LessOrEqual, 2))
	must(t, m.AddConstraint(map[Var]float64{a: 2, cont: 1}, LessOrEqual, 2))
	cuts := detectBinaryRowConflicts(m)
	if len(cuts) != 2 {
		t.Fatalf("got %d cuts, want 2", len(cuts))
	}
	for _, c := range cuts {
		if c.Sense != LessOrEqual || c.RHS != 1 || len(c.Coeffs) != 2 {
			t.Fatalf("invalid conflict cut: %+v", c)
		}
	}
}

func TestDetectBinaryPackingConflictsSkipsRedundantSetPackingExpansion(t *testing.T) {
	m := NewModel(Maximize)
	coeffs := make(map[Var]float64)
	for i := 0; i < 20; i++ {
		coeffs[m.AddVar(VarSpec{Type: Binary})] = 1
	}
	must(t, m.AddConstraint(coeffs, LessOrEqual, 1))
	if cuts := detectBinaryRowConflicts(m); len(cuts) != 0 {
		t.Fatalf("explicit set-packing row expanded into %d redundant cuts", len(cuts))
	}
}

func TestDetectBinaryPackingConflictsStrengthensLP(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Type: Binary})
	y := m.AddVar(VarSpec{Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 2, y: 1}, LessOrEqual, 2))

	opts := normalizeOptions(Options{})
	base, _, _, err := solveLPModelWithBackend(t.Context(), m, nil, opts, nil)
	if err != nil {
		t.Fatalf("base LP: %v", err)
	}
	cuts := detectBinaryRowConflicts(m)
	if len(cuts) != 1 {
		t.Fatalf("got %d cuts, want 1", len(cuts))
	}
	strengthened := cloneModel(m)
	must(t, strengthened.AddNamedConstraint(cuts[0].Name, cuts[0].Coeffs, cuts[0].Sense, cuts[0].RHS))
	withCut, _, _, err := solveLPModelWithBackend(t.Context(), strengthened, nil, opts, nil)
	if err != nil {
		t.Fatalf("strengthened LP: %v", err)
	}
	if base.Objective <= withCut.Objective+0.49 {
		t.Fatalf("LP bound was not materially strengthened: base=%g cut=%g", base.Objective, withCut.Objective)
	}
}

func TestDetectBinaryPackingConflictsRejectsUnsafeRows(t *testing.T) {
	m := NewModel(Maximize)
	a := m.AddVar(VarSpec{Type: Binary})
	b := m.AddVar(VarSpec{Type: Binary})
	y := m.AddVar(VarSpec{Type: Integer})
	must(t, m.AddConstraint(map[Var]float64{a: 2, b: 2, y: 1}, LessOrEqual, 2))
	must(t, m.AddConstraint(map[Var]float64{a: 2, b: 2}, GreaterOrEqual, 2))
	if cuts := detectBinaryRowConflicts(m); len(cuts) != 0 {
		t.Fatalf("unsafe rows produced cuts: %+v", cuts)
	}
}

func TestDetectBinaryRowConflictsMixedGreaterEqual(t *testing.T) {
	m := NewModel(Maximize)
	a := m.AddVar(VarSpec{Type: Binary})
	b := m.AddVar(VarSpec{Type: Binary})
	y := m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0), Upper: Float64(2)})
	must(t, m.AddConstraint(map[Var]float64{y: 1, a: -2, b: -1}, GreaterOrEqual, 0))
	cuts := detectBinaryRowConflicts(m)
	if len(cuts) != 1 || cuts[0].Coeffs[a] != 1 || cuts[0].Coeffs[b] != 1 {
		t.Fatalf("mixed >= row did not produce expected conflict: %+v", cuts)
	}
}

func TestDetectKnapsackCoverCutsMixedAndComplemented(t *testing.T) {
	t.Run("mixed-greater-equal", func(t *testing.T) {
		m := NewModel(Maximize)
		a := m.AddVar(VarSpec{Type: Binary})
		b := m.AddVar(VarSpec{Type: Binary})
		y := m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0), Upper: Float64(2)})
		must(t, m.AddConstraint(map[Var]float64{y: 1, a: -2, b: -1}, GreaterOrEqual, 0))
		cuts := detectKnapsackCoverCuts(m)
		if len(cuts) != 1 || cuts[0].Coeffs[a] != 1 || cuts[0].Coeffs[b] != 1 || cuts[0].RHS != 1 {
			t.Fatalf("unexpected mixed cover: %+v", cuts)
		}
	})

	t.Run("complemented-literals", func(t *testing.T) {
		m := NewModel(Minimize)
		x := m.AddVar(VarSpec{Type: Binary})
		y := m.AddVar(VarSpec{Type: Binary})
		z := m.AddVar(VarSpec{Type: Binary})
		must(t, m.AddConstraint(map[Var]float64{x: -3, y: -2, z: -2}, LessOrEqual, -4))
		cuts := detectKnapsackCoverCuts(m)
		if len(cuts) != 1 || cuts[0].Sense != LessOrEqual || cuts[0].RHS != -1 {
			t.Fatalf("unexpected complemented cover: %+v", cuts)
		}
		neg := 0
		for _, a := range cuts[0].Coeffs {
			if a == -1 {
				neg++
			}
		}
		if neg != 2 {
			t.Fatalf("complemented cover has %d negative literals, want 2: %+v", neg, cuts[0])
		}
	})
}

func TestDetectKnapsackCoverCutsSkipsLPRedundant(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Type: Binary})
	y := m.AddVar(VarSpec{Type: Binary})
	z := m.AddVar(VarSpec{Type: Binary})
	must(t, m.AddConstraint(map[Var]float64{x: 2, y: 2, z: 2}, LessOrEqual, 4))
	if cuts := detectKnapsackCoverCuts(m); len(cuts) != 0 {
		t.Fatalf("LP-redundant cover was emitted: %+v", cuts)
	}
}

func TestDetectSingleBinaryBigMTightenings(t *testing.T) {
	t.Run("negative-coefficient", func(t *testing.T) {
		m := NewModel(Maximize)
		x := m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0), Upper: Float64(10)})
		z := m.AddVar(VarSpec{Type: Binary})
		must(t, m.AddConstraint(map[Var]float64{x: 1, z: -100}, LessOrEqual, 0))
		got := detectSingleBinaryBigMTightenings(m)
		if len(got) != 1 || got[0].cut.Coeffs[x] != 1 || got[0].cut.Coeffs[z] != -10 || got[0].cut.RHS != 0 {
			t.Fatalf("unexpected tightening: %+v", got)
		}
	})

	t.Run("positive-complement", func(t *testing.T) {
		m := NewModel(Maximize)
		x := m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0), Upper: Float64(10)})
		z := m.AddVar(VarSpec{Type: Binary})
		must(t, m.AddConstraint(map[Var]float64{x: 1, z: 100}, LessOrEqual, 100))
		got := detectSingleBinaryBigMTightenings(m)
		if len(got) != 1 || got[0].cut.Coeffs[x] != 1 || got[0].cut.Coeffs[z] != 10 || got[0].cut.RHS != 10 {
			t.Fatalf("unexpected complemented tightening: %+v", got)
		}
	})

	t.Run("greater-equal", func(t *testing.T) {
		m := NewModel(Minimize)
		x := m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0), Upper: Float64(10)})
		z := m.AddVar(VarSpec{Type: Binary})
		must(t, m.AddConstraint(map[Var]float64{x: -1, z: 100}, GreaterOrEqual, 0))
		got := detectSingleBinaryBigMTightenings(m)
		if len(got) != 1 || got[0].cut.Coeffs[x] != 1 || got[0].cut.Coeffs[z] != -10 || got[0].cut.RHS != 0 {
			t.Fatalf("unexpected >= tightening: %+v", got)
		}
	})
}

func TestSingleBinaryBigMTighteningStrengthensLPAndPreservesMIP(t *testing.T) {
	build := func() *Model {
		m := NewModel(Maximize)
		x := m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0), Upper: Float64(10)})
		z := m.AddVar(VarSpec{Type: Binary})
		must(t, m.SetObjective(map[Var]float64{x: 1, z: -5}))
		must(t, m.AddConstraint(map[Var]float64{x: 1, z: -100}, LessOrEqual, 0))
		return m
	}

	baseModel := build()
	baseLP, _, _, err := solveLPModelWithBackend(t.Context(), baseModel, nil, normalizeOptions(Options{}), nil)
	if err != nil {
		t.Fatal(err)
	}
	tightModel := cloneModel(baseModel)
	for _, tightening := range detectSingleBinaryBigMTightenings(tightModel) {
		tightModel.constraints[tightening.row] = tightening.cut
	}
	tightModel.rowsSorted = nil
	tightLP, _, _, err := solveLPModelWithBackend(t.Context(), tightModel, nil, normalizeOptions(Options{}), nil)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(baseLP.Objective-9.5) > 1e-8 || math.Abs(tightLP.Objective-5) > 1e-8 {
		t.Fatalf("unexpected LP bounds: base=%g tightened=%g", baseLP.Objective, tightLP.Objective)
	}

	baseMIP, err := build().Solve(t.Context(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	tightMIP, err := build().Solve(t.Context(), Options{DetectStructure: true})
	if err != nil {
		t.Fatal(err)
	}
	if baseMIP.Status != Optimal || tightMIP.Status != Optimal || math.Abs(baseMIP.Objective-tightMIP.Objective) > 1e-8 {
		t.Fatalf("MIP changed: base=%+v tightened=%+v", baseMIP, tightMIP)
	}
	if baseMIP.Stats.Structural.RootCleanSolves != 0 {
		t.Fatalf("unstrengthened solve ran root clean LP %d times, want 0", baseMIP.Stats.Structural.RootCleanSolves)
	}
	if tightMIP.Stats.Structural.RootCleanSolves != 1 {
		t.Fatalf("strengthened solve ran root clean LP %d times, want exactly 1", tightMIP.Stats.Structural.RootCleanSolves)
	}
	if tightMIP.Stats.Structural.RootCleanElapsed <= 0 {
		t.Fatal("strengthened solve did not record root clean LP elapsed time")
	}
}

func TestDetectSingleBinaryBigMSkipsWeakTightening(t *testing.T) {
	// The ic97_potential shape: v1 - v2 + 60*z >= 0 with v ∈ [0,59] admits
	// only the marginal 60→59 coefficient cap (1.7% of the branch limit).
	// The strength gate must skip it — replacing the row perturbs the search
	// trajectory for a negligible LP-bound gain.
	m := NewModel(Minimize)
	v1 := m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0), Upper: Float64(59)})
	v2 := m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0), Upper: Float64(59)})
	z := m.AddVar(VarSpec{Type: Binary})
	must(t, m.AddConstraint(map[Var]float64{v1: 1, v2: -1, z: 60}, GreaterOrEqual, 0))
	if got := detectSingleBinaryBigMTightenings(m); len(got) != 0 {
		t.Fatalf("weak tightening was not gated: %+v", got)
	}
}

func TestDetectSingleBinaryBigMIC97Gated(t *testing.T) {
	p := loadProblem(t, "bench/problems/miplib/ic97_potential.json")
	m := modelFromProblem(t, p)
	if got := len(detectSingleBinaryBigMTightenings(m)); got != 0 {
		t.Fatalf("ic97_potential: got %d big-M tightenings, want 0 (all below the strength gate)", got)
	}
}

func TestDetectSingleBinaryBigMGlass4(t *testing.T) {
	p := loadProblem(t, "bench/problems/miplib/glass4.json")
	m := modelFromProblem(t, p)
	if got := len(detectSingleBinaryBigMTightenings(m)); got != 48 {
		t.Fatalf("glass4: got %d explicit-bound big-M tightenings, want 48", got)
	}
}

func TestDetectPartitioningOrbitopes(t *testing.T) {
	m := NewModel(Minimize)
	var x [3][3]Var
	for i := range x {
		row := make(map[Var]float64)
		for j := range x[i] {
			x[i][j] = m.AddVar(VarSpec{Type: Binary})
			row[x[i][j]] = 1
		}
		must(t, m.AddConstraint(row, Equal, 1))
	}
	for j := 0; j < 3; j++ {
		col := make(map[Var]float64)
		for i := 0; i < 3; i++ {
			col[x[i][j]] = 1
		}
		must(t, m.AddConstraint(col, Equal, 1))
	}
	cuts := detectPartitioningOrbitopes(m)
	if len(cuts) != 3 {
		t.Fatalf("got %d orbitope fixings, want 3: %+v", len(cuts), cuts)
	}
	want := map[Var]bool{x[0][1]: true, x[0][2]: true, x[1][2]: true}
	for _, c := range cuts {
		if c.Sense != LessOrEqual || c.RHS != 0 || len(c.Coeffs) != 1 {
			t.Fatalf("invalid orbitope fixing: %+v", c)
		}
		for v := range c.Coeffs {
			if !want[v] {
				t.Fatalf("unexpected fixed variable %d", v)
			}
			delete(want, v)
		}
	}
	if len(want) != 0 {
		t.Fatalf("missing orbitope fixings: %v", want)
	}
}

func TestDetectPartitioningOrbitopesAssign158(t *testing.T) {
	p := loadProblem(t, "bench/problems/miplib/assign1-5-8.json")
	m := modelFromProblem(t, p)
	cuts := detectPartitioningOrbitopes(m)
	if len(cuts) != 6 {
		t.Fatalf("assign1-5-8: got %d orbitope fixings, want 6", len(cuts))
	}
}

func TestCloneModelOwnsVariableBounds(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Type: Binary})
	clone := cloneModel(m)
	zero := 0.0
	clone.vars[x].Upper = &zero
	if *m.vars[x].Upper != 1 {
		t.Fatalf("changing clone bound mutated source: source upper=%g", *m.vars[x].Upper)
	}
}
