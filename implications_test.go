package qopt

import "testing"

func TestDetectVariableBoundRelationsNormalizesBothDirections(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0), Upper: Float64(10)})
	z := m.AddVar(VarSpec{Type: Binary})
	must(t, m.AddConstraint(map[Var]float64{x: 2, z: -6}, LessOrEqual, 4))
	rels := detectVariableBoundRelations(m)
	if len(rels) != 2 {
		t.Fatalf("got %d normalized relations, want 2", len(rels))
	}
	if rels[0].SourceRow != 0 || rels[0].Bounded != int(x) || rels[0].Controller != int(z) ||
		!rels[0].Upper || rels[0].Slope != 3 || rels[0].Intercept != 2 {
		t.Fatalf("unexpected x relation: %+v", rels[0])
	}
}

func TestProbeBinaryImplicationsFindsCommonConsequence(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0), Upper: Float64(10)})
	z := m.AddVar(VarSpec{Type: Binary})
	// z=1 implies x>=1 through the first row; z=0 implies x>=1 through
	// the second. Neither row alone can globally derive x>=1.
	must(t, m.AddConstraint(map[Var]float64{x: 1, z: -1}, GreaterOrEqual, 0))
	must(t, m.AddConstraint(map[Var]float64{x: 1, z: 1}, GreaterOrEqual, 1))
	got := probeBinaryImplications(m, 1e-9)
	if got.ProbeSides != 2 || got.InfeasibleSides != 0 {
		t.Fatalf("unexpected probe telemetry: %+v", got)
	}
	if got.GlobalLower[x] != 1 || got.GlobalTightened != 1 {
		t.Fatalf("common consequence not recovered: lower=%v tightened=%d", got.GlobalLower, got.GlobalTightened)
	}
	if len(got.Implications) != 2 {
		t.Fatalf("got %d side implications, want 2", len(got.Implications))
	}
}

func TestProbeBinaryImplicationsFiresOnNidda(t *testing.T) {
	p := loadProblem(t, "bench/problems/miplib/neos-3754480-nidda.json")
	m := modelFromProblem(t, p)
	got := probeBinaryImplications(m, 1e-8)
	rows := make(map[int]struct{})
	for _, rel := range got.Relations {
		rows[rel.SourceRow] = struct{}{}
	}
	if len(rows) == 0 || got.ProbeSides == 0 || len(got.Implications) == 0 {
		t.Fatalf("nidda structure did not fire: vb_rows=%d probe_sides=%d implications=%d", len(rows), got.ProbeSides, len(got.Implications))
	}
}

func TestSeparateImpliedBoundCutsUsesOtherSideGlobalBound(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0), Upper: Float64(10)})
	z := m.AddVar(VarSpec{Type: Binary})
	imps := []boundImplication{{Binary: int(z), Value: 1, Variable: int(x), Bound: 4, Upper: true, SourceRow: 7}}
	cuts := separateImpliedBoundCuts(m, imps, []float64{8, 0.5}, 1e-9)
	if len(cuts) != 1 {
		t.Fatalf("got %d cuts, want 1", len(cuts))
	}
	cut := cuts[0]
	if cut.Sense != LessOrEqual || cut.RHS != 10 || cut.Coeffs[x] != 1 || cut.Coeffs[z] != 6 {
		t.Fatalf("unexpected implied-bound cut: %+v", cut)
	}
	if got := separateImpliedBoundCuts(m, imps, []float64{6, 0.5}, 1e-9); len(got) != 0 {
		t.Fatalf("non-violated cut was emitted: %+v", got)
	}
}

func TestInferenceCountsAndScorePreferBalancedDeductions(t *testing.T) {
	imps := []boundImplication{
		{Binary: 1, Value: 0, Variable: 2},
		{Binary: 1, Value: 0, Variable: 3},
		{Binary: 1, Value: 1, Variable: 2},
		{Binary: 1, Value: 1, Variable: 3},
	}
	down, up := implicationInferenceCounts(4, imps)
	if down[1] != 2 || up[1] != 2 {
		t.Fatalf("unexpected inference counts: down=%v up=%v", down, up)
	}
	if inferenceAdjustedScore(1, down[1], up[1]) <= inferenceAdjustedScore(1, 0, 0) {
		t.Fatal("balanced inference did not increase branching score")
	}
}
