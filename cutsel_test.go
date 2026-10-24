package qopt

import "testing"

func TestSelectCutsRespectsCap(t *testing.T) {
	x, y := Var(0), Var(1)
	values := []float64{0, 0}
	cuts := []Constraint{
		{Name: "a", Coeffs: map[Var]float64{x: 1}, Sense: GreaterOrEqual, RHS: 5},  // efficacy 5
		{Name: "b", Coeffs: map[Var]float64{x: 1}, Sense: GreaterOrEqual, RHS: 3},  // efficacy 3, parallel to a
		{Name: "c", Coeffs: map[Var]float64{y: 1}, Sense: GreaterOrEqual, RHS: 10}, // efficacy 10, orthogonal
		{Name: "d", Coeffs: map[Var]float64{y: 1}, Sense: GreaterOrEqual, RHS: 1},  // efficacy 1, parallel to c
	}
	got := selectCuts(cuts, values, 2)
	if len(got) != 2 {
		t.Fatalf("len(got) = %d, want 2", len(got))
	}
	names := map[string]bool{}
	for _, c := range got {
		names[c.Name] = true
	}
	// c (efficacy 10) picked first; a (efficacy 5) is orthogonal to c so it
	// survives next; b and d get dropped as near-duplicates of a and c.
	if !names["c"] || !names["a"] {
		t.Fatalf("selected %v, want {a, c}", names)
	}
}

func TestSelectCutCandidatesTracksDisposition(t *testing.T) {
	x, y, z := Var(0), Var(1), Var(2)
	values := []float64{0, 0, 0}
	cuts := []Constraint{
		{Name: "gomory_a", Coeffs: map[Var]float64{x: 1}, Sense: GreaterOrEqual, RHS: 5},
		{Name: "gomory_parallel", Coeffs: map[Var]float64{x: 2}, Sense: GreaterOrEqual, RHS: 8},
		{Name: "mir_b", Coeffs: map[Var]float64{y: 1}, Sense: GreaterOrEqual, RHS: 3},
		{Name: "cmir_c", Coeffs: map[Var]float64{z: 1}, Sense: GreaterOrEqual, RHS: 1},
	}
	candidates := makeCutCandidates(cuts, values, 3, 4)
	selected := selectCutCandidates(candidates, 2)
	if len(selected) != 2 {
		t.Fatalf("selected = %v, want two candidates", selected)
	}
	if candidates[0].state != cutCandidateSelected || candidates[2].state != cutCandidateSelected {
		t.Fatalf("selected states = %q/%q, want gomory_a and mir_b", candidates[0].state, candidates[2].state)
	}
	if candidates[1].state != cutCandidateRejectedParallel {
		t.Fatalf("parallel candidate state = %q", candidates[1].state)
	}
	if candidates[3].state != cutCandidateRejectedCap {
		t.Fatalf("cap candidate state = %q", candidates[3].state)
	}
	if candidates[0].family != "gomory" || candidates[2].family != "mir" || candidates[3].family != "cmir" {
		t.Fatalf("families = %q/%q/%q", candidates[0].family, candidates[2].family, candidates[3].family)
	}
	if candidates[0].creationRound != 4 || candidates[0].coefficientRatio != 1 || candidates[0].density != 1.0/3.0 {
		t.Fatalf("candidate metadata = %+v", candidates[0])
	}
}

func TestSelectCutsNoOpBelowCap(t *testing.T) {
	x := Var(0)
	cuts := []Constraint{
		{Name: "a", Coeffs: map[Var]float64{x: 1}, Sense: GreaterOrEqual, RHS: 1},
	}
	got := selectCuts(cuts, []float64{0}, 5)
	if len(got) != 1 || got[0].Name != "a" {
		t.Fatalf("selectCuts under cap changed the input: %v", got)
	}
}

func TestCutEfficacyAndCosineSim(t *testing.T) {
	x, y := Var(0), Var(1)
	c := Constraint{Coeffs: map[Var]float64{x: 1}, Sense: GreaterOrEqual, RHS: 5}
	norm := coeffNorm(c.Coeffs)
	closeTo(t, norm, 1)
	closeTo(t, cutEfficacy(c, []float64{0, 0}, norm), 5)
	closeTo(t, cutEfficacy(c, []float64{5, 0}, norm), 0) // exactly tight -> zero violation

	a := map[Var]float64{x: 1}
	b := map[Var]float64{x: 2}
	orth := map[Var]float64{y: 1}
	closeTo(t, cosineSim(a, coeffNorm(a), b, coeffNorm(b)), 1) // same direction
	closeTo(t, cosineSim(a, coeffNorm(a), orth, coeffNorm(orth)), 0)
}
