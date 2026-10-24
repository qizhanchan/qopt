package qopt

import (
	"context"
	"testing"
)

func TestProximityModelUsesBinaryDistanceObjective(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	z := m.AddVar(VarSpec{Name: "z", Type: Continuous, Upper: Float64(10)})
	must(t, m.SetObjective(map[Var]float64{x: 5, y: 3, z: 1}))
	opts := normalizeOptions(Options{ProximityMinImprove: 0.1})
	sub, ok := proximityModel(m, []float64{1, 0, 2}, 7, 12, opts)
	if !ok {
		t.Fatal("expected proximity model")
	}
	if sub.direction != Minimize {
		t.Fatalf("direction=%v, want minimize", sub.direction)
	}
	if len(sub.objective) != 2 || sub.objective[x] != -1 || sub.objective[y] != 1 {
		t.Fatalf("unexpected proximity objective: %v", sub.objective)
	}
	cutoff := sub.constraints[len(sub.constraints)-1]
	if cutoff.Sense != GreaterOrEqual || cutoff.Coeffs[x] != 5 || cutoff.Coeffs[y] != 3 || cutoff.Coeffs[z] != 1 {
		t.Fatalf("unexpected original-objective cutoff: %+v", cutoff)
	}
	if cutoff.RHS != 7.5 {
		t.Fatalf("cutoff rhs=%g, want 7.5", cutoff.RHS)
	}
}

func TestProximityStateRunsOncePerIncumbent(t *testing.T) {
	var state proximityState
	if !state.eligible(10) {
		t.Fatal("new incumbent should be eligible")
	}
	state.record(10)
	if state.eligible(10) {
		t.Fatal("same incumbent should not be retried")
	}
	if !state.eligible(20) {
		t.Fatal("new incumbent should reset eligibility")
	}
}

func TestRunProximityFindsImprovement(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	z := m.AddVar(VarSpec{Name: "z", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 2, z: 4}))
	opts := normalizeOptions(Options{})
	run := runProximity(context.Background(), m, []float64{1, 0, 0}, 1, 7, 4000, opts)
	if !run.improved || run.objective <= 1 {
		t.Fatalf("proximity result improved=%t objective=%g, want a strict improvement", run.improved, run.objective)
	}
}
