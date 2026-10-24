package qopt

import (
	"context"
	"testing"
)

func TestLocalBranchingConstraintMatchesHammingBall(t *testing.T) {
	m := NewModel(Minimize)
	for i := 0; i < 4; i++ {
		_ = m.AddVar(VarSpec{Name: string(rune('a' + i)), Type: Binary})
	}
	sub, ok := localBranchingModel(m, []float64{1, 0, 1, 0}, 2, 1e-8)
	if !ok {
		t.Fatal("expected local branching model")
	}
	cons := sub.constraints[len(sub.constraints)-1]
	if cons.Sense != LessOrEqual || cons.RHS != 0 {
		t.Fatalf("distance row sense=%v rhs=%g, want <= 0", cons.Sense, cons.RHS)
	}
	if cons.Coeffs[0] != -1 || cons.Coeffs[1] != 1 || cons.Coeffs[2] != -1 || cons.Coeffs[3] != 1 {
		t.Fatalf("unexpected distance coefficients: %v", cons.Coeffs)
	}
}

func TestLocalBranchingStateRunsOncePerIncumbent(t *testing.T) {
	var state localBranchingState
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

func TestRunLocalBranchingFindsImprovement(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	z := m.AddVar(VarSpec{Name: "z", Type: Binary})
	w := m.AddVar(VarSpec{Name: "w", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 2, z: 8, w: 3}))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1, z: 1, w: 1}, Equal, 1))
	opts := normalizeOptions(Options{LocalBranchingRadius: 2})
	run := runLocalBranching(context.Background(), m, Maximize, []float64{1, 0, 0, 0}, 1, 4000, opts)
	if !run.improved || run.objective != 8 {
		t.Fatalf("local branching result improved=%t objective=%g, want true/8", run.improved, run.objective)
	}
}
