package lp

import (
	"context"
	"math"
	"testing"
)

func TestStrongBranchSessionMatchesColdAndPreservesParent(t *testing.T) {
	inc, b := sessionTestBody()
	opts := testOptions()
	base := inc.Solve(context.Background(), opts, b, nil, nil)
	if base.Status != Optimal || base.Solver == nil {
		t.Fatalf("base: status=%s solver=%v", base.Status, base.Solver)
	}
	parentObj := base.Objective
	parentBasis := append([]int(nil), base.Solver.BasicCols()...)
	sb := inc.NewStrongBranchSession(base.Solver, nil)
	if sb == nil {
		t.Fatal("strong-branch session unavailable")
	}

	upper := []float64{1, math.Inf(1), math.Inf(1)}
	probe := sb.Probe(context.Background(), opts, b, upper, 100)
	cold := inc.Solve(context.Background(), opts, b, upper, nil)
	if cold.Status != Optimal || !probe.BoundValid {
		t.Fatalf("probe=%+v cold=%s", probe, cold.Status)
	}
	if math.Abs(probe.Bound-cold.Objective) > 1e-8 || math.Abs(probe.Estimate-cold.Objective) > 1e-8 {
		t.Fatalf("probe estimate/bound=%g/%g, cold=%g", probe.Estimate, probe.Bound, cold.Objective)
	}
	if probe.Stats.ColdStarts != 0 {
		t.Fatalf("probe cold starts=%d, want 0", probe.Stats.ColdStarts)
	}
	if base.Objective != parentObj {
		t.Fatalf("parent objective mutated: %g != %g", base.Objective, parentObj)
	}
	for i, want := range parentBasis {
		if got := base.Solver.BasicCols()[i]; got != want {
			t.Fatalf("parent basis row %d mutated: %d != %d", i, got, want)
		}
	}
}

func TestStrongBranchSessionCertifiesInfeasibleWithoutColdSolve(t *testing.T) {
	c := []float64{1}
	A := [][]float64{{1}, {1}}
	senses := []Sense{GreaterOrEqual, LessOrEqual}
	b := []float64{3, 10}
	opts := testOptions()
	opts.CertifyInfeasible = true
	body := probFromDense(c, A, senses, b, nil)
	inc := NewIncremental(Problem{C: body.C, Cols: body.Cols, Senses: body.Senses})
	base := inc.Solve(context.Background(), opts, b, nil, nil)
	if base.Status != Optimal || base.Solver == nil {
		t.Fatalf("base: status=%s solver=%v", base.Status, base.Solver)
	}
	sb := inc.NewStrongBranchSession(base.Solver, nil)
	probe := sb.Probe(context.Background(), opts, b, []float64{2}, 100)
	if !probe.InfeasibleCertified {
		t.Fatalf("probe=%+v, want certified infeasible", probe)
	}
	if probe.Stats.ColdStarts != 0 {
		t.Fatalf("probe cold starts=%d, want 0", probe.Stats.ColdStarts)
	}
}

func TestStrongBranchIterationLimitSeparatesEstimateAndBound(t *testing.T) {
	inc, b := sessionTestBody()
	opts := testOptions()
	base := inc.Solve(context.Background(), opts, b, nil, nil)
	if base.Status != Optimal || base.Solver == nil {
		t.Fatalf("base: status=%s solver=%v", base.Status, base.Solver)
	}
	sb := inc.NewStrongBranchSession(base.Solver, nil)
	if sb == nil {
		t.Fatal("strong-branch session unavailable")
	}

	// Tighten several structural columns at once, as propagation can do after
	// one branching bound. One pivot is intentionally insufficient.
	upper := []float64{0.5, 0.5, 0.5}
	probe := sb.Probe(context.Background(), opts, b, upper, 1)
	if !probe.IterationLimit || !probe.EstimateValid {
		t.Fatalf("probe=%+v, want an iteration-limit estimate", probe)
	}
	if !probe.BoundValid {
		t.Fatalf("probe=%+v, dual-feasible capped state should provide a valid bound", probe)
	}
	cold := inc.Solve(context.Background(), opts, b, upper, nil)
	if cold.Status == Optimal && probe.Bound+1e-8 < cold.Objective {
		t.Fatalf("capped max bound %g is below cold optimum %g", probe.Bound, cold.Objective)
	}
}
