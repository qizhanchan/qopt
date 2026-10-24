package lp

import (
	"context"
	"math"
	"testing"
)

// TestPersistFactorKeepsReusedLU pins the cost-based cadence counters across
// a PersistFactor reuse: applyWarmStart installs the parent's LU without
// going through installFactor, so it must restore factorNnz/etaNnzSum itself.
// With factorNnz left at 0 the eta-vs-factor rule fires on the very first
// pivot and refactors away the LU this option exists to keep.
func TestPersistFactorKeepsReusedLU(t *testing.T) {
	// max 3x + 2y s.t. x + y <= 4, x <= 3, y <= 3.
	c := []float64{3, 2}
	A := [][]float64{
		{1, 1},
		{1, 0},
		{0, 1},
	}
	senses := []Sense{LessOrEqual, LessOrEqual, LessOrEqual}
	b := []float64{4, 3, 3}
	opts := testOptions()
	opts.PersistFactor = true
	// Dense LU: factorNnz = m² keeps the cost rule quiet for the couple of
	// reopt pivots, so any refactor below is a cadence-counter bug.
	opts.DenseLU = true

	body := probFromDense(c, A, senses, b, nil)
	inc := NewIncremental(Problem{C: body.C, Cols: body.Cols, Senses: body.Senses})

	inf := math.Inf(1)
	first := inc.Solve(context.Background(), opts, b, []float64{inf, inf}, nil)
	if first.Status != Optimal {
		t.Fatalf("cold solve: status=%s", first.Status)
	}
	if first.Warm == nil || first.Warm.lu == nil {
		t.Fatal("PersistFactor solve did not persist the factorization")
	}

	// Tighten x's upper bound below its basic value so the dual reopt must
	// pivot at least once on the reused factor.
	second := inc.Solve(context.Background(), opts, b, []float64{2, inf}, first.Warm)
	if second.Status != Optimal {
		t.Fatalf("warm solve: status=%s", second.Status)
	}
	if math.Abs(second.Objective-10) > 1e-9 {
		t.Fatalf("warm solve: obj=%.12g, want 10 (x=2, y=2)", second.Objective)
	}
	if second.Stats.FactorReuses != 1 {
		t.Fatalf("FactorReuses = %d, want 1", second.Stats.FactorReuses)
	}
	if second.Stats.EtaUpdates == 0 {
		t.Fatal("warm reopt performed no eta pivots — test no longer exercises the cadence")
	}
	if second.Stats.Refactors != 0 {
		t.Fatalf("Refactors = %d after factor reuse, want 0 (reused LU was discarded)", second.Stats.Refactors)
	}
}
