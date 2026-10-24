package lp

import (
	"context"
	"math"
	"testing"
)

// testOptions mirrors the defaults qopt.normalizeOptions applies before
// handing Options to the engine.
func testOptions() Options {
	return Options{
		FeasibilityTol: 1e-8,
		OptimalityTol:  1e-9,
		MaxIterations:  100000,
	}
}

// probFromDense builds a Problem from a dense row-major matrix, converting to
// the sparse-column form the engine consumes.
func probFromDense(c []float64, a [][]float64, senses []Sense, b, upper []float64) Problem {
	n := len(c)
	cols := make([]SparseVector, n)
	for j := 0; j < n; j++ {
		for i := range a {
			cols[j].Append(i, a[i][j])
		}
	}
	return Problem{C: c, Cols: cols, Senses: senses, B: b, Upper: upper}
}

// TestDualReoptimizeAfterBoundTightening exercises the warm-start path
// directly. We solve a small LP cold to extract a basis, tighten the
// upper bound of an active variable, and re-solve with the warm handle.
// The dual simplex should fix the now-infeasible basic value and return
// the same objective as a fresh cold solve.
func TestDualReoptimizeAfterBoundTightening(t *testing.T) {
	// max 3x + 2y s.t. x + y <= 4, x <= 3, y <= 3. Optimum: x=3,y=1,obj=11.
	c := []float64{3, 2}
	A := [][]float64{
		{1, 1},
		{1, 0},
		{0, 1},
	}
	senses := []Sense{LessOrEqual, LessOrEqual, LessOrEqual}
	b := []float64{4, 3, 3}

	opts := testOptions()
	resp := Solve(context.Background(), probFromDense(c, A, senses, b, []float64{math.Inf(1), math.Inf(1)}), opts, nil)
	if resp.Status != Optimal {
		t.Fatalf("base solve status = %s", resp.Status)
	}
	if math.Abs(resp.Objective-11) > 1e-6 {
		t.Fatalf("base obj = %g, want 11", resp.Objective)
	}
	if resp.Warm == nil {
		t.Fatal("expected warm-start handle from base solve")
	}

	// Tighten x's upper to 2 via Problem.Upper. Original optimum had x=3
	// which is now infeasible. Cold solve gives x=2,y=2,obj=10.
	tight := probFromDense(c, A, senses, b, []float64{2, math.Inf(1)})
	hot := Solve(context.Background(), tight, opts, resp.Warm)
	if hot.Status != Optimal {
		t.Fatalf("warm solve status = %s", hot.Status)
	}
	if math.Abs(hot.Objective-10) > 1e-6 {
		t.Fatalf("warm obj = %g, want 10", hot.Objective)
	}

	cold := Solve(context.Background(), tight, opts, nil)
	if cold.Status != Optimal {
		t.Fatalf("cold (tight) status = %s", cold.Status)
	}
	if math.Abs(cold.Objective-hot.Objective) > 1e-6 {
		t.Fatalf("warm obj %g != cold obj %g", hot.Objective, cold.Objective)
	}
}

// TestDualReoptimizeInfeasible tightens a bound to make the problem
// infeasible and confirms the dual simplex reports infeasibility instead
// of looping or returning a bogus solution.
func TestDualReoptimizeInfeasible(t *testing.T) {
	// max x s.t. x >= 3, x <= 10. Optimum: x=10.
	c := []float64{1}
	A := [][]float64{
		{1},
		{1},
	}
	senses := []Sense{GreaterOrEqual, LessOrEqual}
	b := []float64{3, 10}

	opts := testOptions()
	resp := Solve(context.Background(), probFromDense(c, A, senses, b, []float64{math.Inf(1)}), opts, nil)
	if resp.Status != Optimal {
		t.Fatalf("base status = %s", resp.Status)
	}
	if resp.Warm == nil {
		t.Fatal("expected warm handle")
	}

	// Tighten upper to 2, which contradicts x >= 3 → infeasible.
	tight := probFromDense(c, A, senses, b, []float64{2})
	hot := Solve(context.Background(), tight, opts, resp.Warm)
	if hot.Status != Infeasible {
		t.Fatalf("warm tight status = %s, want infeasible", hot.Status)
	}
}

func TestDualReoptimizeCertifiesInfeasibleLeavingRow(t *testing.T) {
	// max x s.t. x >= 3, x <= 10. Tightening x's native upper bound below 3
	// leaves no bounded-variable movement capable of repairing the GE row.
	c := []float64{1}
	A := [][]float64{{1}, {1}}
	senses := []Sense{GreaterOrEqual, LessOrEqual}
	b := []float64{3, 10}
	opts := testOptions()
	opts.CertifyInfeasible = true

	body := probFromDense(c, A, senses, b, nil)
	inc := NewIncremental(Problem{C: body.C, Cols: body.Cols, Senses: body.Senses})
	base := inc.Solve(context.Background(), opts, b, []float64{math.Inf(1)}, nil)
	if base.Status != Optimal || base.Warm == nil {
		t.Fatalf("base solve: status=%s warm=%v", base.Status, base.Warm)
	}

	hot := inc.Solve(context.Background(), opts, b, []float64{2}, base.Warm)
	if hot.Status != Infeasible || !hot.InfeasibilityCertified {
		t.Fatalf("warm solve: status=%s certified=%v, want certified infeasible", hot.Status, hot.InfeasibilityCertified)
	}
	if hot.Stats.ColdStarts != 0 || hot.Stats.WarmFailures != 0 {
		t.Fatalf("certified solve stats: %+v, want no cold fallback/failure", hot.Stats)
	}
	if hot.Stats.WarmDualInfeasible != 1 || hot.Stats.WarmInfeasibleCertificates != 1 {
		t.Fatalf("certified solve stats: %+v, want one dual infeasible certificate", hot.Stats)
	}
}

func TestDualReoptimizeDoesNotCertifyRepairableRow(t *testing.T) {
	// The same parent with upper x=4 remains feasible. Its warm basis starts
	// outside the tightened bound, but slack/surplus movement repairs the row;
	// the certificate must not prune it.
	c := []float64{1}
	A := [][]float64{{1}, {1}}
	senses := []Sense{GreaterOrEqual, LessOrEqual}
	b := []float64{3, 10}
	opts := testOptions()
	opts.CertifyInfeasible = true

	body := probFromDense(c, A, senses, b, nil)
	inc := NewIncremental(Problem{C: body.C, Cols: body.Cols, Senses: body.Senses})
	base := inc.Solve(context.Background(), opts, b, []float64{math.Inf(1)}, nil)
	if base.Status != Optimal || base.Warm == nil {
		t.Fatalf("base solve: status=%s warm=%v", base.Status, base.Warm)
	}

	hot := inc.Solve(context.Background(), opts, b, []float64{4}, base.Warm)
	if hot.Status != Optimal || math.Abs(hot.Objective-4) > 1e-9 {
		t.Fatalf("warm solve: status=%s obj=%g, want optimal 4", hot.Status, hot.Objective)
	}
	if hot.InfeasibilityCertified || hot.Stats.WarmInfeasibleCertificates != 0 {
		t.Fatalf("repairable solve was certified infeasible: %+v", hot.Stats)
	}
}

func TestDualCertificateSkipsToleranceZeroUnboundedColumn(t *testing.T) {
	// max x s.t. x + 1e-12*z >= 3, x <= 10, z unbounded above. Tightening
	// x's upper to 2 leaves the GE row unrepairable except through z's
	// sub-tolerance coefficient (|alpha| = 1e-12 is below the dual ratio
	// test's own candidacy threshold, so the dual exits with no entering
	// column). SCIP's checkfarkas treats such coefficients as numerical
	// zero (SCIPsetIsDualfeasZero) even on unbounded columns; the
	// certificate must do the same instead of rejecting on z's infinite
	// width — that rejection blocked 374 of 376 neos17 proofs.
	c := []float64{1, 0}
	A := [][]float64{
		{1, 1e-12},
		{1, 0},
	}
	senses := []Sense{GreaterOrEqual, LessOrEqual}
	b := []float64{3, 10}
	opts := testOptions()
	opts.CertifyInfeasible = true

	body := probFromDense(c, A, senses, b, nil)
	inc := NewIncremental(Problem{C: body.C, Cols: body.Cols, Senses: body.Senses})
	inf := math.Inf(1)
	base := inc.Solve(context.Background(), opts, b, []float64{inf, inf}, nil)
	if base.Status != Optimal || base.Warm == nil {
		t.Fatalf("base solve: status=%s warm=%v", base.Status, base.Warm)
	}

	hot := inc.Solve(context.Background(), opts, b, []float64{2, inf}, base.Warm)
	if hot.Status != Infeasible || !hot.InfeasibilityCertified {
		t.Fatalf("warm solve: status=%s certified=%v stats=%+v, want certified infeasible",
			hot.Status, hot.InfeasibilityCertified, hot.Stats)
	}
	if hot.Stats.WarmCertRejectInfinite != 0 {
		t.Fatalf("certificate rejected on the tolerance-zero unbounded column: %+v", hot.Stats)
	}
}
