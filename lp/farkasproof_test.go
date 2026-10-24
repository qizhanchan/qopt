package lp

import (
	"context"
	"math"
	"testing"
)

// TestFarkasProofExport verifies the certified dual exit exports a
// bound-independent proof row: max x s.t. x >= 3, x <= 10, with x's box
// tightened to [0,2]. The proof coef·x >= rhs must be violated by the
// certifying box, satisfied by every row-feasible point (x in [3,10]), and
// must NOT certify the wider, feasible box [0,4].
func TestFarkasProofExport(t *testing.T) {
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
		t.Fatalf("warm solve: status=%s certified=%v", hot.Status, hot.InfeasibilityCertified)
	}
	fp := hot.FarkasProof
	if fp == nil {
		t.Fatal("certified infeasible exit did not export a Farkas proof")
	}
	if len(fp.Inds) == 0 || len(fp.Inds) != len(fp.Coefs) {
		t.Fatalf("malformed proof: inds=%v coefs=%v", fp.Inds, fp.Coefs)
	}
	eval := func(x float64) float64 {
		sum := 0.0
		for k, j := range fp.Inds {
			if j != 0 {
				t.Fatalf("proof references unexpected structural column %d", j)
			}
			sum += fp.Coefs[k] * x
		}
		return sum
	}
	maxOver := func(lo, hi float64) float64 {
		return math.Max(eval(lo), eval(hi))
	}

	// Row-feasible points satisfy the proof inequality.
	for _, x := range []float64{3, 5, 10} {
		if eval(x) < fp.RHS-1e-6 {
			t.Fatalf("row-feasible x=%g violates proof: %g < rhs %g", x, eval(x), fp.RHS)
		}
	}
	// The certifying box [0,2] provably violates it.
	if maxOver(0, 2) >= fp.RHS {
		t.Fatalf("certifying box not violated: maxact=%g rhs=%g", maxOver(0, 2), fp.RHS)
	}
	// The feasible box [0,4] must not be certified.
	if maxOver(0, 4) < fp.RHS {
		t.Fatalf("feasible box falsely certified: maxact=%g rhs=%g", maxOver(0, 4), fp.RHS)
	}
}
