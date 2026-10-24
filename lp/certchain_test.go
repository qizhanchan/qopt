package lp

import (
	"context"
	"math"
	"testing"
)

// certChainProblem is a small LP whose repeated warm re-solves stay on
// classifyWarm's exact-match certified class: same shape, same flips, no
// bound relaxations.
func certChainProblem() (Problem, []float64) {
	// max 3x + 2y s.t. x + y <= 4, x <= 3, y <= 3.
	c := []float64{3, 2}
	A := [][]float64{
		{1, 1},
		{1, 0},
		{0, 1},
	}
	senses := []Sense{LessOrEqual, LessOrEqual, LessOrEqual}
	b := []float64{4, 3, 3}
	return probFromDense(c, A, senses, b, nil), b
}

// TestCertChainCadence drives a chain of certified warm re-solves and checks
// that the chain counter grows only through certificate-trusted accepts, that
// the solve at the limit runs the explicit dual-feasibility check instead,
// and that the verified accept resets the chain.
func TestCertChainCadence(t *testing.T) {
	saved := certChainLimit
	certChainLimit = 3
	defer func() { certChainLimit = saved }()

	prob, b := certChainProblem()
	inc := NewIncremental(Problem{C: prob.C, Cols: prob.Cols, Senses: prob.Senses})
	opts := testOptions()

	cold := inc.Solve(context.Background(), opts, b, nil, nil)
	if cold.Status != Optimal || cold.Warm == nil {
		t.Fatalf("cold solve: status=%s warm=%v", cold.Status, cold.Warm)
	}
	if cold.Warm.certChain != 0 {
		t.Fatalf("cold handle certChain=%d, want 0", cold.Warm.certChain)
	}

	warm := cold.Warm
	// Solves 1..3: incoming chains 0,1,2 are under the limit — every accept
	// rides the certificate and increments the chain.
	for i := 1; i <= 3; i++ {
		sol := inc.Solve(context.Background(), opts, b, nil, warm)
		if sol.Status != Optimal || math.Abs(sol.Objective-cold.Objective) > 1e-9 {
			t.Fatalf("solve %d: status=%s obj=%g, want optimal %g", i, sol.Status, sol.Objective, cold.Objective)
		}
		if sol.Stats.WarmCertAccepts != 1 || sol.Stats.WarmCertVerifies != 0 {
			t.Fatalf("solve %d: accepts=%d verifies=%d, want certificate accept without verification",
				i, sol.Stats.WarmCertAccepts, sol.Stats.WarmCertVerifies)
		}
		if sol.Warm.certChain != i {
			t.Fatalf("solve %d: certChain=%d, want %d", i, sol.Warm.certChain, i)
		}
		warm = sol.Warm
	}

	// Solve 4: incoming chain 3 hits the limit — the accept must run the
	// explicit check, pass it, and hand back a reset chain.
	verified := inc.Solve(context.Background(), opts, b, nil, warm)
	if verified.Status != Optimal || math.Abs(verified.Objective-cold.Objective) > 1e-9 {
		t.Fatalf("verified solve: status=%s obj=%g", verified.Status, verified.Objective)
	}
	if verified.Stats.WarmCertAccepts != 0 || verified.Stats.WarmCertVerifies != 1 || verified.Stats.WarmCertVerifyFails != 0 {
		t.Fatalf("verified solve stats: %+v, want one passed chain-limit verification", verified.Stats)
	}
	if verified.Warm.certChain != 0 {
		t.Fatalf("verified handle certChain=%d, want 0", verified.Warm.certChain)
	}
}

// TestDualInfeasGapSafeBound checks the bound-correction identity behind the
// verify-failure repair: for a basis that is optimal under one objective but
// dual-infeasible under another, vertex objective + gap must still be a valid
// upper bound on the second objective's true optimum, and the gap must be
// exactly zero under the objective the basis is optimal for.
func TestDualInfeasGapSafeBound(t *testing.T) {
	prob, b := certChainProblem()
	inc := NewIncremental(Problem{C: prob.C, Cols: prob.Cols, Senses: prob.Senses})
	opts := testOptions()
	// Bound both variables so every structural column has a finite range.
	upper := []float64{3, 3}
	sol := inc.Solve(context.Background(), opts, b, upper, nil)
	if sol.Status != Optimal || sol.Solver == nil {
		t.Fatalf("solve: status=%s", sol.Status)
	}
	s := sol.Solver

	// The optimal basis must have gap 0 under its own cost.
	own := makePhaseIICost(s)
	if gap, finite := s.dualInfeasGap(own); !finite || gap != 0 {
		t.Fatalf("own cost: gap=%g finite=%v, want 0/true", gap, finite)
	}

	// max 2x + 3y over the same feasible set: optimum x=1, y=3 → 11, while
	// the current vertex (x=3, y=1) scores 9 — the basis is dual-infeasible
	// for this cost and the corrected bound must cover the true optimum.
	alt := []float64{2, 3}
	altCost := make([]float64, s.NCols())
	vertexObj := 0.0
	for j := 0; j < s.N(); j++ {
		altCost[j] = -alt[j]
		vertexObj += alt[j] * s.XValue(j)
	}
	gap, finite := s.dualInfeasGap(altCost)
	if !finite || gap <= 0 {
		t.Fatalf("alt cost: gap=%g finite=%v, want positive finite", gap, finite)
	}
	if vertexObj+gap < 11 {
		t.Fatalf("safe bound %g does not cover true optimum 11 (vertex %g, gap %g)", vertexObj+gap, vertexObj, gap)
	}
}

func TestDualFeasibilityIgnoresFixedNonbasicColumn(t *testing.T) {
	prob := probFromDense(
		[]float64{1, -10},
		[][]float64{{1, 0}},
		[]Sense{LessOrEqual},
		[]float64{1},
		[]float64{1, 0},
	)
	s, err := newSolver(context.Background(), prob, testOptions())
	if err != nil {
		t.Fatal(err)
	}
	if st := s.phaseI(); st != Optimal {
		t.Fatalf("phase I status=%s", st)
	}
	cost := makePhaseIICost(s)
	if st := s.runSimplex(cost, false); st != Optimal {
		t.Fatalf("phase II status=%s", st)
	}
	if !s.dualFeasibleAt(cost) {
		t.Fatal("fixed nonbasic column incorrectly makes basis dual-infeasible")
	}
	if gap, finite := s.dualInfeasGap(cost); !finite || gap != 0 {
		t.Fatalf("gap=%g finite=%v, want 0/true", gap, finite)
	}
}

// TestCertChainLimitOverrides pins the 0 (always verify) and negative
// (legacy unlimited) override semantics of certChainLimit.
func TestCertChainLimitOverrides(t *testing.T) {
	saved := certChainLimit
	defer func() { certChainLimit = saved }()

	prob, b := certChainProblem()

	certChainLimit = 0
	inc := NewIncremental(Problem{C: prob.C, Cols: prob.Cols, Senses: prob.Senses})
	opts := testOptions()
	cold := inc.Solve(context.Background(), opts, b, nil, nil)
	sol := inc.Solve(context.Background(), opts, b, nil, cold.Warm)
	if sol.Stats.WarmCertAccepts != 0 || sol.Stats.WarmCertVerifies != 1 {
		t.Fatalf("limit 0: stats=%+v, want every certified accept verified", sol.Stats)
	}
	if sol.Warm.certChain != 0 {
		t.Fatalf("limit 0: certChain=%d, want 0", sol.Warm.certChain)
	}

	certChainLimit = -1
	inc = NewIncremental(Problem{C: prob.C, Cols: prob.Cols, Senses: prob.Senses})
	warm := inc.Solve(context.Background(), opts, b, nil, nil).Warm
	for i := 1; i <= 5; i++ {
		sol = inc.Solve(context.Background(), opts, b, nil, warm)
		if sol.Stats.WarmCertAccepts != 1 || sol.Stats.WarmCertVerifies != 0 {
			t.Fatalf("unlimited solve %d: stats=%+v, want unverified certificate accept", i, sol.Stats)
		}
		warm = sol.Warm
	}
	if warm.certChain != 5 {
		t.Fatalf("unlimited: certChain=%d, want 5", warm.certChain)
	}
}
