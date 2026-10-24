package lp

import (
	"context"
	"math"
	"testing"
)

// sessionTestBody is a small bounded LP used by every session test:
//
//	max 3x + 2y + z  s.t.  x + y + z <= 6, x + 2y >= 2, x - y + 2z = 3.
func sessionTestBody() (*Incremental, []float64) {
	c := []float64{3, 2, 1}
	A := [][]float64{
		{1, 1, 1},
		{1, 2, 0},
		{1, -1, 2},
	}
	senses := []Sense{LessOrEqual, GreaterOrEqual, Equal}
	b := []float64{6, 2, 3}
	body := probFromDense(c, A, senses, b, nil)
	return NewIncremental(Problem{C: body.C, Cols: body.Cols, Senses: body.Senses}), b
}

// TestSessionMatchesStateless walks a deterministic bound sequence —
// tightenings, relaxations, RHS shifts, and an infeasible node — through a
// Session and checks every result against a fresh stateless solve of the same
// LP. The sequence deliberately revisits shapes so the in-place path (not
// just first-adoption) is exercised.
func TestSessionMatchesStateless(t *testing.T) {
	inc, b := sessionTestBody()
	ref, _ := sessionTestBody()
	sess := NewSession(inc)
	opts := testOptions()

	inf := math.Inf(1)
	steps := []struct {
		b  []float64
		ub []float64
	}{
		{b, []float64{inf, inf, inf}},
		{b, []float64{2, inf, inf}},          // tighten x
		{b, []float64{2, 1, inf}},            // tighten y
		{b, []float64{inf, inf, inf}},        // relax back (uncertified path)
		{[]float64{5, 2, 3}, []float64{inf, inf, inf}}, // RHS change
		{[]float64{5, 2, 3}, []float64{0, 0, 0}},       // heavily fixed
		{[]float64{6, 8, 3}, []float64{1, 1, 1}},       // x+2y>=8 impossible under ub → infeasible
		{b, []float64{inf, 2, 2}},            // recover after infeasible
		{b, []float64{inf, inf, inf}},
	}

	var warm *WarmStart
	for i, step := range steps {
		got := sess.SolveRows(context.Background(), opts, nil, step.b, step.ub, warm)
		want := ref.SolveRows(context.Background(), opts, nil, step.b, step.ub, nil)
		if got.Status != want.Status {
			t.Fatalf("step %d: status session=%s stateless=%s", i, got.Status, want.Status)
		}
		if got.Status == Optimal && math.Abs(got.Objective-want.Objective) > 1e-8 {
			t.Fatalf("step %d: obj session=%.12g stateless=%.12g", i, got.Objective, want.Objective)
		}
		warm = got.Warm
	}
}

// TestSessionInPlaceReoptIsUsed verifies the fast path actually engages: after
// the first (adopted) solve, a bound tightening must run as an in-place
// session reoptimize with no fallback and no fresh adoption.
func TestSessionInPlaceReoptIsUsed(t *testing.T) {
	inc, b := sessionTestBody()
	sess := NewSession(inc)
	opts := testOptions()

	first := sess.SolveRows(context.Background(), opts, nil, b, nil, nil)
	if first.Status != Optimal {
		t.Fatalf("first: status=%s", first.Status)
	}
	if first.Stats.SessionAdoptions != 1 {
		t.Fatalf("first: SessionAdoptions=%d, want 1", first.Stats.SessionAdoptions)
	}

	// Identical re-solve: zero primal violation, so the in-place path must
	// terminate with no dual pivots and, unlike a warm install, no refactor.
	second := sess.SolveRows(context.Background(), opts, nil, b, nil, first.Warm)
	if second.Status != Optimal || math.Abs(second.Objective-first.Objective) > 1e-9 {
		t.Fatalf("second: status=%s obj=%g, want optimal %g", second.Status, second.Objective, first.Objective)
	}
	if second.Stats.SessionReopts != 1 || second.Stats.SessionFallbacks != 0 || second.Stats.SessionAdoptions != 0 {
		t.Fatalf("second: session stats %+v, want one clean in-place reopt", second.Stats)
	}
	if second.Stats.Refactors != 0 {
		t.Fatalf("second: Refactors=%d, want 0 (in-place reopt must not refactor on install)", second.Stats.Refactors)
	}

	// A bound tightening may pivot (and refactor on the eta-cost cadence),
	// but must still be answered in place.
	third := sess.SolveRows(context.Background(), opts, nil, b, []float64{1, math.Inf(1), math.Inf(1)}, first.Warm)
	if third.Status != Optimal {
		t.Fatalf("third: status=%s", third.Status)
	}
	if third.Stats.SessionReopts != 1 || third.Stats.SessionFallbacks != 0 || third.Stats.SessionAdoptions != 0 {
		t.Fatalf("third: session stats %+v, want one clean in-place reopt", third.Stats)
	}
}

// TestSessionExtRowsMatchesStateless drives the session across two RowSet
// identities (base body and one appended cut row) with interleaved bound
// changes — the node-LP/heuristic-LP interleave the MIP layer produces — and
// checks every solve statelessly.
func TestSessionExtRowsMatchesStateless(t *testing.T) {
	inc, b := sessionTestBody()
	ref, _ := sessionTestBody()
	sess := NewSession(inc)
	opts := testOptions()

	var cut SparseVector
	cut.Append(0, 1)
	cut.Append(1, 1)
	ext := &RowSet{Senses: []Sense{LessOrEqual}, Rows: []SparseVector{cut}}
	bExt := append(append([]float64(nil), b...), 4) // x + y <= 4

	inf := math.Inf(1)
	type step struct {
		ext *RowSet
		b   []float64
		ub  []float64
	}
	steps := []step{
		{nil, b, nil},
		{ext, bExt, nil},
		{ext, bExt, []float64{2, inf, inf}},
		{nil, b, []float64{2, inf, inf}},
		{ext, bExt, []float64{2, 1, inf}},
		{ext, bExt, nil},
		{nil, b, nil},
	}
	for i, st := range steps {
		got := sess.SolveRows(context.Background(), opts, st.ext, st.b, st.ub, nil)
		want := ref.SolveRows(context.Background(), opts, st.ext, st.b, st.ub, nil)
		if got.Status != want.Status {
			t.Fatalf("step %d: status session=%s stateless=%s", i, got.Status, want.Status)
		}
		if got.Status == Optimal && math.Abs(got.Objective-want.Objective) > 1e-8 {
			t.Fatalf("step %d: obj session=%.12g stateless=%.12g", i, got.Objective, want.Objective)
		}
	}
}

// TestSessionSolverDetached ensures the Solver a session solve exposes is a
// stable snapshot: a later session solve on the same body must not change the
// snapshot's basis or primal values (cut generation and reduced-cost fixing
// read it after subsequent solves have run).
func TestSessionSolverDetached(t *testing.T) {
	inc, b := sessionTestBody()
	sess := NewSession(inc)
	opts := testOptions()

	first := sess.SolveRows(context.Background(), opts, nil, b, nil, nil)
	if first.Status != Optimal || first.Solver == nil {
		t.Fatalf("first: status=%s solver=%v", first.Status, first.Solver)
	}
	snapBasis := append([]int(nil), first.Solver.BasicCols()...)
	snapX := make([]float64, first.Solver.N())
	for j := range snapX {
		snapX[j] = first.Solver.XValue(j)
	}

	second := sess.SolveRows(context.Background(), opts, nil, b, []float64{2, 2, 2}, first.Warm)
	if second.Status != Optimal {
		t.Fatalf("second: status=%s", second.Status)
	}

	for i, want := range snapBasis {
		if got := first.Solver.BasicCols()[i]; got != want {
			t.Fatalf("snapshot basis mutated at row %d: %d != %d", i, got, want)
		}
	}
	for j, want := range snapX {
		if got := first.Solver.XValue(j); math.Abs(got-want) > 1e-12 {
			t.Fatalf("snapshot x[%d] mutated: %g != %g", j, got, want)
		}
	}
}

// TestSessionDeterministic runs the same solve sequence through two fresh
// sessions and requires bit-identical objectives and iteration counts — the
// property the detsnap harness depends on.
func TestSessionDeterministic(t *testing.T) {
	run := func() ([]float64, []int) {
		inc, b := sessionTestBody()
		sess := NewSession(inc)
		opts := testOptions()
		inf := math.Inf(1)
		ubs := [][]float64{
			nil,
			{2, inf, inf},
			{2, 1, inf},
			{inf, inf, inf},
			{1, 1, 1},
			nil,
		}
		objs := make([]float64, 0, len(ubs))
		iters := make([]int, 0, len(ubs))
		for _, ub := range ubs {
			sol := sess.SolveRows(context.Background(), opts, nil, b, ub, nil)
			objs = append(objs, sol.Objective)
			iters = append(iters, sol.Iterations)
		}
		return objs, iters
	}
	o1, i1 := run()
	o2, i2 := run()
	for k := range o1 {
		if o1[k] != o2[k] || i1[k] != i2[k] {
			t.Fatalf("step %d: nondeterministic (obj %g vs %g, iters %d vs %d)", k, o1[k], o2[k], i1[k], i2[k])
		}
	}
}
