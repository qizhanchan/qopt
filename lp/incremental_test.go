package lp

import (
	"context"
	"math"
	"testing"
)

// TestIncrementalMatchesOneShot re-solves one body across a sequence of bound
// changes on a single Incremental (warm-started, like B&B nodes) and checks
// every solve against an independent one-shot Solve of the same LP.
func TestIncrementalMatchesOneShot(t *testing.T) {
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

	body := probFromDense(c, A, senses, b, nil)
	inc := NewIncremental(Problem{C: body.C, Cols: body.Cols, Senses: body.Senses})

	uppers := [][]float64{
		{math.Inf(1), math.Inf(1)}, // x=3,y=1 → 11
		{2, math.Inf(1)},           // x=2,y=2 → 10
		{2, 1},                     // x=2,y=1 → 8
		{math.Inf(1), math.Inf(1)}, // relax back → 11
	}
	var warm *WarmStart
	for i, ub := range uppers {
		got := inc.Solve(context.Background(), opts, b, ub, warm)
		want := Solve(context.Background(), probFromDense(c, A, senses, b, ub), opts, nil)
		if got.Status != want.Status {
			t.Fatalf("step %d: status incremental=%s oneshot=%s", i, got.Status, want.Status)
		}
		if math.Abs(got.Objective-want.Objective) > 1e-9 {
			t.Fatalf("step %d: obj incremental=%.12g oneshot=%.12g", i, got.Objective, want.Objective)
		}
		warm = got.Warm
	}
}

// TestIncrementalFlipPatternChange changes the RHS sign between solves. A
// compatible warm solve should retain the prior row orientation instead of
// rebuilding/remapping the standard form, while still matching the cold
// optimum.
func TestIncrementalFlipPatternChange(t *testing.T) {
	// min x s.t. x >= b0 (as -x <= -b0), x <= 10.
	c := []float64{-1} // max -x == min x
	A := [][]float64{
		{-1},
		{1},
	}
	senses := []Sense{LessOrEqual, LessOrEqual}
	opts := testOptions()

	body := probFromDense(c, A, senses, []float64{-3, 10}, nil)
	inc := NewIncremental(Problem{C: body.C, Cols: body.Cols, Senses: body.Senses})

	// b0 = -3: row 0 flips (b<0), optimum x=3, obj=-3.
	first := inc.Solve(context.Background(), opts, []float64{-3, 10}, nil, nil)
	if first.Status != Optimal || math.Abs(first.Objective-(-3)) > 1e-9 {
		t.Fatalf("first solve: status=%s obj=%g, want optimal -3", first.Status, first.Objective)
	}
	// b0 = +2: row 0 no longer flips (-x <= 2 is slack at x=0), optimum x=0.
	second := inc.Solve(context.Background(), opts, []float64{2, 10}, nil, first.Warm)
	if second.Status != Optimal || math.Abs(second.Objective) > 1e-9 {
		t.Fatalf("second solve: status=%s obj=%g, want optimal 0", second.Status, second.Objective)
	}
	if second.Stats.WarmReopts != 1 || second.Stats.WarmFlipRemaps != 0 || second.Stats.WarmFailures != 0 {
		t.Fatalf("second solve warm stats: %+v, want one direct reopt without remap/failure", second.Stats)
	}
	// Back to a flipped pattern with a stale warm handle from the other shape.
	third := inc.Solve(context.Background(), opts, []float64{-5, 10}, nil, second.Warm)
	if third.Status != Optimal || math.Abs(third.Objective-(-5)) > 1e-9 {
		t.Fatalf("third solve: status=%s obj=%g, want optimal -5", third.Status, third.Objective)
	}
	if third.Stats.WarmReopts != 1 || third.Stats.WarmFlipRemaps != 0 || third.Stats.WarmFailures != 0 {
		t.Fatalf("third solve warm stats: %+v, want one direct reopt without remap/failure", third.Stats)
	}
}

// TestSolveRowsMatchesOneShot solves a base LP, then re-solves with an
// appended cut row three ways — cold, with the base solve's warm handle
// (prefix extension), and with the extended solve's own handle (exact match)
// — and checks each against a one-shot Solve of the equivalent full LP.
func TestSolveRowsMatchesOneShot(t *testing.T) {
	// max 3x + 2y s.t. x + y <= 4, x <= 3, y <= 3. Optimum x=3,y=1 → 11.
	c := []float64{3, 2}
	A := [][]float64{
		{1, 1},
		{1, 0},
		{0, 1},
	}
	senses := []Sense{LessOrEqual, LessOrEqual, LessOrEqual}
	b := []float64{4, 3, 3}
	opts := testOptions()

	body := probFromDense(c, A, senses, b, nil)
	inc := NewIncremental(Problem{C: body.C, Cols: body.Cols, Senses: body.Senses})

	base := inc.Solve(context.Background(), opts, b, nil, nil)
	if base.Status != Optimal || math.Abs(base.Objective-11) > 1e-9 {
		t.Fatalf("base: status=%s obj=%g, want optimal 11", base.Status, base.Objective)
	}

	// Cut row: x + 2y >= 6 — violated at (3,1) (lhs=5). New optimum: the LP
	// max 3x+2y s.t. x+y<=4, x<=3, y<=3, x+2y>=6 → x=2,y=2 → 10.
	var cutRow SparseVector
	cutRow.Append(0, 1)
	cutRow.Append(1, 2)
	ext := &RowSet{Senses: []Sense{GreaterOrEqual}, Rows: []SparseVector{cutRow}}
	bExt := append(append([]float64(nil), b...), 6)

	// Reference: one-shot solve of the full 4-row LP.
	fullA := append(append([][]float64(nil), A...), []float64{1, 2})
	fullSenses := append(append([]Sense(nil), senses...), GreaterOrEqual)
	want := Solve(context.Background(), probFromDense(c, fullA, fullSenses, bExt, nil), opts, nil)
	if want.Status != Optimal || math.Abs(want.Objective-10) > 1e-9 {
		t.Fatalf("reference: status=%s obj=%g, want optimal 10", want.Status, want.Objective)
	}

	cold := inc.SolveRows(context.Background(), opts, ext, bExt, nil, nil)
	if cold.Status != Optimal || math.Abs(cold.Objective-want.Objective) > 1e-9 {
		t.Fatalf("cold ext: status=%s obj=%g, want %g", cold.Status, cold.Objective, want.Objective)
	}

	warmExt := inc.SolveRows(context.Background(), opts, ext, bExt, nil, base.Warm)
	if warmExt.Status != Optimal || math.Abs(warmExt.Objective-want.Objective) > 1e-9 {
		t.Fatalf("warm prefix ext: status=%s obj=%g, want %g", warmExt.Status, warmExt.Objective, want.Objective)
	}

	again := inc.SolveRows(context.Background(), opts, ext, bExt, nil, warmExt.Warm)
	if again.Status != Optimal || math.Abs(again.Objective-want.Objective) > 1e-9 {
		t.Fatalf("warm exact ext: status=%s obj=%g, want %g", again.Status, again.Objective, want.Objective)
	}

	// Dropping the ext again with the extended handle must not corrupt the
	// base solve (handle shape mismatch → cold).
	backToBase := inc.Solve(context.Background(), opts, b, nil, warmExt.Warm)
	if backToBase.Status != Optimal || math.Abs(backToBase.Objective-11) > 1e-9 {
		t.Fatalf("back to base: status=%s obj=%g, want optimal 11", backToBase.Status, backToBase.Objective)
	}
}

// TestSolveRowsLECut exercises an LE ext row (slack basic on extension) and a
// second appended row on top of an already-extended warm handle.
func TestSolveRowsLECut(t *testing.T) {
	// max x + y s.t. x <= 3, y <= 3. Optimum (3,3) → 6.
	c := []float64{1, 1}
	A := [][]float64{
		{1, 0},
		{0, 1},
	}
	senses := []Sense{LessOrEqual, LessOrEqual}
	b := []float64{3, 3}
	opts := testOptions()

	body := probFromDense(c, A, senses, b, nil)
	inc := NewIncremental(Problem{C: body.C, Cols: body.Cols, Senses: body.Senses})
	base := inc.Solve(context.Background(), opts, b, nil, nil)
	if base.Status != Optimal || math.Abs(base.Objective-6) > 1e-9 {
		t.Fatalf("base: obj=%g want 6", base.Objective)
	}

	// Cut 1: x + y <= 5 → optimum 5.
	var r1 SparseVector
	r1.Append(0, 1)
	r1.Append(1, 1)
	ext1 := &RowSet{Senses: []Sense{LessOrEqual}, Rows: []SparseVector{r1}}
	b1 := append(append([]float64(nil), b...), 5)
	s1 := inc.SolveRows(context.Background(), opts, ext1, b1, nil, base.Warm)
	if s1.Status != Optimal || math.Abs(s1.Objective-5) > 1e-9 {
		t.Fatalf("ext1: status=%s obj=%g want 5", s1.Status, s1.Objective)
	}

	// Cut 2 appended: 2x + y <= 6 → optimum x=1,y=4? y<=3 so x=1.5,y=3 → 4.5.
	var r2 SparseVector
	r2.Append(0, 2)
	r2.Append(1, 1)
	ext2 := &RowSet{
		Senses: []Sense{LessOrEqual, LessOrEqual},
		Rows:   []SparseVector{r1, r2},
	}
	b2 := append(append([]float64(nil), b...), 5, 6)
	s2 := inc.SolveRows(context.Background(), opts, ext2, b2, nil, s1.Warm)
	if s2.Status != Optimal || math.Abs(s2.Objective-4.5) > 1e-9 {
		t.Fatalf("ext2: status=%s obj=%g want 4.5", s2.Status, s2.Objective)
	}
}

func TestWarmSolveExposesRetainedRowOrientation(t *testing.T) {
	// The second RHS crosses zero. A cold form would flip GE to LE and use one
	// auxiliary column, while the warm solve deliberately retains GE and its
	// surplus+artificial layout from the producing basis.
	body := probFromDense(
		[]float64{0},
		[][]float64{{1}},
		[]Sense{GreaterOrEqual},
		[]float64{1},
		[]float64{10},
	)
	inc := NewIncremental(Problem{C: body.C, Cols: body.Cols, Senses: body.Senses})
	first := inc.Solve(context.Background(), testOptions(), []float64{1}, []float64{10}, nil)
	if first.Status != Optimal || first.Solver == nil {
		t.Fatalf("first solve: status=%s solver=%v", first.Status, first.Solver)
	}
	second := inc.Solve(context.Background(), testOptions(), []float64{-1}, []float64{10}, first.Warm)
	if second.Status != Optimal || second.Solver == nil {
		t.Fatalf("warm solve: status=%s solver=%v", second.Status, second.Solver)
	}
	if got := second.Solver.RowSenses(); len(got) != 1 || got[0] != GreaterOrEqual {
		t.Fatalf("warm row senses = %v, want retained GE orientation", got)
	}
	if got := second.Solver.RowFlips(); len(got) != 1 || got[0] {
		t.Fatalf("warm row flips = %v, want [false]", got)
	}
	if second.Solver.NCols() != 3 {
		t.Fatalf("warm standard-form columns = %d, want structural+surplus+artificial = 3", second.Solver.NCols())
	}
}
