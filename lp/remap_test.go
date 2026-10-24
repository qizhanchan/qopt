package lp

import (
	"context"
	"math"
	"math/rand"
	"testing"
)

// TestWarmFlipRemapMatchesCold drives the flip-drift remap hard: random small
// LPs with mixed senses are solved cold at one RHS, then re-solved at RHS
// vectors whose entry signs differ (which changes the row-sign flip pattern
// and hence the standard-form aux layout). The warm-started solve must agree
// with a cold solve of the same node to tight tolerance in status and
// objective — the remapped basis is supposed to be an exact change of
// coordinates, not a heuristic.
func TestWarmFlipRemapMatchesCold(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	senseChoices := []Sense{LessOrEqual, GreaterOrEqual, Equal}

	for trial := 0; trial < 200; trial++ {
		n := 2 + rng.Intn(4)
		m := 2 + rng.Intn(4)

		c := make([]float64, n)
		for j := range c {
			c[j] = math.Round(rng.Float64()*10) - 5
		}
		a := make([][]float64, m)
		senses := make([]Sense, m)
		for i := range a {
			a[i] = make([]float64, n)
			for j := range a[i] {
				if rng.Float64() < 0.4 {
					a[i][j] = math.Round(rng.Float64()*8) - 4
				}
			}
			senses[i] = senseChoices[rng.Intn(3)]
		}
		upper := make([]float64, n)
		for j := range upper {
			upper[j] = 1 + rng.Float64()*9 // finite bounds keep every LP bounded
		}
		newB := func() []float64 {
			b := make([]float64, m)
			for i := range b {
				b[i] = math.Round(rng.Float64()*12) - 6 // signs vary → flip drift
			}
			return b
		}

		body := probFromDense(c, a, senses, nil, upper)
		inc := NewIncremental(Problem{C: body.C, Cols: body.Cols, Senses: body.Senses})
		opts := testOptions()

		var warm *WarmStart
		for step := 0; step < 6; step++ {
			b := newB()
			got := inc.Solve(context.Background(), opts, b, upper, warm)
			cold := inc.Solve(context.Background(), opts, b, upper, nil)
			if got.Status != cold.Status {
				t.Fatalf("trial %d step %d: warm status=%s cold=%s (b=%v senses=%v)",
					trial, step, got.Status, cold.Status, b, senses)
			}
			if got.Status == Optimal && math.Abs(got.Objective-cold.Objective) > 1e-7 {
				t.Fatalf("trial %d step %d: warm obj=%.12g cold=%.12g (b=%v senses=%v)",
					trial, step, got.Objective, cold.Objective, b, senses)
			}
			if got.Status == Optimal {
				warm = got.Warm
			}
		}
	}
}

// TestAuxLayoutMatchesCompile pins auxLayout to the sequential assignment
// compile actually performs, via the compiled form's initial basis: each
// row's initially-basic column is its slack (LE) or artificial (GE/EQ).
func TestAuxLayoutMatchesCompile(t *testing.T) {
	c := []float64{1, 2}
	a := [][]float64{
		{1, 1},
		{1, -1},
		{2, 1},
		{0, 1},
	}
	senses := []Sense{LessOrEqual, GreaterOrEqual, Equal, LessOrEqual}
	body := probFromDense(c, a, senses, nil, nil)
	inc := NewIncremental(Problem{C: body.C, Cols: body.Cols, Senses: body.Senses})

	for _, flip := range [][]bool{
		{false, false, false, false},
		{true, false, true, false},
		{true, true, true, true},
	} {
		cf := inc.compile(flip)
		logical, artificial, nCols := auxLayout(inc.n, cf.senses)
		if nCols != cf.nCols {
			t.Fatalf("flip %v: auxLayout nCols=%d, compile nCols=%d", flip, nCols, cf.nCols)
		}
		for i, sense := range cf.senses {
			want := cf.basicCols0[i]
			var got int
			switch sense {
			case LessOrEqual:
				got = logical[i]
			default:
				got = artificial[i]
			}
			if got != want {
				t.Fatalf("flip %v row %d (sense %d): layout col=%d, compile basic col=%d",
					flip, i, sense, got, want)
			}
		}
	}
}
