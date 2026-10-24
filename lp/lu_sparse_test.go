package lp

import (
	"math"
	"math/rand"
	"testing"
)

func denseColumnFromSparse(m int, c SparseVector) []float64 {
	v := make([]float64, m)
	for k, i := range c.idx {
		v[i] = c.val[k]
	}
	return v
}

func SparseVectorFromDense(v []float64) SparseVector {
	var c SparseVector
	for i, x := range v {
		if x != 0 {
			c.idx = append(c.idx, i)
			c.val = append(c.val, x)
		}
	}
	return c
}

// crossCheckSparseLU factors B with both backends and asserts ftran/btran
// agree on a random RHS to 1e-9.
func crossCheckSparseLU(t *testing.T, name string, denseCols [][]float64, rhs []float64) {
	t.Helper()
	m := len(denseCols)
	sparseCols := make([]SparseVector, m)
	for j, col := range denseCols {
		sparseCols[j] = SparseVectorFromDense(col)
	}
	dense, err := factorDense(denseCols)
	if err != nil {
		t.Fatalf("%s: factorDense: %v", name, err)
	}
	sparse, err := factorSparse(sparseCols, m)
	if err != nil {
		t.Fatalf("%s: factorSparse: %v", name, err)
	}
	df := dense.ftran(rhs)
	sf := sparse.ftran(rhs)
	for i := range df {
		if math.Abs(df[i]-sf[i]) > 1e-9 {
			t.Fatalf("%s: ftran mismatch row %d: dense=%g sparse=%g (rhs=%v)", name, i, df[i], sf[i], rhs)
		}
	}
	db := dense.btran(rhs)
	sb := sparse.btran(rhs)
	for i := range db {
		if math.Abs(db[i]-sb[i]) > 1e-9 {
			t.Fatalf("%s: btran mismatch row %d: dense=%g sparse=%g (rhs=%v)", name, i, db[i], sb[i], rhs)
		}
	}
}

func TestSparseLUMatchesDenseSmall(t *testing.T) {
	// 2x2 well-conditioned.
	crossCheckSparseLU(t, "2x2-basic", [][]float64{
		{2, 4},
		{1, 3},
	}, []float64{7, 11})

	// 3x3 sparse identity-ish.
	crossCheckSparseLU(t, "3x3-near-identity", [][]float64{
		{1, 0, 0},
		{0, 2, 0},
		{0, 1, 4},
	}, []float64{3, 5, 9})

	// 4x4 with mixed signs and zeros.
	crossCheckSparseLU(t, "4x4-mixed", [][]float64{
		{2, 0, 1, 0},
		{0, 3, 0, 1},
		{1, 0, 2, 0},
		{0, 1, 0, 5},
	}, []float64{1, -2, 3, -4})
}

func TestSparseLUSingularBasis(t *testing.T) {
	cols := []SparseVector{
		SparseVectorFromDense([]float64{1, 2}),
		SparseVectorFromDense([]float64{2, 4}),
	}
	_, err := factorSparse(cols, 2)
	if err != errSingularBasis {
		t.Fatalf("err = %v, want errSingularBasis", err)
	}
}

func TestSparseLURandom(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0FFEE))
	for trial := 0; trial < 20; trial++ {
		m := 5 + rng.Intn(15)
		denseCols := make([][]float64, m)
		for j := 0; j < m; j++ {
			denseCols[j] = make([]float64, m)
			// Ensure diagonal dominance so the random basis is non-singular.
			denseCols[j][j] = 5 + rng.Float64()
			for i := 0; i < m; i++ {
				if i == j {
					continue
				}
				if rng.Float64() < 0.3 {
					denseCols[j][i] = rng.NormFloat64()
				}
			}
		}
		rhs := make([]float64, m)
		for i := range rhs {
			rhs[i] = rng.NormFloat64()
		}
		crossCheckSparseLU(t, "random", denseCols, rhs)
	}
}
