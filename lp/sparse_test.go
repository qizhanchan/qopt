package lp

import (
	"math"
	"testing"
)

func TestSparseColumnHelpers(t *testing.T) {
	c := SparseVector{}
	c.Append(1, 2.0)
	c.Append(3, -1.5)
	c.Append(4, 0) // dropped
	if c.Len() != 2 {
		t.Fatalf("nnz = %d, want 2", c.Len())
	}

	y := []float64{10, 20, 30, 40, 50}
	if got := c.dotDense(y); math.Abs(got-(2.0*20+(-1.5)*40)) > 1e-12 {
		t.Fatalf("dotDense = %g, want %g", got, 2.0*20+(-1.5)*40)
	}

	dst := make([]float64, 5)
	c.writeDense(dst)
	want := []float64{0, 2.0, 0, -1.5, 0}
	for i, w := range want {
		if math.Abs(dst[i]-w) > 1e-12 {
			t.Fatalf("writeDense[%d] = %g, want %g", i, dst[i], w)
		}
	}

	scratch := make([]float64, 5)
	c.addToDense(scratch, 2)
	if math.Abs(scratch[1]-4) > 1e-12 || math.Abs(scratch[3]+3) > 1e-12 {
		t.Fatalf("addToDense scratch = %#v", scratch)
	}
	c.addToDense(scratch, 0) // no-op
	if math.Abs(scratch[1]-4) > 1e-12 || math.Abs(scratch[3]+3) > 1e-12 {
		t.Fatalf("addToDense scale=0 mutated scratch = %#v", scratch)
	}
}
