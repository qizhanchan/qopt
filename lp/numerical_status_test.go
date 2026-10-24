package lp

import (
	"context"
	"math"
	"testing"
	"time"
)

func TestSingularBasisUpdateIsNumericalLimit(t *testing.T) {
	var basic, entering SparseVector
	basic.Append(0, 1)
	entering.Append(0, 1e-13)

	lu, err := factorDense([][]float64{{1}})
	if err != nil {
		t.Fatal(err)
	}
	s := &Solver{
		ctx:          context.Background(),
		opts:         Options{FeasibilityTol: 1e-14, OptimalityTol: 1e-9, MaxIterations: 10},
		start:        time.Now(),
		m:            1,
		n:            2,
		nCols:        2,
		cols:         []SparseVector{basic, entering},
		lower:        []float64{0, 0},
		upper:        []float64{math.Inf(1), math.Inf(1)},
		bNorm:        []float64{1},
		basicCols:    []int{0},
		status:       []ColStatus{ColBasic, ColAtLower},
		xBasic:       []float64{1},
		isArtificial: []bool{false, false},
		lu:           lu,
	}
	s.initScratch()

	if got := s.runSimplex([]float64{0, -1}, false); got != IterationLimit {
		t.Fatalf("status = %s, want iteration_limit for a singular basis update", got)
	}
}
