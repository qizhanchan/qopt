package lp

import (
	"context"
	"math"
	"testing"
)

func TestRangedRow(t *testing.T) {
	prob := probFromDense([]float64{1}, [][]float64{{1}}, []Sense{LessOrEqual}, []float64{3}, nil)
	prob.RangeWidths = []float64{2}
	res := Solve(context.Background(), prob, testOptions(), nil)
	if res.Status != Optimal || math.Abs(res.Objective-3) > 1e-8 || math.Abs(res.Values[0]-3) > 1e-8 {
		t.Fatalf("status=%s objective=%g values=%v", res.Status, res.Objective, res.Values)
	}
}

func TestRangedRowNegativeUpper(t *testing.T) {
	prob := probFromDense([]float64{-1}, [][]float64{{1}}, []Sense{LessOrEqual}, []float64{-1}, nil)
	prob.RangeWidths = []float64{2} // -3 <= x <= -1; shift-free LP allows x >= 0, so infeasible.
	res := Solve(context.Background(), prob, testOptions(), nil)
	if res.Status != Infeasible {
		t.Fatalf("status=%s, want infeasible", res.Status)
	}
}
