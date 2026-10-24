package lp

import (
	"context"
	"math"
	"testing"
)

func TestDenseLUFtranAndBtran(t *testing.T) {
	// B = [[2, 1], [4, 3]], represented column-wise.
	lu, err := factorDense([][]float64{
		{2, 4},
		{1, 3},
	})
	if err != nil {
		t.Fatalf("factorDense: %v", err)
	}

	x := lu.ftran([]float64{1, 2})
	if len(x) != 2 || math.Abs(x[0]-0.5) > 1e-12 || math.Abs(x[1]) > 1e-12 {
		t.Fatalf("ftran = %#v, want [0.5 0]", x)
	}

	y := lu.btran([]float64{3, 2})
	if len(y) != 2 || math.Abs(y[0]-0.5) > 1e-12 || math.Abs(y[1]-0.5) > 1e-12 {
		t.Fatalf("btran = %#v, want [0.5 0.5]", y)
	}

	if lu.minPivot() <= 0 {
		t.Fatalf("minPivot = %g, want positive", lu.minPivot())
	}
	if lu.growthFactor() < 1 {
		t.Fatalf("growthFactor = %g, want >= 1", lu.growthFactor())
	}
}

func TestDenseLUEtaUpdateMatchesFullRefactor(t *testing.T) {
	b0 := []float64{2, 4}
	b1 := []float64{1, 3}
	enter := []float64{3, 5}

	base, err := factorDense([][]float64{b0, b1})
	if err != nil {
		t.Fatalf("base factorDense: %v", err)
	}
	u := base.ftran(enter)
	eta := newEtaUpdate(0, u)

	full, err := factorDense([][]float64{enter, b1})
	if err != nil {
		t.Fatalf("full factorDense: %v", err)
	}

	rhs := []float64{7, 11}
	updatedF := base.ftran(rhs)
	applyEtaFTRAN(updatedF, eta)
	fullF := full.ftran(rhs)
	for i := range fullF {
		if math.Abs(updatedF[i]-fullF[i]) > 1e-12 {
			t.Fatalf("updated ftran = %#v, full = %#v", updatedF, fullF)
		}
	}

	updatedBRHS := append([]float64(nil), rhs...)
	applyEtaBTRAN(updatedBRHS, eta)
	updatedB := base.btran(updatedBRHS)
	fullB := full.btran(rhs)
	for i := range fullB {
		if math.Abs(updatedB[i]-fullB[i]) > 1e-12 {
			t.Fatalf("updated btran = %#v, full = %#v", updatedB, fullB)
		}
	}
}

func TestEtaUpdateStoresSparseColumn(t *testing.T) {
	u := []float64{0, 2, 0, -3, 0}
	eta := newEtaUpdate(1, u)
	if eta.pivot != 2 {
		t.Fatalf("pivot = %v, want 2", eta.pivot)
	}
	if eta.col.Len() != 1 || eta.col.idx[0] != 3 || eta.col.val[0] != -3 {
		t.Fatalf("eta col = idx %v val %v, want only row 3 = -3", eta.col.idx, eta.col.val)
	}

	x := []float64{4, 8, 6, 10, 12}
	applyEtaFTRAN(x, eta)
	wantF := []float64{4, 4, 6, 22, 12}
	for i := range x {
		if math.Abs(x[i]-wantF[i]) > 1e-12 {
			t.Fatalf("FTRAN x[%d] = %v, want %v (all=%v)", i, x[i], wantF[i], x)
		}
	}

	y := []float64{4, 8, 6, 10, 12}
	applyEtaBTRAN(y, eta)
	wantB := []float64{4, 19, 6, 10, 12}
	for i := range y {
		if math.Abs(y[i]-wantB[i]) > 1e-12 {
			t.Fatalf("BTRAN y[%d] = %v, want %v (all=%v)", i, y[i], wantB[i], y)
		}
	}
}

func TestRevisedRefactorRecomputesBasicValues(t *testing.T) {
	prob := probFromDense(
		[]float64{1},
		[][]float64{{1}},
		[]Sense{LessOrEqual},
		[]float64{5},
		[]float64{2},
	)
	s, err := newSolver(context.Background(), prob, testOptions())
	if err != nil {
		t.Fatalf("newSolver: %v", err)
	}

	// Force the original variable to its upper bound while the slack remains
	// basic. A correct refactor must rebuild slack = 5 - 2 = 3 from B^{-1}rhs.
	s.status[0] = ColAtUpper
	if err := s.refactor(); err != nil {
		t.Fatalf("refactor: %v", err)
	}
	if len(s.xBasic) != 1 || math.Abs(s.xBasic[0]-3) > 1e-12 {
		t.Fatalf("xBasic = %#v, want [3]", s.xBasic)
	}
	resid, _, _ := s.basisResidualInf()
	if resid > 1e-12 {
		t.Fatalf("basis residual = %g, want near zero", resid)
	}
}

func TestDenseLUSingularBasis(t *testing.T) {
	_, err := factorDense([][]float64{
		{1, 2},
		{2, 4},
	})
	if err != errSingularBasis {
		t.Fatalf("err = %v, want errSingularBasis", err)
	}
}
