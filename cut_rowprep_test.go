package qopt

import (
	"math"
	"testing"
)

func TestCutRowprepEliminatesSmallPositiveCoefficientWithLowerBound(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Lower: Float64(2), Upper: Float64(5)})
	y := m.AddVar(VarSpec{Name: "y", Lower: Float64(0), Upper: Float64(10)})
	cut := Constraint{Coeffs: map[Var]float64{x: 1e-12, y: 1}, Sense: LessOrEqual, RHS: 4}

	got := cleanupCutRow(m, cut, []float64{3, 5}, testCutRowprepConfig())
	if !got.success || !got.modified || got.removedCoefficients != 1 {
		t.Fatalf("rowprep = %+v", got)
	}
	if _, exists := got.row.Coeffs[x]; exists {
		t.Fatalf("small coefficient survived: %+v", got.row)
	}
	if math.Abs(got.row.RHS-(4-2e-12)) > 1e-15 {
		t.Fatalf("rhs = %.16g, want %.16g", got.row.RHS, 4-2e-12)
	}
	assertCutRelaxationOnGrid(t, m, cut, got.row, [][]float64{{2, 0}, {2, 4}, {5, 0}, {5, 3}})
}

func TestCutRowprepEliminatesSmallNegativeCoefficientWithUpperBound(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Lower: Float64(2), Upper: Float64(5)})
	y := m.AddVar(VarSpec{Name: "y", Lower: Float64(0), Upper: Float64(10)})
	cut := Constraint{Coeffs: map[Var]float64{x: -1e-12, y: 1}, Sense: LessOrEqual, RHS: 4}

	got := cleanupCutRow(m, cut, []float64{3, 5}, testCutRowprepConfig())
	if !got.success || got.removedCoefficients != 1 {
		t.Fatalf("rowprep = %+v", got)
	}
	if math.Abs(got.row.RHS-(4+5e-12)) > 1e-15 {
		t.Fatalf("rhs = %.16g, want %.16g", got.row.RHS, 4+5e-12)
	}
	assertCutRelaxationOnGrid(t, m, cut, got.row, [][]float64{{2, 0}, {2, 4}, {5, 0}, {5, 4}})
}

func TestCutRowprepRestoresGreaterEqualSense(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Lower: Float64(2), Upper: Float64(5)})
	y := m.AddVar(VarSpec{Name: "y", Lower: Float64(0), Upper: Float64(10)})
	cut := Constraint{Coeffs: map[Var]float64{x: 1e-12, y: 1}, Sense: GreaterOrEqual, RHS: 4}

	got := cleanupCutRow(m, cut, []float64{3, 3}, testCutRowprepConfig())
	if !got.success || got.row.Sense != GreaterOrEqual {
		t.Fatalf("rowprep = %+v", got)
	}
	if math.Abs(got.row.RHS-(4-5e-12)) > 1e-15 {
		t.Fatalf("rhs = %.16g, want %.16g", got.row.RHS, 4-5e-12)
	}
	assertCutRelaxationOnGrid(t, m, cut, got.row, [][]float64{{2, 4}, {5, 4}, {2, 5}, {5, 5}})
}

func TestCutRowprepRoundsCoefficientWithSideRelaxation(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Lower: Float64(0), Upper: Float64(5)})
	y := m.AddVar(VarSpec{Name: "y", Lower: Float64(0), Upper: Float64(10)})
	cut := Constraint{Coeffs: map[Var]float64{x: 1 - 5e-10, y: 1}, Sense: LessOrEqual, RHS: 4}

	got := cleanupCutRow(m, cut, []float64{2, 5}, testCutRowprepConfig())
	if !got.success || got.roundedCoefficients != 1 || got.row.Coeffs[x] != 1 {
		t.Fatalf("rowprep = %+v", got)
	}
	if math.Abs(got.row.RHS-(4+2.5e-9)) > 1e-14 {
		t.Fatalf("rhs = %.16g, want %.16g", got.row.RHS, 4+2.5e-9)
	}
	assertCutRelaxationOnGrid(t, m, cut, got.row, [][]float64{{0, 0}, {0, 4}, {5, 0}})
}

func TestCutRowprepScalesByPowerOfTwo(t *testing.T) {
	m := NewModel(Minimize)
	y := m.AddVar(VarSpec{Name: "y", Lower: Float64(0), Upper: Float64(10)})
	cut := Constraint{Coeffs: map[Var]float64{y: 1e6}, Sense: LessOrEqual, RHS: 1e6}

	got := cleanupCutRow(m, cut, []float64{2}, testCutRowprepConfig())
	if !got.success || got.scale >= 1 || got.scale <= 0 {
		t.Fatalf("rowprep = %+v", got)
	}
	if got.row.Coeffs[y] > cutRowprepMaxCoefficient {
		t.Fatalf("scaled coefficient = %g", got.row.Coeffs[y])
	}
	assertCutRelaxationOnGrid(t, m, cut, got.row, [][]float64{{0}, {1}})
}

func TestCutRowprepRejectsWeakAndEmptyRows(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Lower: Float64(0), Upper: Float64(1)})
	weak := Constraint{Coeffs: map[Var]float64{x: 1}, Sense: GreaterOrEqual, RHS: 1e-5}
	if got := cleanupCutRow(m, weak, []float64{0}, testCutRowprepConfig()); got.reason != cutCandidateRejectedRowprepInefficacious {
		t.Fatalf("weak rowprep = %+v", got)
	}

	fixed := NewModel(Minimize)
	z := fixed.AddVar(VarSpec{Name: "z", Lower: Float64(1), Upper: Float64(1)})
	empty := Constraint{Coeffs: map[Var]float64{z: 1}, Sense: LessOrEqual, RHS: 0}
	if got := cleanupCutRow(fixed, empty, []float64{1}, testCutRowprepConfig()); got.reason != cutCandidateRejectedRowprepEmpty {
		t.Fatalf("empty rowprep = %+v", got)
	}
}

func testCutRowprepConfig() cutRowprepConfig {
	return cutRowprepConfig{
		maxCoefficientRatio: 1e9,
		maxCoefficient:      cutRowprepMaxCoefficient,
		minEfficacy:         cutRowprepMinEfficacy,
		integralTolerance:   1e-9,
	}
}

func assertCutRelaxationOnGrid(t *testing.T, m *Model, original, cleaned Constraint, points [][]float64) {
	t.Helper()
	for _, point := range points {
		if !constraintSatisfied(original, point, 1e-12) {
			continue
		}
		if !constraintSatisfied(cleaned, point, 1e-9) {
			t.Fatalf("cleaned cut excludes original-feasible point %v: original=%+v cleaned=%+v", point, original, cleaned)
		}
	}
	_ = m
}

func constraintSatisfied(c Constraint, values []float64, tol float64) bool {
	lhs := 0.0
	for _, entry := range sortedCoeffs(c.Coeffs) {
		lhs += entry.coef * values[entry.v]
	}
	switch c.Sense {
	case LessOrEqual:
		return lhs <= c.RHS+tol
	case GreaterOrEqual:
		return lhs >= c.RHS-tol
	case Equal:
		return math.Abs(lhs-c.RHS) <= tol
	default:
		return false
	}
}
