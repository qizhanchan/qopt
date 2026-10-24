package qopt

import (
	"context"
	"math"
	"testing"
)

// makeMIRTestModel builds a 3-binary model for the single-row tests.
func makeMIRTestModel(t *testing.T) *Model {
	t.Helper()
	m := NewModel(Maximize)
	for i := 0; i < 3; i++ {
		m.AddVar(VarSpec{Type: Binary})
	}
	return m
}

// TestMIR_SingleRowOnHalfScaling hand-computes a known MIR cut and checks
// our implementation matches. Constraint: 3x1 + 5x2 + 7x3 ≤ 11. All binary.
// Scale k=0.5 → scaled LE: 1.5x1 + 2.5x2 + 3.5x3 ≤ 5.5, floor RHS = 5, f0=0.5.
// For each coef: 1.5 → f=0.5 → coef=floor(1.5)=1; 2.5 → coef=floor(2.5)=2;
// 3.5 → coef=floor(3.5)=3. Cut: x1 + 2x2 + 3x3 ≤ 5.
// LP point that violates it: x = (1, 1, 1) → LHS = 6 > 5 ✓ violated.
func TestMIR_SingleRowOnHalfScaling(t *testing.T) {
	m := makeMIRTestModel(t)
	coefs := map[Var]float64{0: 3, 1: 5, 2: 7}
	lpValues := []float64{1, 1, 1}
	cut, ok := singleRowMIR(m, coefs, 11, 0.5, lpValues, 1e-6)
	if !ok {
		t.Fatalf("expected MIR cut to fire on hand-computed example")
	}
	if cut.Sense != LessOrEqual {
		t.Fatalf("sense = %v, want LE", cut.Sense)
	}
	if math.Abs(cut.RHS-5) > 1e-9 {
		t.Fatalf("cut RHS = %v, want 5", cut.RHS)
	}
	expected := map[Var]float64{0: 1, 1: 2, 2: 3}
	if len(cut.Coeffs) != len(expected) {
		t.Fatalf("coef count = %d, want %d (cut=%v)", len(cut.Coeffs), len(expected), cut.Coeffs)
	}
	for v, want := range expected {
		got := cut.Coeffs[v]
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("coef[v%d] = %v, want %v", v, got, want)
		}
	}
}

// TestMIR_NoFireOnIntegerEverything confirms that scale=1 on a pure-integer
// constraint with integer RHS yields no cut (f0 = 0).
func TestMIR_NoFireOnIntegerRHSAtScale1(t *testing.T) {
	m := makeMIRTestModel(t)
	coefs := map[Var]float64{0: 3, 1: 5}
	_, ok := singleRowMIR(m, coefs, 11, 1.0, []float64{1, 1, 0}, 1e-6)
	if ok {
		t.Fatalf("scale=1 on integer-RHS constraint must not produce a cut (f0=0)")
	}
}

// TestMIR_NotViolatedDropped ensures cuts that the current LP satisfies are
// not returned (we only want separating cuts).
func TestMIR_NotViolatedDropped(t *testing.T) {
	m := makeMIRTestModel(t)
	coefs := map[Var]float64{0: 3, 1: 5, 2: 7}
	// LP at (0,0,0): LHS=0, RHS=5; not violated.
	_, ok := singleRowMIR(m, coefs, 11, 0.5, []float64{0, 0, 0}, 1e-6)
	if ok {
		t.Fatalf("non-violating cut must be dropped")
	}
}

// TestMIR_ContinuousSlackCarriesThrough mirrors pk1's c1 row shape: positive
// integer coefs on binaries plus a -1 continuous slack. Under k=0.5 the
// slack's φ_cont = -1 / (1-0.5) = -2... wait actually k·c = -0.5, divided
// by (1-0.5)=0.5 gives -1. So the slack's cut coef is -1.
func TestMIR_ContinuousSlackCarriesThrough(t *testing.T) {
	m := NewModel(Maximize)
	m.AddVar(VarSpec{Type: Binary}) // v0
	m.AddVar(VarSpec{Type: Binary}) // v1
	m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0)})
	coefs := map[Var]float64{0: 5, 1: 7, 2: -1}
	rhs := 11.0
	// LP point: x0=1, x1=1, slack=1 → LHS = 5+7-1 = 11 OK
	// MIR k=0.5: f0=0.5; integer coefs: floor(2.5)=2, floor(3.5)=3;
	// slack: k·(-1) = -0.5, φ_cont = -0.5 / 0.5 = -1
	// Cut: 2*x0 + 3*x1 - 1*slack ≤ 5
	// At LP (1,1,1): LHS = 2+3-1 = 4 ≤ 5 → not violated.
	// At LP (1,1,0): LHS = 2+3-0 = 5 → not violated (≤ tol).
	// Force violation: x0=1, x1=1, slack=0 with stricter check. Try (1,1,-0.5)
	// — but slack ≥ 0, so use (1, 1, 0) and check sign behavior by inflating
	// the integer side. The point of this test is the COEFFICIENT, not
	// violation; pass a point that would violate the cut to receive it.
	lpValues := []float64{1, 1, -1} // slack value −1 makes LHS = 5+3−(−1) = wait
	// Let me recompute. Coefs are {2, 3, -1}. At (1,1,−1): LHS = 2+3+(−1)·(−1) = 2+3+1 = 6 > 5 ✓
	cut, ok := singleRowMIR(m, coefs, rhs, 0.5, lpValues, 1e-6)
	if !ok {
		t.Fatalf("expected cut to fire when LP violates")
	}
	if math.Abs(cut.Coeffs[Var(0)]-2) > 1e-9 || math.Abs(cut.Coeffs[Var(1)]-3) > 1e-9 {
		t.Errorf("integer coefs wrong: %v", cut.Coeffs)
	}
	if math.Abs(cut.Coeffs[Var(2)]-(-1)) > 1e-9 {
		t.Errorf("continuous slack coef = %v, want -1", cut.Coeffs[Var(2)])
	}
}

func TestCMIR_UpperBoundComplementation(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Continuous, Lower: Float64(0), Upper: Float64(1)})

	// x + y <= 1.5 at (0.5, 1). Complementing y at its upper bound gives
	// x - z_y <= 0.5. MIR yields x - 2*z_y <= 0, which translates back to
	// x + 2*y <= 2 and separates the supplied LP point by 0.5.
	cut, ok := complementedMIRBest(m, map[Var]float64{x: 1, y: 1}, 1.5, []float64{0.5, 1}, 1e-6)
	if !ok {
		t.Fatal("expected complemented MIR cut")
	}
	scale := cut.Coeffs[x]
	if math.Abs(scale) < 1e-12 || math.Abs(cut.Coeffs[y]/scale-2) > 1e-9 || math.Abs(cut.RHS/scale-2) > 1e-9 {
		t.Fatalf("cut = %v <= %g, want x + 2y <= 2", cut.Coeffs, cut.RHS)
	}
}

func TestCMIR_CancelContinuousVariableWithPositiveCombination(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Continuous, Lower: Float64(0), Upper: Float64(1)})
	z := m.AddVar(VarSpec{Name: "z", Type: Binary})

	rowA := map[Var]float64{x: 1, y: 1}
	rowB := map[Var]float64{y: -2, z: 1}
	factor := -rowA[y] / rowB[y]
	if factor <= 0 {
		t.Fatalf("factor = %g, want a valid nonnegative row multiplier", factor)
	}
	agg, rhs := aggregateRow(rowA, 1.2, rowB, 0, 1, factor)
	cleanSmallCoefficients(agg, 1e-10)
	if _, exists := agg[y]; exists {
		t.Fatalf("continuous variable was not cancelled: %v", agg)
	}
	if math.Abs(agg[x]-1) > 1e-9 || math.Abs(agg[z]-0.5) > 1e-9 || math.Abs(rhs-1.2) > 1e-9 {
		t.Fatalf("aggregation = %v <= %g, want x + 0.5z <= 1.2", agg, rhs)
	}
	_ = m
}

// TestMIR_OptionDefaultOff: with UseMIRCuts=false (default), solver behavior
// is unchanged. Sanity vs the default path.
func TestMIR_OptionDefaultOff(t *testing.T) {
	build := func() *Model {
		m := NewModel(Maximize)
		x := m.AddVar(VarSpec{Name: "x", Type: Binary})
		y := m.AddVar(VarSpec{Name: "y", Type: Binary})
		must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
		must(t, m.AddConstraint(map[Var]float64{x: 3, y: 2}, LessOrEqual, 5))
		return m
	}
	off, errOff := build().Solve(context.Background(), Options{})
	mir, errMir := build().Solve(context.Background(), Options{UseMIRCuts: true, MaxCutRounds: 2})
	if errOff != nil || errMir != nil {
		t.Fatalf("solve errors: off=%v mir=%v", errOff, errMir)
	}
	if off.Status != Optimal || mir.Status != Optimal {
		t.Fatalf("non-optimal: off=%s mir=%s", off.Status, mir.Status)
	}
	if math.Abs(off.Objective-mir.Objective) > 1e-9 {
		t.Fatalf("objectives diverged: off=%v mir=%v", off.Objective, mir.Objective)
	}
}

// TestMIR_DoesNotMutateModel — like Gomory's no-mutation test but for MIR.
// MIR cuts must not leak into the user's model.constraints.
func TestMIR_DoesNotMutateModel(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	z := m.AddVar(VarSpec{Name: "z", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 5, y: 4, z: 3}))
	must(t, m.AddConstraint(map[Var]float64{x: 3, y: 5, z: 7}, LessOrEqual, 11))

	before := len(m.constraints)
	res, err := m.Solve(context.Background(), Options{UseMIRCuts: true, MaxCutRounds: 3, MaxCutDepth: 2})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal {
		t.Fatalf("status = %s, want optimal", res.Status)
	}
	if len(m.constraints) != before {
		t.Fatalf("MIR mutated model constraints: before=%d after=%d", before, len(m.constraints))
	}
}

// TestMIR_MarkshareLikePreservesOptimum is the closest analog to pk1 we can
// fit in a unit test: an equality constraint with positive integer coefs
// summed to an odd RHS, all binaries. MIR + B&B must not break correctness.
func TestMIR_MarkshareLikePreservesOptimum(t *testing.T) {
	build := func() *Model {
		// min x1 + x2 + x3 + x4 + x5  s.t.  3x1 + 5x2 + 7x3 + 9x4 + 11x5 = 17
		// All binaries. The optimum picks the smallest subset summing to 17.
		// 17 = 11 + 3 + ... not 3 vars. Try {x5=1 (11), x3=1 (7)} → sum=18 nope.
		// {x4=1 (9), x3=1 (7), x1=1 (3)} → 19 nope.
		// {x5=1 (11), x2=1 (5), x1=1 (3)} → 19.
		// Actually exact 17 from {3,5,7,9,11}: need subset summing to 17.
		// 11+5+? → 17-16=1, no. 11+? → 6, no. 9+7+? → 16+1, no. 9+5+3=17 ✓ (3 vars).
		// 7+5+? → 12+5, no. 7+3+? → 10+7, no. So unique: x2=x3=x4=1, others=0.
		// Objective = 3 (sum of x's).
		m := NewModel(Minimize)
		vs := make([]Var, 5)
		coefs := []float64{3, 5, 7, 9, 11}
		obj := map[Var]float64{}
		row := map[Var]float64{}
		for i := 0; i < 5; i++ {
			vs[i] = m.AddVar(VarSpec{Type: Binary})
			obj[vs[i]] = 1
			row[vs[i]] = coefs[i]
		}
		must(t, m.SetObjective(obj))
		must(t, m.AddConstraint(row, Equal, 17))
		return m
	}
	off, _ := build().Solve(context.Background(), Options{})
	mir, _ := build().Solve(context.Background(), Options{UseMIRCuts: true, MaxCutRounds: 5})
	if off.Status != Optimal || mir.Status != Optimal {
		t.Fatalf("non-optimal: off=%s mir=%s", off.Status, mir.Status)
	}
	if math.Abs(off.Objective-3) > 1e-9 {
		t.Fatalf("baseline obj=%v, expected 3", off.Objective)
	}
	if math.Abs(mir.Objective-off.Objective) > 1e-9 {
		t.Fatalf("MIR drifted objective: off=%v mir=%v", off.Objective, mir.Objective)
	}
}
