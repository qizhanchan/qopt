package qopt

import (
	"context"
	"math"
	"testing"
	"time"
)

// TestSymmetry_IdenticalBinaryColumns is the simplest possible orbit: two
// binary variables with identical objective coefficient, bounds, and
// constraint columns should be verified as a single orbit.
func TestSymmetry_IdenticalBinaryColumns(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	z := m.AddVar(VarSpec{Name: "z", Type: Binary}) // asymmetric control

	if err := m.SetObjective(map[Var]float64{x: 3, y: 3, z: 5}); err != nil {
		t.Fatalf("set objective: %v", err)
	}
	if err := m.AddConstraint(map[Var]float64{x: 1, y: 1, z: 2}, LessOrEqual, 3); err != nil {
		t.Fatalf("add constraint: %v", err)
	}

	orbits := detectSymmetryOrbits(m)
	if len(orbits) != 1 {
		t.Fatalf("expected 1 orbit, got %d: %v", len(orbits), orbits)
	}
	if len(orbits[0]) != 2 || orbits[0][0] != x || orbits[0][1] != y {
		t.Fatalf("expected orbit {x,y}, got %v", orbits[0])
	}
}

// TestSymmetry_NoFalsePositiveOnAsymmetricColumns guards against verifying a
// transposition that isn't a real automorphism: two vars share every
// WL-refined color-invariant feature except one row's coefficient.
func TestSymmetry_NoFalsePositiveOnAsymmetricColumns(t *testing.T) {
	m := NewModel(Minimize)
	a := m.AddVar(VarSpec{Name: "a", Type: Integer, Lower: Float64(0), Upper: Float64(10)})
	b := m.AddVar(VarSpec{Name: "b", Type: Integer, Lower: Float64(0), Upper: Float64(10)})

	if err := m.SetObjective(map[Var]float64{a: 1, b: 1}); err != nil {
		t.Fatalf("set objective: %v", err)
	}
	if err := m.AddConstraint(map[Var]float64{a: 2, b: 3}, LessOrEqual, 10); err != nil {
		t.Fatalf("add constraint: %v", err)
	}

	orbits := detectSymmetryOrbits(m)
	if len(orbits) != 0 {
		t.Fatalf("expected no orbits for asymmetric columns, got %v", orbits)
	}
}

func TestSymmetry_ExactFloatVerification(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Binary})
	y := m.AddVar(VarSpec{Name: "y", Type: Binary})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, LessOrEqual, 1))
	must(t, m.AddConstraint(map[Var]float64{y: math.Nextafter(1, 2)}, LessOrEqual, 1))
	if orbits := detectSymmetryOrbits(m); len(orbits) != 0 {
		t.Fatalf("near-equal but distinct coefficients produced false symmetry: %v", orbits)
	}
}

// TestSymmetry_MarkshareHasNoOrbits regression-guards the 2026-06-24 symmetry
// survey finding: markshare_4_0's integer variables have zero verified
// automorphisms (its hardness is number-theoretic, not symmetry), so
// DetectSymmetry must not add any breaking constraints — and therefore must
// not change the proven optimum.
func TestSymmetry_MarkshareHasNoOrbits(t *testing.T) {
	p := loadProblem(t, "bench/problems/miplib/markshare_4_0.json")
	m := modelFromProblem(t, p)
	orbits := detectSymmetryOrbits(m)
	if len(orbits) != 0 {
		t.Fatalf("expected markshare_4_0 to have zero verified orbits, got %v", orbits)
	}
}

// TestSymmetry_Assign158IsBlockNotColumnSymmetry documents a real limitation:
// the WL color refinement alone lands 23 candidate classes on assign1-5-8
// (sizes up to 16 — consistent with the 2026-06-24 survey's orbit count), but
// every single one fails pairwise-transposition verification. This confirms
// assign1-5-8's symmetry is *block* symmetry — permuting an assignment-problem
// index (e.g. "worker") moves many variables' columns together, which a
// single-column transposition test cannot detect or verify. Capturing it
// needs a real automorphism-group search (individualization-refinement, the
// P1 sub-goal 2 "saucy/SASSY-style refinement" note), not yet implemented.
// This test pins the current (narrower) scope so a future change to either
// the verifier or this expectation is a deliberate decision, not a silent
// regression.
func TestSymmetry_Assign158IsBlockNotColumnSymmetry(t *testing.T) {
	p := loadProblem(t, "bench/problems/miplib/assign1-5-8.json")
	m := modelFromProblem(t, p)
	orbits := detectSymmetryOrbits(m)
	if len(orbits) != 0 {
		t.Fatalf("column-transposition verifier unexpectedly found orbits on assign1-5-8 (%v) — "+
			"if this is a genuine new capability, update this test's expectation and comment", orbits)
	}
}

// TestSymmetry_DoesNotChangeOptimum runs a small MIP with known symmetric
// structure both with and without DetectSymmetry and checks the optimal
// objective is unchanged — the breaking constraints must only remove
// duplicate symmetric solutions, never cut off the true optimum.
func TestSymmetry_DoesNotChangeOptimum(t *testing.T) {
	build := func() *Model {
		m := NewModel(Maximize)
		vs := make([]Var, 4)
		coeffs := make(map[Var]float64, 4)
		for i := range vs {
			vs[i] = m.AddVar(VarSpec{Type: Binary})
			coeffs[vs[i]] = 1
		}
		if err := m.SetObjective(coeffs); err != nil {
			t.Fatalf("set objective: %v", err)
		}
		if err := m.AddConstraint(coeffs, LessOrEqual, 2); err != nil {
			t.Fatalf("add constraint: %v", err)
		}
		return m
	}

	base, err := build().Solve(context.Background(), Options{TimeLimit: 2 * time.Second})
	if err != nil {
		t.Fatalf("solve base: %v", err)
	}
	withSym, err := build().Solve(context.Background(), Options{TimeLimit: 2 * time.Second, DetectSymmetry: true})
	if err != nil {
		t.Fatalf("solve with symmetry: %v", err)
	}
	if base.Objective != withSym.Objective {
		t.Fatalf("objective changed under DetectSymmetry: base=%v withSym=%v", base.Objective, withSym.Objective)
	}
}
