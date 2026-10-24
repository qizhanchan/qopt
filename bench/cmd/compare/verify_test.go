package main

import (
	"testing"

	"github.com/qizhanchan/qopt/bench/internal/problem"
)

func ptr(v float64) *float64 { return &v }

func newProb() *problem.Problem {
	return &problem.Problem{
		Name:      "p",
		Direction: "min",
		Vars: []problem.VarSpec{
			{Name: "x", Type: "continuous", Lower: ptr(0), Upper: ptr(10)},
			{Name: "y", Type: "integer", Lower: ptr(0), Upper: ptr(5)},
		},
		Objective: map[string]float64{"x": 1, "y": 1},
		Constraints: []problem.Constraint{
			{Name: "eq", Coeffs: map[string]float64{"x": 1, "y": 1}, Sense: "=", RHS: 3},
			{Name: "le", Coeffs: map[string]float64{"x": 1}, Sense: "<=", RHS: 2},
		},
	}
}

func TestVerifyIncumbent_Feasible(t *testing.T) {
	p := newProb()
	values := map[string]float64{"x": 1.5, "y": 1.5}
	values["y"] = 2 // ensure integral
	values["x"] = 1
	// x=1, y=2: eq=3 ✓, le: 1<=2 ✓, integer y ✓, bounds ✓
	if v := verifyIncumbent(p, values, 1e-6); v != nil {
		t.Fatalf("expected feasible, got %s", v)
	}
}

func TestVerifyIncumbent_ConstraintBreach(t *testing.T) {
	p := newProb()
	// x=2, y=2 → eq lhs=4, RHS=3, breach=1
	values := map[string]float64{"x": 2, "y": 2}
	v := verifyIncumbent(p, values, 1e-6)
	if v == nil {
		t.Fatal("expected violation, got nil")
	}
	if v.kind != "constraint" || v.subject != "eq" {
		t.Fatalf("expected constraint eq, got %+v", v)
	}
	if v.amount < 0.999 || v.amount > 1.001 {
		t.Fatalf("expected breach≈1, got %g", v.amount)
	}
}

func TestVerifyIncumbent_BoundBreach(t *testing.T) {
	p := newProb()
	// x=12 → upper(10) breached by 2; eq also breached, but bound breach is reported
	// as long as it's the largest; here eq breach = |12+0-3|=9 > 2, so eq wins.
	values := map[string]float64{"x": 12, "y": 0}
	v := verifyIncumbent(p, values, 1e-6)
	if v == nil || v.kind != "constraint" {
		t.Fatalf("expected constraint breach, got %+v", v)
	}
}

func TestVerifyIncumbent_IntegralityBreach(t *testing.T) {
	p := newProb()
	// x=1.5, y=1.5: eq satisfied, le satisfied, but y is integer-typed
	values := map[string]float64{"x": 1.5, "y": 1.5}
	v := verifyIncumbent(p, values, 1e-6)
	if v == nil || v.kind != "integrality" || v.subject != "y" {
		t.Fatalf("expected integrality breach on y, got %+v", v)
	}
}

func TestVerifyIncumbent_TolSlack(t *testing.T) {
	p := newProb()
	// eq off by 1e-9 — within 1e-6 tol, should still be feasible
	values := map[string]float64{"x": 1 + 1e-9, "y": 2}
	if v := verifyIncumbent(p, values, 1e-6); v != nil {
		t.Fatalf("expected within-tol pass, got %s", v)
	}
}
