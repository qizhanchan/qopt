package main

import (
	"math"
	"testing"

	qopt "github.com/qizhanchan/qopt"
	"github.com/qizhanchan/qopt/bench/internal/problem"
)

func TestToVarSpecLowerUnbounded(t *testing.T) {
	spec, err := toVarSpec(problem.VarSpec{
		Name: "free", Type: "continuous", LowerUnbounded: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if spec.Lower == nil || !math.IsInf(*spec.Lower, -1) {
		t.Fatalf("lower = %v, want -Inf", spec.Lower)
	}
}

func TestToVarSpecRejectsAmbiguousLowerBound(t *testing.T) {
	lower := -1.0
	_, err := toVarSpec(problem.VarSpec{
		Name: "ambiguous", Type: "continuous", Lower: &lower, LowerUnbounded: true,
	})
	if err == nil {
		t.Fatal("expected lower/lower_unbounded conflict")
	}
}

func TestToVarSpecRejectsLowerUnboundedBinary(t *testing.T) {
	_, err := toVarSpec(problem.VarSpec{
		Name: "binary", Type: "binary", LowerUnbounded: true,
	})
	if err == nil {
		t.Fatal("expected lower-unbounded binary rejection")
	}
}

func TestToVarSpecKeepsDefaultLowerBound(t *testing.T) {
	spec, err := toVarSpec(problem.VarSpec{Name: "default", Type: "continuous"})
	if err != nil {
		t.Fatal(err)
	}
	if spec.Type != qopt.Continuous || spec.Lower != nil {
		t.Fatalf("spec = %+v, want continuous with unspecified lower", spec)
	}
}
