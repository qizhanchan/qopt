package qopt

import (
	"context"
	"math"
	"testing"
)

func TestFreeVariableWithImpliedFiniteLowerBound(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{
		Name: "x", Type: Continuous, Lower: Float64(math.Inf(-1)),
	})
	if err := m.SetObjective(map[Var]float64{x: 1}); err != nil {
		t.Fatal(err)
	}
	if err := m.AddConstraint(
		map[Var]float64{x: 1}, GreaterOrEqual, -2,
	); err != nil {
		t.Fatal(err)
	}

	result, err := m.Solve(context.Background(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Status != Optimal {
		t.Fatalf("status = %v, want optimal", result.Status)
	}
	if math.Abs(result.Objective-(-2)) > 1e-8 || math.Abs(result.Values[x]-(-2)) > 1e-8 {
		t.Fatalf("objective/value = %g/%g, want -2/-2", result.Objective, result.Values[x])
	}
}
