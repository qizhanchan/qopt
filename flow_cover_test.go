package qopt

import "testing"

func TestFlowCoverSeparationAndIntegerGridValidity(t *testing.T) {
	m := NewModel(Minimize)
	x1 := m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0), Upper: Float64(4)})
	z1 := m.AddVar(VarSpec{Type: Binary})
	x2 := m.AddVar(VarSpec{Type: Continuous, Lower: Float64(0), Upper: Float64(4)})
	z2 := m.AddVar(VarSpec{Type: Binary})
	must(t, m.AddConstraint(map[Var]float64{x1: 1, x2: 1}, LessOrEqual, 5))
	must(t, m.AddConstraint(map[Var]float64{x1: 1, z1: -4}, LessOrEqual, 0))
	must(t, m.AddConstraint(map[Var]float64{x2: 1, z2: -4}, LessOrEqual, 0))
	rels := detectVariableBoundRelations(m)
	cuts := separateFlowCoverCuts(m, rels, nil, []float64{3, .75, 2, .5}, 1e-9)
	if len(cuts) != 1 {
		t.Fatalf("got %d flow covers, want 1", len(cuts))
	}
	cut := cuts[0]
	if cut.RHS != 3 || cut.Coeffs[x1] != 1 || cut.Coeffs[x2] != 1 || cut.Coeffs[z1] != -1 || cut.Coeffs[z2] != -1 {
		t.Fatalf("unexpected flow-cover inequality: %+v", cut)
	}
	for b1 := 0; b1 <= 1; b1++ {
		for b2 := 0; b2 <= 1; b2++ {
			for a := 0; a <= 4*b1; a++ {
				for b := 0; b <= 4*b2; b++ {
					if a+b > 5 {
						continue
					}
					activity := float64(a + b - b1 - b2)
					if activity > cut.RHS+1e-9 {
						t.Fatalf("cut invalid at x=(%d,%d) z=(%d,%d): activity=%g rhs=%g", a, b, b1, b2, activity, cut.RHS)
					}
				}
			}
		}
	}
}
