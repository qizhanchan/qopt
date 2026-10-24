package qopt

import (
	"math"
	"sort"
)

const rootProbePropagationPasses = 5

type variableBoundRelation struct {
	SourceRow  int
	Bounded    int
	Controller int
	Slope      float64
	Intercept  float64
	Upper      bool
}

type boundImplication struct {
	Binary    int
	Value     int
	Variable  int
	Bound     float64
	Upper     bool
	SourceRow int
}

type implicationProbeResult struct {
	Relations       []variableBoundRelation
	Implications    []boundImplication
	ProbeSides      int
	InfeasibleSides int
	GlobalLower     []float64
	GlobalUpper     []float64
	GlobalTightened int
}

func implicationInferenceCounts(nvars int, implications []boundImplication) ([]int, []int) {
	down, up := make([]int, nvars), make([]int, nvars)
	for _, imp := range implications {
		if imp.Binary < 0 || imp.Binary >= nvars || imp.Variable == imp.Binary {
			continue
		}
		if imp.Value == 0 {
			down[imp.Binary]++
		} else if imp.Value == 1 {
			up[imp.Binary]++
		}
	}
	return down, up
}

// detectVariableBoundRelations normalizes every finite two-variable
// inequality into x <= a*y+b or x >= a*y+b. Both directions are retained;
// SourceRow is the exact original row used to justify the relation.
func detectVariableBoundRelations(m *Model) []variableBoundRelation {
	const tol = 1e-12
	var out []variableBoundRelation
	for r, c := range m.constraints {
		if c.Sense == Equal {
			continue
		}
		entries := m.sortedRows()[r]
		if len(entries) != 2 {
			continue
		}
		sign := 1.0
		if c.Sense == GreaterOrEqual {
			sign = -1
		}
		rhs := sign * c.RHS
		for i := 0; i < 2; i++ {
			x, y := entries[i], entries[1-i]
			cx, cy := sign*x.coef, sign*y.coef
			if math.Abs(cx) <= tol || math.IsNaN(cx) || math.IsNaN(cy) {
				continue
			}
			rel := variableBoundRelation{
				SourceRow: r, Bounded: x.v, Controller: y.v,
				Slope: -cy / cx, Intercept: rhs / cx, Upper: cx > 0,
			}
			if !math.IsNaN(rel.Slope) && !math.IsInf(rel.Slope, 0) &&
				!math.IsNaN(rel.Intercept) && !math.IsInf(rel.Intercept, 0) {
				out = append(out, rel)
			}
		}
	}
	return out
}

// probeBinaryImplications fixes every unfixed binary to each side and runs
// only the event-driven domain propagator. Side-specific consequences remain
// implications; only consequences valid on both sides are returned as global
// bounds. No LP solve is performed here.
func probeBinaryImplications(m *Model, tol float64) implicationProbeResult {
	lower, upper := effectiveBounds(m, nil)
	baseLower := append([]float64(nil), lower...)
	baseUpper := append([]float64(nil), upper...)
	base := newDomainState(m, baseLower, baseUpper, tol)
	if !base.propagate(rootProbePropagationPasses) {
		return implicationProbeResult{Relations: detectVariableBoundRelations(m)}
	}

	result := implicationProbeResult{
		Relations:   detectVariableBoundRelations(m),
		GlobalLower: append([]float64(nil), base.lower...),
		GlobalUpper: append([]float64(nil), base.upper...),
	}
	for z, spec := range m.vars {
		if spec.Type != Binary || base.lower[z] >= 0.5 || base.upper[z] <= 0.5 {
			continue
		}
		type sideResult struct {
			feasible                 bool
			lower, upper             []float64
			lowerReason, upperReason []boundReason
		}
		var sides [2]sideResult
		for value := 0; value <= 1; value++ {
			lo := append([]float64(nil), base.lower...)
			hi := append([]float64(nil), base.upper...)
			lo[z], hi[z] = float64(value), float64(value)
			d := newDomainState(m, lo, hi, tol)
			result.ProbeSides++
			sides[value] = sideResult{feasible: d.propagate(rootProbePropagationPasses), lower: d.lower, upper: d.upper, lowerReason: d.lowerReason, upperReason: d.upperReason}
			if !sides[value].feasible {
				result.InfeasibleSides++
				continue
			}
			for v := range m.vars {
				if v == z {
					continue
				}
				if d.lower[v] > base.lower[v]+tol {
					result.Implications = append(result.Implications, boundImplication{Binary: z, Value: value, Variable: v, Bound: d.lower[v], SourceRow: d.lowerReason[v].Row})
				}
				if d.upper[v] < base.upper[v]-tol {
					result.Implications = append(result.Implications, boundImplication{Binary: z, Value: value, Variable: v, Bound: d.upper[v], Upper: true, SourceRow: d.upperReason[v].Row})
				}
			}
		}

		if !sides[0].feasible && sides[1].feasible {
			result.GlobalLower[z] = 1
			result.GlobalUpper[z] = 1
		} else if sides[0].feasible && !sides[1].feasible {
			result.GlobalLower[z] = 0
			result.GlobalUpper[z] = 0
		} else if sides[0].feasible && sides[1].feasible {
			for v := range m.vars {
				if b := math.Min(sides[0].lower[v], sides[1].lower[v]); b > result.GlobalLower[v]+tol {
					result.GlobalLower[v] = b
				}
				if b := math.Max(sides[0].upper[v], sides[1].upper[v]); b < result.GlobalUpper[v]-tol {
					result.GlobalUpper[v] = b
				}
			}
		}
	}
	for v := range m.vars {
		if result.GlobalLower[v] > base.lower[v]+tol || result.GlobalUpper[v] < base.upper[v]-tol {
			result.GlobalTightened++
		}
	}
	return result
}

// separateImpliedBoundCuts takes the disjunction z=0 or z=1 and interpolates
// the propagated bound on a continuous variable. For lower bounds this is
// x >= l0+(l1-l0)z; upper bounds are analogous. The inequality is valid for
// both integer sides and can cut the fractional root relaxation.
func separateImpliedBoundCuts(m *Model, implications []boundImplication, values []float64, tol float64) []Constraint {
	if len(values) != len(m.vars) {
		return nil
	}
	type key struct {
		z, v  int
		upper bool
	}
	type sides struct {
		bound  [2]float64
		have   [2]bool
		source [2]int
	}
	byKey := make(map[key]sides)
	for _, imp := range implications {
		if imp.Binary < 0 || imp.Binary >= len(m.vars) || imp.Variable < 0 || imp.Variable >= len(m.vars) ||
			imp.Value < 0 || imp.Value > 1 || m.vars[imp.Variable].Type != Continuous {
			continue
		}
		k := key{z: imp.Binary, v: imp.Variable, upper: imp.Upper}
		s := byKey[k]
		if !s.have[imp.Value] || (!imp.Upper && imp.Bound > s.bound[imp.Value]) || (imp.Upper && imp.Bound < s.bound[imp.Value]) {
			s.bound[imp.Value], s.have[imp.Value], s.source[imp.Value] = imp.Bound, true, imp.SourceRow
		}
		byKey[k] = s
	}
	var out []Constraint
	seen := make(map[string]struct{})
	for k, s := range byKey {
		for side := 0; side <= 1; side++ {
			if s.have[side] {
				continue
			}
			if k.upper {
				s.bound[side] = *m.vars[k.v].Upper
			} else {
				s.bound[side] = *m.vars[k.v].Lower
			}
		}
		if math.IsInf(s.bound[0], 0) || math.IsInf(s.bound[1], 0) || math.IsNaN(s.bound[0]) || math.IsNaN(s.bound[1]) {
			continue
		}
		delta := s.bound[1] - s.bound[0]
		if math.Abs(delta) <= tol {
			continue
		}
		maxCoef := math.Max(1, math.Abs(delta))
		minCoef := math.Min(1, math.Abs(delta))
		if minCoef == 0 || maxCoef/minCoef > 1e8 {
			continue
		}
		activity := values[k.v] - delta*values[k.z]
		violated := (!k.upper && activity < s.bound[0]-tol) || (k.upper && activity > s.bound[0]+tol)
		if !violated {
			continue
		}
		sense := GreaterOrEqual
		if k.upper {
			sense = LessOrEqual
		}
		cut := Constraint{
			Name:   "structure_implied_bound",
			Coeffs: map[Var]float64{Var(k.v): 1, Var(k.z): -delta},
			Sense:  sense, RHS: s.bound[0],
		}
		sig := rowSignaturePermutation(cut, nil)
		if _, ok := seen[sig]; ok {
			continue
		}
		seen[sig] = struct{}{}
		out = append(out, cut)
	}
	sort.Slice(out, func(i, j int) bool {
		return rowSignaturePermutation(out[i], nil) < rowSignaturePermutation(out[j], nil)
	})
	return out
}
