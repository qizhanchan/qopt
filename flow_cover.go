package qopt

import (
	"fmt"
	"math"
	"sort"
)

type binaryVUB struct {
	x, z       int
	u          float64
	complement bool
}

func collectBinaryVUBs(m *Model, relations []variableBoundRelation, implications []boundImplication, tol float64) map[int]binaryVUB {
	out := make(map[int]binaryVUB)
	keep := func(v binaryVUB) {
		if v.u <= tol || math.IsInf(v.u, 0) || math.IsNaN(v.u) || *m.vars[v.x].Lower < -tol {
			return
		}
		old, ok := out[v.x]
		if !ok || v.u < old.u {
			out[v.x] = v
		}
	}
	for _, rel := range relations {
		if rel.Upper && rel.Bounded >= 0 && rel.Controller >= 0 &&
			m.vars[rel.Bounded].Type == Continuous && m.vars[rel.Controller].Type == Binary &&
			math.Abs(rel.Intercept) <= tol && rel.Slope > tol {
			keep(binaryVUB{x: rel.Bounded, z: rel.Controller, u: rel.Slope})
		}
	}
	type key struct{ z, x, side int }
	upper := make(map[key]float64)
	for _, imp := range implications {
		if !imp.Upper || m.vars[imp.Variable].Type != Continuous {
			continue
		}
		k := key{imp.Binary, imp.Variable, imp.Value}
		if old, ok := upper[k]; !ok || imp.Bound < old {
			upper[k] = imp.Bound
		}
	}
	for k, b := range upper {
		if math.Abs(b) > tol {
			continue
		}
		other, ok := upper[key{k.z, k.x, 1 - k.side}]
		if !ok {
			other = *m.vars[k.x].Upper
		}
		if k.side == 0 {
			keep(binaryVUB{x: k.x, z: k.z, u: other})
		} else {
			keep(binaryVUB{x: k.x, z: k.z, u: other, complement: true})
		}
	}
	return out
}

type flowCoverItem struct {
	vub      binaryVUB
	rowCoef  float64
	capacity float64
	flow     float64
	open     float64
}

// separateFlowCoverCuts implements the unlifted single-node fixed-charge
// flow-cover inequality. Non-flow row terms are relaxed to their finite
// minimum activity, so every emitted inequality remains valid for the full
// model rather than assuming an exact network formulation.
func separateFlowCoverCuts(m *Model, relations []variableBoundRelation, implications []boundImplication, values []float64, tol float64) []Constraint {
	if len(values) != len(m.vars) {
		return nil
	}
	vubs := collectBinaryVUBs(m, relations, implications, tol)
	if len(vubs) < 2 {
		return nil
	}
	var cuts []Constraint
	seen := make(map[string]struct{})
	for r, row := range m.constraints {
		if row.Sense == Equal {
			continue
		}
		sign := 1.0
		if row.Sense == GreaterOrEqual {
			sign = -1
		}
		d := sign * row.RHS
		var items []flowCoverItem
		valid := true
		for _, e := range m.sortedRows()[r] {
			a := sign * e.coef
			if vub, ok := vubs[e.v]; ok && a > tol {
				y := values[vub.z]
				if vub.complement {
					y = 1 - y
				}
				items = append(items, flowCoverItem{vub: vub, rowCoef: a, capacity: a * vub.u, flow: a * values[e.v], open: y})
				continue
			}
			lo, hi := a**m.vars[e.v].Lower, a**m.vars[e.v].Upper
			if lo > hi {
				lo = hi
			}
			if math.IsInf(lo, -1) || math.IsNaN(lo) {
				valid = false
				break
			}
			d -= lo
		}
		if !valid || len(items) < 2 || d <= tol {
			continue
		}
		// Prefer arcs carrying fractional flow, then large capacities. This is
		// deterministic and produces one minimal cover per source row.
		sort.Slice(items, func(i, j int) bool {
			if items[i].flow != items[j].flow {
				return items[i].flow > items[j].flow
			}
			if items[i].capacity != items[j].capacity {
				return items[i].capacity > items[j].capacity
			}
			return items[i].vub.x < items[j].vub.x
		})
		var cover []flowCoverItem
		sumU := 0.0
		for _, item := range items {
			cover = append(cover, item)
			sumU += item.capacity
			if sumU > d+tol {
				break
			}
		}
		if sumU <= d+tol {
			continue
		}
		for i := len(cover) - 1; i >= 0; i-- {
			if sumU-cover[i].capacity > d+tol {
				sumU -= cover[i].capacity
				cover = append(cover[:i], cover[i+1:]...)
			}
		}
		lambda := sumU - d
		coeffs := make(map[Var]float64, 2*len(cover))
		rhs := d
		activity := 0.0
		for _, item := range cover {
			alpha := math.Max(0, item.capacity-lambda)
			coeffs[Var(item.vub.x)] += item.rowCoef
			if item.vub.complement {
				coeffs[Var(item.vub.z)] += alpha
			} else {
				coeffs[Var(item.vub.z)] -= alpha
				rhs -= alpha
			}
		}
		for v, a := range coeffs {
			activity += a * values[v]
		}
		if activity <= rhs+tol {
			continue
		}
		cut := Constraint{Name: fmt.Sprintf("structure_flow_cover_%d", r), Coeffs: coeffs, Sense: LessOrEqual, RHS: rhs}
		sig := rowSignaturePermutation(cut, nil)
		if _, ok := seen[sig]; ok {
			continue
		}
		seen[sig] = struct{}{}
		cuts = append(cuts, cut)
	}
	return cuts
}
