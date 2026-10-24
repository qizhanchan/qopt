package qopt

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// detectBinaryRowConflicts finds pairwise binary conflicts implied by one
// constraint row plus the variables' global bounds. For each row it computes
// the minimum and maximum attainable activity. It then fixes each binary pair
// to one and asks whether the row can still reach its RHS. If not,
// x_i+x_j<=1 is globally valid.
//
// This activity formulation handles <=, >=, and equality rows, mixed
// continuous/integer variables, negative coefficients, and big-M rows. Cuts
// already implied by the source row's LP relaxation are skipped; in
// particular, an explicit k-variable set-packing row is not expanded into
// O(k^2) weaker pair rows.
func detectBinaryRowConflicts(m *Model) []Constraint {
	type pair struct{ a, b int }
	seen := make(map[pair]struct{})
	var out []Constraint
	const tol = 1e-9
	const maxConflicts = 5000

	for r, row := range m.constraints {
		entries := m.sortedRows()[r]
		vars := make([]int, 0, len(row.Coeffs))
		minContrib := make(map[int]float64, len(row.Coeffs))
		maxContrib := make(map[int]float64, len(row.Coeffs))
		minActivity, maxActivity := 0.0, 0.0
		valid := true
		for _, e := range entries {
			if math.IsNaN(e.coef) || math.Abs(e.coef) <= tol {
				if math.IsNaN(e.coef) {
					valid = false
				}
				continue
			}
			spec := m.vars[e.v]
			lo, hi := e.coef**spec.Lower, e.coef**spec.Upper
			if lo > hi {
				lo, hi = hi, lo
			}
			minActivity += lo
			maxActivity += hi
			if spec.Type == Binary && *spec.Lower < 0.5 && *spec.Upper > 0.5 {
				vars = append(vars, e.v)
				minContrib[e.v] = lo
				maxContrib[e.v] = hi
			}
		}
		if !valid || len(vars) < 2 {
			continue
		}
		// sortedRows already gives deterministic ascending variable order.
		for i := 0; i < len(vars); i++ {
			for j := i + 1; j < len(vars); j++ {
				vi, vj := vars[i], vars[j]
				ai, aj := row.Coeffs[Var(vi)], row.Coeffs[Var(vj)]
				conflict, strengthens := false, false

				if row.Sense == LessOrEqual || row.Sense == Equal {
					if !math.IsInf(minActivity, -1) {
						di := ai - minContrib[vi]
						dj := aj - minContrib[vj]
						capacity := row.RHS - minActivity
						if di+dj > capacity+tol {
							conflict = true
							// The pair cut strengthens the single-row LP only when
							// a fractional sum above one can still fit.
							strengthens = math.Min(di, dj) < capacity-tol
						}
					}
				}
				if row.Sense == GreaterOrEqual || row.Sense == Equal {
					if !math.IsInf(maxActivity, 1) {
						li := maxContrib[vi] - ai
						lj := maxContrib[vj] - aj
						slack := maxActivity - row.RHS
						if li+lj > slack+tol {
							conflict = true
							strengthens = strengthens || math.Min(li, lj) < slack-tol
						}
					}
				}
				if !conflict || !strengthens {
					continue
				}
				p := pair{vi, vj}
				if _, ok := seen[p]; ok {
					continue
				}
				seen[p] = struct{}{}
				out = append(out, Constraint{
					Name:   fmt.Sprintf("structure_pack_%d_%d", p.a, p.b),
					Coeffs: map[Var]float64{Var(p.a): 1, Var(p.b): 1},
					Sense:  LessOrEqual,
					RHS:    1,
				})
				if len(out) >= maxConflicts {
					return out
				}
			}
		}
	}
	return out
}

type coverLiteral struct {
	v          Var
	complement bool
	weight     float64
}

type bigMTightening struct {
	row int
	cut Constraint
}

// detectSingleBinaryBigMTightenings strengthens rows containing exactly one
// unfixed binary variable. After normalizing to rest+a*z<=b, the two integer
// branches have limits b (z=0) and b-a (z=1). Either limit can be capped by
// the maximum activity U of rest under propagated global bounds. Interpolating
// the capped branch limits gives a new linear row that is identical on every
// feasible binary point and dominates the original at fractional z.
//
// A replacement only happens when a capped branch limit improves by at least
// bigMMinRelImprovement of the row's limit scale. Any replacement perturbs
// the LP vertex and therefore the whole search trajectory, and on instances
// whose incumbent discovery has no slack (ic97_potential finds its first
// incumbent at 2/3 of a 5-min budget) a near-worthless tightening can cost
// the incumbent outright: ic97's single candidate improves its row by 1.7%
// (big-M 60→59) and taking it measured as incumbent 4276 → none, while
// glass4's 48 genuine big-M rows all improve by 89% or more.
const bigMMinRelImprovement = 0.05

func detectSingleBinaryBigMTightenings(m *Model) []bigMTightening {
	const tol = 1e-9
	// Use only explicit global bounds here. Bounds propagated from the same
	// row being tightened would be circular evidence and could make the
	// replacement invalid. Reason-independent propagated bounds can be added
	// later once presolve exposes dependency information.
	lower, upper := effectiveBounds(m, nil)
	var out []bigMTightening
	for r, row := range m.constraints {
		if row.Sense == Equal {
			continue
		}
		sign := 1.0
		if row.Sense == GreaterOrEqual {
			sign = -1
		}
		binary := -1
		maxRest := 0.0
		valid := true
		for _, e := range m.sortedRows()[r] {
			a := sign * e.coef
			if m.vars[e.v].Type == Binary && lower[e.v] < 0.5 && upper[e.v] > 0.5 && math.Abs(a) > tol {
				if binary >= 0 {
					valid = false
					break
				}
				binary = e.v
				continue
			}
			lo, hi := a*lower[e.v], a*upper[e.v]
			if lo > hi {
				lo, hi = hi, lo
			}
			if math.IsInf(hi, 1) || math.IsNaN(hi) {
				valid = false
				break
			}
			maxRest += hi
		}
		if !valid || binary < 0 || math.IsInf(maxRest, 1) {
			continue
		}
		b0 := sign * row.RHS
		a := sign * row.Coeffs[Var(binary)]
		b1 := b0 - a
		newB0 := math.Min(b0, maxRest)
		newB1 := math.Min(b1, maxRest)
		if newB0 == maxRest && newB1 == maxRest {
			continue // the whole row is box-redundant
		}
		newA := newB0 - newB1
		if math.Abs(newB0-b0) <= tol*math.Max(1, math.Abs(b0)) &&
			math.Abs(newA-a) <= tol*math.Max(1, math.Abs(a)) {
			continue
		}
		improvement := math.Max(b0-newB0, b1-newB1)
		limitScale := math.Max(math.Max(math.Abs(b0), math.Abs(b1)), 1)
		if improvement < bigMMinRelImprovement*limitScale {
			continue
		}
		coeffs := make(map[Var]float64, len(row.Coeffs))
		for v, coef := range row.Coeffs {
			if int(v) == binary {
				if math.Abs(newA) > tol {
					coeffs[v] = newA
				}
			} else if math.Abs(sign*coef) > tol {
				coeffs[v] = sign * coef
			}
		}
		out = append(out, bigMTightening{
			row: r,
			cut: Constraint{
				Name:   fmt.Sprintf("structure_bigm_%d", r),
				Coeffs: coeffs,
				Sense:  LessOrEqual,
				RHS:    newB0,
			},
		})
	}
	return out
}

// detectKnapsackCoverCuts derives one deterministic minimal cover from each
// eligible row orientation. Negative binary coefficients are represented by
// complemented literals (1-x), >= rows are multiplied by -1, and bounded
// nonbinary terms are fixed to their most favorable activity. Therefore every
// emitted cover remains valid for the original mixed row, not only for a pure
// positive binary knapsack.
func detectKnapsackCoverCuts(m *Model) []Constraint {
	return separateKnapsackCoverCuts(m, nil)
}

func separateKnapsackCoverCuts(m *Model, values []float64) []Constraint {
	const tol = 1e-9
	const maxCovers = 1000
	seen := make(map[string]bool)
	var out []Constraint
	for r, row := range m.constraints {
		signs := []float64{1}
		if row.Sense == GreaterOrEqual {
			signs[0] = -1
		} else if row.Sense == Equal {
			signs = []float64{1, -1}
		}
		for _, sign := range signs {
			capacity := sign * row.RHS
			lits := make([]coverLiteral, 0, len(row.Coeffs))
			valid := true
			for _, e := range m.sortedRows()[r] {
				a := sign * e.coef
				if math.IsNaN(a) {
					valid = false
					break
				}
				spec := m.vars[e.v]
				if spec.Type == Binary && *spec.Lower < 0.5 && *spec.Upper > 0.5 {
					if a > tol {
						lits = append(lits, coverLiteral{v: Var(e.v), weight: a})
					} else if a < -tol {
						// a*x = a + (-a)*(1-x).
						capacity -= a
						lits = append(lits, coverLiteral{v: Var(e.v), complement: true, weight: -a})
					}
					continue
				}
				lo, hi := a**spec.Lower, a**spec.Upper
				if lo > hi {
					lo, hi = hi, lo
				}
				if math.IsInf(lo, -1) || math.IsNaN(lo) {
					valid = false
					break
				}
				capacity -= lo
			}
			if !valid || capacity < -tol || len(lits) < 2 {
				continue
			}
			sort.Slice(lits, func(i, j int) bool {
				if len(values) == len(m.vars) {
					vi, vj := values[lits[i].v], values[lits[j].v]
					if lits[i].complement {
						vi = 1 - vi
					}
					if lits[j].complement {
						vj = 1 - vj
					}
					if vi != vj {
						return vi > vj
					}
				}
				if lits[i].weight != lits[j].weight {
					return lits[i].weight > lits[j].weight
				}
				if lits[i].v != lits[j].v {
					return lits[i].v < lits[j].v
				}
				return !lits[i].complement && lits[j].complement
			})
			sum := 0.0
			coverEnd := 0
			for coverEnd < len(lits) && sum <= capacity+tol {
				sum += lits[coverEnd].weight
				coverEnd++
			}
			if sum <= capacity+tol || coverEnd < 2 {
				continue
			}
			cover := append([]coverLiteral(nil), lits[:coverEnd]...)
			// Make the cover inclusion-minimal. This improves efficacy and
			// avoids carrying literals that do not participate in infeasibility.
			for i := len(cover) - 1; i >= 0; i-- {
				if sum-cover[i].weight > capacity+tol {
					sum -= cover[i].weight
					cover = append(cover[:i], cover[i+1:]...)
				}
			}
			if len(cover) < 2 {
				continue
			}
			// If the cheapest k-1 literals already consume the capacity, the
			// source LP row itself implies the cover inequality.
			ascending := append([]coverLiteral(nil), cover...)
			sort.Slice(ascending, func(i, j int) bool { return ascending[i].weight < ascending[j].weight })
			lpCost := 0.0
			for i := 0; i+1 < len(ascending); i++ {
				lpCost += ascending[i].weight
			}
			if lpCost >= capacity-tol {
				continue
			}

			makeCut := func(extra *coverLiteral, alpha float64) Constraint {
				coeffs := make(map[Var]float64, len(cover)+1)
				rhs := float64(len(cover) - 1)
				for _, lit := range cover {
					if lit.complement {
						coeffs[lit.v]--
						rhs--
					} else {
						coeffs[lit.v]++
					}
				}
				name := fmt.Sprintf("structure_cover_%d", r)
				if extra != nil && alpha > 0 {
					name = fmt.Sprintf("structure_lifted_cover_%d_%d", r, extra.v)
					if extra.complement {
						coeffs[extra.v] -= alpha
						rhs -= alpha
					} else {
						coeffs[extra.v] += alpha
					}
				}
				return Constraint{Name: name, Coeffs: coeffs, Sense: LessOrEqual, RHS: rhs}
			}
			emit := func(cut Constraint) bool {
				if len(values) == len(m.vars) && cutEfficacy(cut, values, coeffNorm(cut.Coeffs)) <= 1e-6 {
					return false
				}
				key := rowSignaturePermutation(cut, nil)
				if seen[key] {
					return false
				}
				seen[key] = true
				out = append(out, cut)
				return len(out) >= maxCovers
			}
			if emit(makeCut(nil, 0)) {
				return out
			}
			if len(values) == len(m.vars) {
				inCover := make(map[Var]bool, len(cover))
				for _, lit := range cover {
					inCover[lit.v] = true
				}
				for i := range lits {
					outside := lits[i]
					if inCover[outside.v] {
						continue
					}
					residual := capacity - outside.weight
					maxCount, used := 0, 0.0
					for _, lit := range ascending {
						if used+lit.weight <= residual+tol {
							used += lit.weight
							maxCount++
						}
					}
					alpha := float64(len(cover) - 1 - maxCount)
					if alpha > 0 && emit(makeCut(&outside, alpha)) {
						return out
					}
				}
			}
		}
	}
	return out
}

// detectPartitioningOrbitopes recognizes rectangular blocks of disjoint
// exact-one binary rows and verifies simultaneous column transpositions as
// exact automorphisms of the complete model. A verified orbit of columns can
// be lexicographically ordered. For a partitioning matrix, the corresponding
// triangular zero pattern x[row_i,column_j]=0 for j>i is a compact orbitope
// fixing that removes symmetric column permutations without adding dense rows.
func detectPartitioningOrbitopes(m *Model) []Constraint {
	byWidth := make(map[int][][]Var)
	usedByWidth := make(map[int]map[Var]bool)
	for _, row := range m.constraints {
		if row.Sense != Equal || row.RHS != 1 || len(row.Coeffs) < 2 {
			continue
		}
		group := make([]Var, 0, len(row.Coeffs))
		valid := true
		for v, a := range row.Coeffs {
			if a != 1 || m.vars[v].Type != Binary {
				valid = false
				break
			}
			group = append(group, v)
		}
		if !valid {
			continue
		}
		sort.Slice(group, func(i, j int) bool { return group[i] < group[j] })
		width := len(group)
		if usedByWidth[width] == nil {
			usedByWidth[width] = make(map[Var]bool)
		}
		for _, v := range group {
			if usedByWidth[width][v] {
				valid = false
				break
			}
		}
		if !valid {
			continue
		}
		for _, v := range group {
			usedByWidth[width][v] = true
		}
		byWidth[width] = append(byWidth[width], group)
	}

	widths := make([]int, 0, len(byWidth))
	for width := range byWidth {
		widths = append(widths, width)
	}
	sort.Ints(widths)
	var out []Constraint
	for _, width := range widths {
		rows := byWidth[width]
		if len(rows) < 2 {
			continue
		}
		uf := newUnionFind(width)
		for a := 0; a < width; a++ {
			for b := a + 1; b < width; b++ {
				perm := make([]int, len(m.vars))
				for v := range perm {
					perm[v] = v
				}
				for _, row := range rows {
					perm[row[a]], perm[row[b]] = int(row[b]), int(row[a])
				}
				if verifyVariablePermutation(m, perm) {
					uf.union(a, b)
				}
			}
		}
		orbits := make(map[int][]int)
		for col := 0; col < width; col++ {
			orbits[uf.find(col)] = append(orbits[uf.find(col)], col)
		}
		roots := make([]int, 0, len(orbits))
		for root, orbit := range orbits {
			if len(orbit) > 1 {
				roots = append(roots, root)
			}
		}
		sort.Ints(roots)
		for _, root := range roots {
			orbit := orbits[root]
			sort.Ints(orbit)
			for r := 0; r < len(rows) && r+1 < len(orbit); r++ {
				for j := r + 1; j < len(orbit); j++ {
					v := rows[r][orbit[j]]
					if *m.vars[v].Upper <= 0 {
						continue
					}
					out = append(out, Constraint{
						Name:   fmt.Sprintf("structure_orbitope_%d_%d", r, v),
						Coeffs: map[Var]float64{v: 1},
						Sense:  LessOrEqual,
						RHS:    0,
					})
				}
			}
		}
	}
	return out
}

func verifyVariablePermutation(m *Model, perm []int) bool {
	if len(perm) != len(m.vars) {
		return false
	}
	seen := make([]bool, len(perm))
	for v, mapped := range perm {
		if mapped < 0 || mapped >= len(perm) || seen[mapped] {
			return false
		}
		seen[mapped] = true
		a, b := m.vars[v], m.vars[mapped]
		if a.Type != b.Type || *a.Lower != *b.Lower || *a.Upper != *b.Upper ||
			m.objective[Var(v)] != m.objective[Var(mapped)] {
			return false
		}
	}
	before := make([]string, len(m.constraints))
	after := make([]string, len(m.constraints))
	for r, row := range m.constraints {
		before[r] = rowSignaturePermutation(row, nil)
		after[r] = rowSignaturePermutation(row, perm)
	}
	sort.Strings(before)
	sort.Strings(after)
	for i := range before {
		if before[i] != after[i] {
			return false
		}
	}
	return true
}

func rowSignaturePermutation(row Constraint, perm []int) string {
	pairs := make([]string, 0, len(row.Coeffs))
	for v, a := range row.Coeffs {
		mapped := int(v)
		if perm != nil {
			mapped = perm[v]
		}
		pairs = append(pairs, fmt.Sprintf("%d:%s", mapped, floatKey(a)))
	}
	sort.Strings(pairs)
	return fmt.Sprintf("%d|%s|%s", row.Sense, floatKey(row.RHS), strings.Join(pairs, ","))
}
