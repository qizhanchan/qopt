package qopt

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

// detectSymmetryOrbits finds groups of integer/binary variables that are
// verified pairwise interchangeable: swapping any two members of an orbit,
// together with some permutation of the constraint rows, reproduces the
// model's (A, b, c, bounds) exactly. Because verified transpositions compose
// (the automorphisms of a linear system form a group), a connected component
// under "verified transposition" generates the full symmetric group on that
// component — so a lexicographic ordering constraint across the whole orbit
// is a sound symmetry-breaking cut, not just a heuristic.
//
// Candidate pairs are pruned first via Weisfeiler-Leman color refinement over
// the colored bipartite variable/constraint graph (cheap, but only an
// over-approximation of the true orbits — two variables can share a color
// without being genuinely interchangeable). Every candidate pair is then
// individually verified against the actual coefficient data before it is
// trusted, per the P1 roadmap note that raw color classes can cut off the
// true optimum if used directly.
func detectSymmetryOrbits(m *Model) [][]Var {
	n := len(m.vars)
	if n == 0 || len(m.constraints) == 0 {
		return nil
	}

	rowsOf := make([][]int, n)
	for r, c := range m.constraints {
		for v := range c.Coeffs {
			rowsOf[v] = append(rowsOf[v], r)
		}
	}

	varClass := initialVarClasses(m)
	rowClass := initialRowClasses(m)

	for iter := 0; iter < 20; iter++ {
		newVarClass, vChanged := refineVarClasses(m, rowsOf, varClass, rowClass)
		newRowClass, rChanged := refineRowClasses(m, varClass, rowClass)
		varClass, rowClass = newVarClass, newRowClass
		if !vChanged && !rChanged {
			break
		}
	}

	// Group candidate variables by final color, restricted to integer/binary
	// types (continuous symmetry doesn't help B&B pruning).
	candidates := make(map[int][]int)
	for v := 0; v < n; v++ {
		if m.vars[v].Type == Continuous {
			continue
		}
		candidates[varClass[v]] = append(candidates[varClass[v]], v)
	}

	uf := newUnionFind(n)
	verifiedEdges := 0
	for _, group := range candidates {
		if len(group) < 2 {
			continue
		}
		for a := 0; a < len(group); a++ {
			for b := a + 1; b < len(group); b++ {
				i, j := group[a], group[b]
				if verifyTransposition(m, rowsOf, i, j) {
					uf.union(i, j)
					verifiedEdges++
				}
			}
		}
	}

	orbits := make(map[int][]Var)
	for v := 0; v < n; v++ {
		if m.vars[v].Type == Continuous {
			continue
		}
		root := uf.find(v)
		orbits[root] = append(orbits[root], Var(v))
	}

	var result [][]Var
	for _, orbit := range orbits {
		if len(orbit) < 2 {
			continue
		}
		sort.Slice(orbit, func(i, j int) bool { return orbit[i] < orbit[j] })
		result = append(result, orbit)
	}
	sort.Slice(result, func(i, j int) bool { return result[i][0] < result[j][0] })

	if debugEnabled() {
		fmt.Fprintf(os.Stderr, "qopt: symmetry detection found %d verified edges, %d orbits (sizes: %v)\n",
			verifiedEdges, len(result), orbitSizes(result))
	}
	return result
}

func orbitSizes(orbits [][]Var) []int {
	sizes := make([]int, len(orbits))
	for i, o := range orbits {
		sizes[i] = len(o)
	}
	return sizes
}

// --- color refinement (Weisfeiler-Leman over the colored bipartite graph) ---

func initialVarClasses(m *Model) []int {
	sig := make([]string, len(m.vars))
	for v, spec := range m.vars {
		obj := m.objective[Var(v)]
		sig[v] = fmt.Sprintf("%d|%s|%s|%s", spec.Type, floatKey(*spec.Lower), floatKey(*spec.Upper), floatKey(obj))
	}
	return classIDsFromSignatures(sig)
}

func initialRowClasses(m *Model) []int {
	sig := make([]string, len(m.constraints))
	for r, c := range m.constraints {
		sig[r] = fmt.Sprintf("%d|%s", c.Sense, floatKey(c.RHS))
	}
	return classIDsFromSignatures(sig)
}

func refineVarClasses(m *Model, rowsOf [][]int, varClass, rowClass []int) ([]int, bool) {
	n := len(varClass)
	sig := make([]string, n)
	for v := 0; v < n; v++ {
		pairs := make([]string, 0, len(rowsOf[v]))
		for _, r := range rowsOf[v] {
			coeff := m.constraints[r].Coeffs[Var(v)]
			pairs = append(pairs, fmt.Sprintf("%d:%s", rowClass[r], floatKey(coeff)))
		}
		sort.Strings(pairs)
		sig[v] = fmt.Sprintf("%d#%s", varClass[v], strings.Join(pairs, ","))
	}
	next := classIDsFromSignatures(sig)
	return next, maxClass(next) != maxClass(varClass)
}

func refineRowClasses(m *Model, varClass, rowClass []int) ([]int, bool) {
	n := len(rowClass)
	sig := make([]string, n)
	for r, c := range m.constraints {
		pairs := make([]string, 0, len(c.Coeffs))
		for v, coeff := range c.Coeffs {
			pairs = append(pairs, fmt.Sprintf("%d:%s", varClass[v], floatKey(coeff)))
		}
		sort.Strings(pairs)
		sig[r] = fmt.Sprintf("%d#%s", rowClass[r], strings.Join(pairs, ","))
	}
	next := classIDsFromSignatures(sig)
	return next, maxClass(next) != maxClass(rowClass)
}

func maxClass(classes []int) int {
	m := -1
	for _, c := range classes {
		if c > m {
			m = c
		}
	}
	return m
}

// classIDsFromSignatures assigns a stable, dense integer id to each distinct
// signature string, preserving first-seen order (so results are
// deterministic across runs of the same model).
func classIDsFromSignatures(sig []string) []int {
	ids := make(map[string]int, len(sig))
	out := make([]int, len(sig))
	for i, s := range sig {
		id, ok := ids[s]
		if !ok {
			id = len(ids)
			ids[s] = id
		}
		out[i] = id
	}
	return out
}

func floatKey(f float64) string {
	// Verification must not merge distinct coefficients: a false automorphism
	// would make the generated symmetry-breaking constraints invalid.
	if f == 0 {
		return "0" // normalize signed zero, which compares equal numerically
	}
	return strconv.FormatUint(math.Float64bits(f), 16)
}

// --- verification: is swapping columns i and j (with some row permutation)
// an exact automorphism of the model? ---

func verifyTransposition(m *Model, rowsOf [][]int, i, j int) bool {
	vi, vj := m.vars[i], m.vars[j]
	if vi.Type != vj.Type || *vi.Lower != *vj.Lower || *vi.Upper != *vj.Upper {
		return false
	}
	if m.objective[Var(i)] != m.objective[Var(j)] {
		return false
	}

	touched := unionRows(rowsOf[i], rowsOf[j])
	before := make([]string, 0, len(touched))
	after := make([]string, 0, len(touched))
	for _, r := range touched {
		c := m.constraints[r]
		before = append(before, rowSignature(c, -1, -1))
		after = append(after, rowSignature(c, i, j))
	}
	sort.Strings(before)
	sort.Strings(after)
	if len(before) != len(after) {
		return false
	}
	for k := range before {
		if before[k] != after[k] {
			return false
		}
	}
	return true
}

// rowSignature encodes a constraint's sense, RHS, and sorted coefficients.
// If swapI/swapJ are both >= 0, the coefficients at those two variable
// indices are swapped before encoding (an entry absent from Coeffs is
// treated as 0).
func rowSignature(c Constraint, swapI, swapJ int) string {
	var ci, cj float64
	if swapI >= 0 {
		ci, cj = c.Coeffs[Var(swapI)], c.Coeffs[Var(swapJ)]
	}
	pairs := make([]string, 0, len(c.Coeffs))
	for v, coeff := range c.Coeffs {
		if swapI >= 0 {
			if int(v) == swapI {
				coeff = cj
			} else if int(v) == swapJ {
				coeff = ci
			}
		}
		if coeff == 0 {
			continue
		}
		pairs = append(pairs, fmt.Sprintf("%d:%s", v, floatKey(coeff)))
	}
	sort.Strings(pairs)
	return fmt.Sprintf("%d|%s|%s", c.Sense, floatKey(c.RHS), strings.Join(pairs, ","))
}

func unionRows(a, b []int) []int {
	seen := make(map[int]struct{}, len(a)+len(b))
	for _, r := range a {
		seen[r] = struct{}{}
	}
	for _, r := range b {
		seen[r] = struct{}{}
	}
	out := make([]int, 0, len(seen))
	for r := range seen {
		out = append(out, r)
	}
	return out
}

// --- union-find ---

type unionFind struct {
	parent []int
}

func newUnionFind(n int) *unionFind {
	p := make([]int, n)
	for i := range p {
		p[i] = i
	}
	return &unionFind{parent: p}
}

func (u *unionFind) find(x int) int {
	for u.parent[x] != x {
		u.parent[x] = u.parent[u.parent[x]]
		x = u.parent[x]
	}
	return x
}

func (u *unionFind) union(a, b int) {
	ra, rb := u.find(a), u.find(b)
	if ra != rb {
		u.parent[ra] = rb
	}
}

// addSymmetryBreakingConstraints appends a lexicographic ordering constraint
// chain (x_v1 <= x_v2 <= ... <= x_vk) for each orbit to m. Valid because
// every permutation of an orbit is a genuine model automorphism (verified
// transpositions generate the full symmetric group on a connected
// component), so any feasible solution can be re-sorted within its orbit
// without changing feasibility or objective value.
func addSymmetryBreakingConstraints(m *Model, orbits [][]Var) {
	for _, orbit := range orbits {
		for k := 0; k+1 < len(orbit); k++ {
			_ = m.AddNamedConstraint(
				fmt.Sprintf("symbreak_%d_%d", orbit[k], orbit[k+1]),
				map[Var]float64{orbit[k]: 1, orbit[k+1]: -1},
				LessOrEqual,
				0,
			)
		}
	}
}
