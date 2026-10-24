package lp

import (
	"errors"
	"math"
)

// denseLU is a dense LU factorization of a square m×m matrix with partial
// pivoting. After factor, the relation PA = LU holds, where L is unit lower
// triangular, U is upper triangular, and P is the row permutation. Stored
// compactly: lu[i][j] holds U[i][j] for j ≥ i and L[i][j] for j < i (L's
// unit diagonal is implicit).
//
// This is the "refactor every basis change" implementation — no Bartels-
// Golub or product-form updates. Acceptable for the v1 sizes we benchmark
// against (m ≤ ~1k); the next step beyond this prototype is rank-1 updates,
// not LU itself.
type denseLU struct {
	m         int
	lu        [][]float64 // m×m compact storage
	perm      []int       // row permutation: row k after factor was original perm[k]
	minPiv    float64
	maxBefore float64
	maxAfter  float64
}

func (f *denseLU) minPivot() float64 { return f.minPiv }

func (f *denseLU) nnz() int { return f.m * f.m }

var errSingularBasis = errors.New("revised simplex: singular basis")

// basisFactor is the small interface revisedSolver uses to abstract over its
// LU representation. denseLU and sparseLU both satisfy it. Eta product-form
// updates layer on top regardless of which concrete factor is underneath.
//
// ftranInto/btranInto are the allocation-free variants used by the simplex
// hot loops: they write the solution into dst, may use scratch as workspace,
// and require dst, rhs, and scratch to be pairwise distinct length-m slices
// (rhs is read-only). ftran/btran are allocating wrappers kept for the
// low-frequency callers (accessors, dual values).
type basisFactor interface {
	ftran(rhs []float64) []float64
	btran(rhs []float64) []float64
	ftranInto(dst, rhs, scratch []float64)
	btranInto(dst, rhs, scratch []float64)
	minPivot() float64
	growthFactor() float64
	// nnz is the stored-factor size — the per-solve triangular-solve cost in
	// the same unit as an eta column's length. Drives the cost-based
	// refactor rule (refactor when the eta chain outweighs a fresh factor).
	nnz() int
}

// factorDense builds an LU from a column-oriented basis B. cols[k] is the
// k-th column of B as a length-m vector. Returns errSingularBasis when the
// matrix is numerically singular.
func factorDense(cols [][]float64) (*denseLU, error) {
	m := len(cols)
	if m == 0 {
		return &denseLU{}, nil
	}
	if len(cols[0]) != m {
		return nil, errors.New("revised simplex: basis is not square")
	}
	lu := make([][]float64, m)
	maxBefore := 0.0
	for i := 0; i < m; i++ {
		lu[i] = make([]float64, m)
		for k := 0; k < m; k++ {
			if len(cols[k]) != m {
				return nil, errors.New("revised simplex: basis is not square")
			}
			lu[i][k] = cols[k][i]
			if a := math.Abs(lu[i][k]); a > maxBefore {
				maxBefore = a
			}
		}
	}
	perm := make([]int, m)
	for i := range perm {
		perm[i] = i
	}

	const pivotTol = 1e-12
	minPivot := math.Inf(1)
	maxAfter := maxBefore
	for k := 0; k < m; k++ {
		// Partial pivot: pick the row with largest |lu[i][k]| for i in [k, m).
		piv := k
		maxAbs := math.Abs(lu[k][k])
		for i := k + 1; i < m; i++ {
			if a := math.Abs(lu[i][k]); a > maxAbs {
				maxAbs = a
				piv = i
			}
		}
		if maxAbs < pivotTol {
			return nil, errSingularBasis
		}
		if maxAbs < minPivot {
			minPivot = maxAbs
		}
		if piv != k {
			lu[k], lu[piv] = lu[piv], lu[k]
			perm[k], perm[piv] = perm[piv], perm[k]
		}
		// Eliminate rows below.
		pivVal := lu[k][k]
		for i := k + 1; i < m; i++ {
			f := lu[i][k] / pivVal
			lu[i][k] = f
			row := lu[i]
			pkRow := lu[k]
			for j := k + 1; j < m; j++ {
				row[j] -= f * pkRow[j]
				if a := math.Abs(row[j]); a > maxAfter {
					maxAfter = a
				}
			}
		}
	}
	return &denseLU{m: m, lu: lu, perm: perm, minPiv: minPivot, maxBefore: maxBefore, maxAfter: maxAfter}, nil
}

func (f *denseLU) growthFactor() float64 {
	if f == nil || f.maxBefore == 0 {
		return 1
	}
	return f.maxAfter / f.maxBefore
}

// etaUpdate represents a basis replacement B' = B E, where E is the identity
// with column row replaced by col = B^{-1}a_enter. It is the standard
// product-form update used between full LU refactors.
type etaUpdate struct {
	row   int
	col   SparseVector
	pivot float64
}

func newEtaUpdate(row int, col []float64) etaUpdate {
	eta := etaUpdate{row: row}
	if row >= 0 && row < len(col) {
		eta.pivot = col[row]
	}
	for i, v := range col {
		if i == row || v == 0 {
			continue
		}
		eta.col.Append(i, v)
	}
	return eta
}

func applyEtaFTRAN(x []float64, eta etaUpdate) {
	if eta.row < 0 || eta.row >= len(x) {
		return
	}
	zr := x[eta.row] / eta.pivot
	for k, i := range eta.col.idx {
		x[i] -= eta.col.val[k] * zr
	}
	x[eta.row] = zr
}

func applyEtaBTRAN(y []float64, eta etaUpdate) {
	if eta.row < 0 || eta.row >= len(y) {
		return
	}
	sum := y[eta.row]
	for k, i := range eta.col.idx {
		sum -= eta.col.val[k] * y[i]
	}
	y[eta.row] = sum / eta.pivot
}

// ftran solves B x = rhs. Equivalent to applying B^{-1} to rhs.
func (f *denseLU) ftran(rhs []float64) []float64 {
	if f.m == 0 {
		return nil
	}
	x := make([]float64, f.m)
	f.ftranInto(x, rhs, make([]float64, f.m))
	return x
}

// ftranInto solves B x = rhs into dst using scratch as workspace. Performs
// the identical arithmetic in the identical order as the allocating ftran.
func (f *denseLU) ftranInto(dst, rhs, scratch []float64) {
	m := f.m
	if m == 0 {
		return
	}
	// y = P · rhs.
	y := scratch
	for k := 0; k < m; k++ {
		y[k] = rhs[f.perm[k]]
	}
	// Forward solve L · y' = y. L has unit diagonal; sub-diagonal entries
	// live in f.lu[i][j] for j < i.
	for i := 0; i < m; i++ {
		row := f.lu[i]
		sum := y[i]
		for j := 0; j < i; j++ {
			sum -= row[j] * y[j]
		}
		y[i] = sum
	}
	// Back solve U · x = y'.
	for i := m - 1; i >= 0; i-- {
		row := f.lu[i]
		sum := y[i]
		for j := i + 1; j < m; j++ {
			sum -= row[j] * dst[j]
		}
		dst[i] = sum / row[i]
	}
}

// btran solves B^T y = rhs. Equivalent to applying B^{-T} to rhs.
//
// Derivation: PA = LU ⇒ A = P^T LU ⇒ A^T = U^T L^T P. So solving A^T y = c:
// first solve U^T z = c (lower-triangular with U[i][i] as diagonal), then
// L^T w = z (upper-triangular with unit diagonal), then y = P^T w.
func (f *denseLU) btran(rhs []float64) []float64 {
	if f.m == 0 {
		return nil
	}
	y := make([]float64, f.m)
	f.btranInto(y, rhs, make([]float64, f.m))
	return y
}

// btranInto solves B^T y = rhs into dst using scratch as workspace. The L^T
// back-solve runs in place on scratch: w[i] reads z[i] (not yet overwritten)
// and w[j] for j > i (already overwritten), so the arithmetic is identical
// to the two-array version.
func (f *denseLU) btranInto(dst, rhs, scratch []float64) {
	m := f.m
	if m == 0 {
		return
	}
	// U^T z = rhs (forward solve in the "U columns become rows" sense).
	z := scratch
	for i := 0; i < m; i++ {
		sum := rhs[i]
		for j := 0; j < i; j++ {
			sum -= f.lu[j][i] * z[j]
		}
		z[i] = sum / f.lu[i][i]
	}
	// L^T w = z (back solve, unit diagonal), in place on z.
	for i := m - 1; i >= 0; i-- {
		sum := z[i]
		for j := i + 1; j < m; j++ {
			sum -= f.lu[j][i] * z[j]
		}
		z[i] = sum
	}
	// y = P^T w. perm[k] is the original row that ended up at permuted
	// position k, so y[perm[k]] = w[k].
	for k := 0; k < m; k++ {
		dst[f.perm[k]] = z[k]
	}
}
