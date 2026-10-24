package lp

import (
	"errors"
	"fmt"
	"math"
)

// sparseLU is a left-looking sparse LU factorization with row partial
// pivoting. After factor, PA = LU where L is unit lower triangular, U is
// upper triangular, and P is the row permutation.
//
// Storage:
//   - L[k] is column k of L, holding the strict sub-diagonal entries L[i,k]
//     for permuted rows i > k. L[k].idx[*] are ORIGINAL row indices; their
//     permuted positions are pinv[idx[*]], always > k after step k.
//     L's unit diagonal is implicit.
//   - U[k] is column k of U, holding U[j,k] for permuted rows j in [0, k].
//     U[k].idx[*] are PERMUTED row indices, sorted ascending; the last entry
//     is the diagonal U[k,k].
//   - pivot[k] = original row chosen as the pivot for permuted row k.
//   - pinv[origRow] = k such that pivot[k] == origRow.
//
// This is the algorithmic counterpart of denseLU and implements basisFactor.
// It does not yet use Gilbert-Peierls' symbolic DFS; the per-column work is
// O(k) for the j-iteration plus O(sum nnz of touched L columns) for the
// numerical subtractions. That's enough to exploit basis sparsity without
// the extra bookkeeping the topological order phase requires.
type sparseLU struct {
	m         int
	L         []SparseVector
	U         []SparseVector
	pivot     []int
	pinv      []int
	minPiv    float64
	maxBefore float64
	maxAfter  float64
}

func (f *sparseLU) minPivot() float64 { return f.minPiv }

func (f *sparseLU) nnz() int {
	total := 0
	for k := 0; k < f.m; k++ {
		total += len(f.L[k].idx) + len(f.U[k].idx)
	}
	return total
}

func (f *sparseLU) growthFactor() float64 {
	if f == nil || f.maxBefore == 0 {
		return 1
	}
	return f.maxAfter / f.maxBefore
}

// factorSparse factors a square m×m basis given as sparse columns.
func factorSparse(cols []SparseVector, m int) (*sparseLU, error) {
	if m == 0 {
		return &sparseLU{}, nil
	}
	if len(cols) != m {
		return nil, errors.New("revised simplex: basis is not square")
	}

	L := make([]SparseVector, m)
	U := make([]SparseVector, m)
	pivot := make([]int, m)
	pinv := make([]int, m)
	for i := range pinv {
		pinv[i] = -1
	}

	v := make([]float64, m)
	onList := make([]bool, m)
	touched := make([]int, 0, m)

	maxBefore := 0.0
	for j := 0; j < m; j++ {
		for _, val := range cols[j].val {
			if a := math.Abs(val); a > maxBefore {
				maxBefore = a
			}
		}
	}
	maxAfter := maxBefore
	const pivotTol = 1e-12
	minPiv := math.Inf(1)

	for k := 0; k < m; k++ {
		// Reset v at previously touched indices.
		for _, i := range touched {
			v[i] = 0
			onList[i] = false
		}
		touched = touched[:0]

		// Load B[:, k] into v.
		col := cols[k]
		for kk, origRow := range col.idx {
			v[origRow] = col.val[kk]
			if !onList[origRow] {
				onList[origRow] = true
				touched = append(touched, origRow)
			}
		}

		// Triangular solve: subtract v[pivot[j]] * L[j] from v, for j in order.
		// Order matters because v[pivot[j]] is read at step j (it equals
		// U[j, k]) and L[j']'s entries (j' > j) may write to v[pivot[j']],
		// which feeds the next iteration.
		for j := 0; j < k; j++ {
			pj := pivot[j]
			if !onList[pj] {
				continue
			}
			vj := v[pj]
			if vj == 0 {
				continue
			}
			Lj := L[j]
			for kk, i := range Lj.idx {
				if !onList[i] {
					onList[i] = true
					touched = append(touched, i)
				}
				v[i] -= Lj.val[kk] * vj
			}
		}

		// Build U[k] from v[pivot[0..k-1]] (sorted ascending since we iterate j).
		var Uk SparseVector
		for j := 0; j < k; j++ {
			val := v[pivot[j]]
			if val != 0 {
				Uk.idx = append(Uk.idx, j)
				Uk.val = append(Uk.val, val)
			}
		}

		// Pivot selection among touched rows with pinv == -1.
		pivRow := -1
		maxAbs := pivotTol
		for _, i := range touched {
			if pinv[i] >= 0 {
				continue
			}
			if a := math.Abs(v[i]); a > maxAbs {
				maxAbs = a
				pivRow = i
			}
		}
		if pivRow < 0 {
			if debugEnabled() {
				best := 0.0
				cand := 0
				for _, i := range touched {
					if pinv[i] >= 0 {
						continue
					}
					cand++
					if a := math.Abs(v[i]); a > best {
						best = a
					}
				}
				return nil, fmt.Errorf("%w (col %d/%d: %d unassigned candidates, max |v| = %g)",
					errSingularBasis, k, m, cand, best)
			}
			return nil, errSingularBasis
		}
		if maxAbs < minPiv {
			minPiv = maxAbs
		}
		pivot[k] = pivRow
		pinv[pivRow] = k

		// Diagonal U[k,k] = v[pivRow], appended last so idx is sorted.
		Uk.idx = append(Uk.idx, k)
		Uk.val = append(Uk.val, v[pivRow])
		U[k] = Uk

		// Track post-factor magnitudes for growthFactor stats.
		for _, val := range Uk.val {
			if a := math.Abs(val); a > maxAfter {
				maxAfter = a
			}
		}

		// L[k]: rows i ∈ touched with pinv == -1 (still unassigned) and v[i] != 0.
		pivVal := v[pivRow]
		var Lk SparseVector
		for _, i := range touched {
			if pinv[i] != -1 {
				continue
			}
			if v[i] == 0 {
				continue
			}
			lval := v[i] / pivVal
			Lk.idx = append(Lk.idx, i)
			Lk.val = append(Lk.val, lval)
			if a := math.Abs(lval); a > maxAfter {
				maxAfter = a
			}
		}
		L[k] = Lk
	}

	return &sparseLU{
		m:         m,
		L:         L,
		U:         U,
		pivot:     pivot,
		pinv:      pinv,
		minPiv:    minPiv,
		maxBefore: maxBefore,
		maxAfter:  maxAfter,
	}, nil
}

// ftran solves B x = rhs.
func (f *sparseLU) ftran(rhs []float64) []float64 {
	if f.m == 0 {
		return nil
	}
	x := make([]float64, f.m)
	f.ftranInto(x, rhs, nil)
	return x
}

// ftranInto solves B x = rhs into dst. The whole solve runs in place on dst,
// so no scratch is needed (the parameter exists to satisfy basisFactor).
// dst must be distinct from rhs.
func (f *sparseLU) ftranInto(dst, rhs, _ []float64) {
	m := f.m
	if m == 0 {
		return
	}
	// y = P · rhs (permute rows).
	y := dst
	for k := 0; k < m; k++ {
		y[k] = rhs[f.pivot[k]]
	}

	// Forward solve L z = y in-place on y. Column-oriented:
	//   z[j] = y[j] (unit diag); for each (i, l) in L[j]: y[i] -= l * y[j].
	for j := 0; j < m; j++ {
		zj := y[j]
		if zj == 0 {
			continue
		}
		Lj := f.L[j]
		for kk, origRow := range Lj.idx {
			permRow := f.pinv[origRow]
			y[permRow] -= Lj.val[kk] * zj
		}
	}

	// Back solve U x = z in-place on y. Column-oriented:
	//   for k = m-1..0: x[k] = z[k] / U[k,k]; for each (j, u) in U[k] with j<k: z[j] -= u * x[k].
	for k := m - 1; k >= 0; k-- {
		Uk := f.U[k]
		nlu := len(Uk.idx)
		diag := Uk.val[nlu-1]
		y[k] /= diag
		xk := y[k]
		for kk := 0; kk < nlu-1; kk++ {
			j := Uk.idx[kk]
			y[j] -= Uk.val[kk] * xk
		}
	}
}

// btran solves B^T y = rhs. Derivation: PA = LU ⇒ A^T = U^T L^T P, so
// solving A^T y = rhs reduces to (1) U^T z = rhs, (2) L^T w = z,
// (3) y = P^T w.
func (f *sparseLU) btran(rhs []float64) []float64 {
	if f.m == 0 {
		return nil
	}
	y := make([]float64, f.m)
	f.btranInto(y, rhs, make([]float64, f.m))
	return y
}

// btranInto solves B^T y = rhs into dst using scratch as workspace. The L^T
// back-solve runs in place on scratch: w[i] overwrites z[i], reading only
// z[i] itself and w[permRow] for rows assigned after step i (permRow > i),
// so the arithmetic matches the two-array version exactly. dst must be
// distinct from both rhs and scratch.
func (f *sparseLU) btranInto(dst, rhs, scratch []float64) {
	m := f.m
	if m == 0 {
		return
	}
	// Forward solve U^T z = rhs (U^T is lower triangular, indexed by permuted rows).
	// Row-oriented, column-stored U: for k = 0..m-1:
	//   z[k] = (rhs[k] - sum_{j<k} U[j,k] * z[j]) / U[k,k]
	z := scratch
	for k := 0; k < m; k++ {
		Uk := f.U[k]
		nlu := len(Uk.idx)
		diag := Uk.val[nlu-1]
		sum := rhs[k]
		for kk := 0; kk < nlu-1; kk++ {
			j := Uk.idx[kk]
			sum -= Uk.val[kk] * z[j]
		}
		z[k] = sum / diag
	}

	// Back solve L^T w = z (L^T is unit upper triangular, in permuted rows),
	// in place on z. L[i].idx are original rows; permuted = pinv[origRow],
	// always > i, so w-reads only touch already-finished entries.
	for i := m - 1; i >= 0; i-- {
		sum := z[i]
		Li := f.L[i]
		for kk, origRow := range Li.idx {
			permRow := f.pinv[origRow]
			sum -= Li.val[kk] * z[permRow]
		}
		z[i] = sum
	}

	// y = P^T w: y[pivot[k]] = w[k].
	for k := 0; k < m; k++ {
		dst[f.pivot[k]] = z[k]
	}
}
