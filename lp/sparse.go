package lp

// SparseVector is a compressed-storage column: idx holds the row indices of
// the nonzero entries in ascending order; val holds the matching values. The
// two slices always have the same length.
//
// Most LP/MIP matrices have few nonzeros per column (a constraint usually
// references a small subset of the variables). Storing the columns this way
// keeps the per-iteration work in pricing, the dual ratio test, and the
// basic-value/residual passes proportional to nnz rather than m. The dense
// LU factor still consumes a dense basis — sparse columns are materialized
// into a dense buffer at refactor time.
type SparseVector struct {
	idx []int
	val []float64
}

// Append pushes (row, value) onto the vector. Entries must be appended in
// strictly increasing row order. Zero values are silently dropped.
func (c *SparseVector) Append(row int, value float64) {
	if value == 0 {
		return
	}
	c.idx = append(c.idx, row)
	c.val = append(c.val, value)
}

// Len reports the number of stored entries.
func (c SparseVector) Len() int { return len(c.idx) }

// dotDense returns y · col, where col is viewed as a length-m dense vector.
func (c SparseVector) dotDense(y []float64) float64 {
	s := 0.0
	for k, i := range c.idx {
		s += y[i] * c.val[k]
	}
	return s
}

// addToDense does y[i] += scale * col[i] for every nonzero of the column.
func (c SparseVector) addToDense(y []float64, scale float64) {
	if scale == 0 {
		return
	}
	for k, i := range c.idx {
		y[i] += scale * c.val[k]
	}
}

// writeDense materializes the column into dst (length m), zeroing the
// other entries. Returns dst for convenience.
func (c SparseVector) writeDense(dst []float64) []float64 {
	for i := range dst {
		dst[i] = 0
	}
	for k, i := range c.idx {
		dst[i] = c.val[k]
	}
	return dst
}

// toDense allocates a fresh length-m dense vector containing the column.
func (c SparseVector) toDense(m int) []float64 {
	return c.writeDense(make([]float64, m))
}
