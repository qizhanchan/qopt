package qopt

import "math"

// scaleFactors holds the diagonal pre/post multipliers used to equilibrate a
// linear program. After scaling, the simplex solves
//
//	min  cScaled · y       s.t. AScaled · y {≤,≥,=} bScaled,  y ≥ 0
//
// where AScaled = diag(rowScale) · A · diag(colScale) and y = x / colScale.
// Solutions in y-space are mapped back with x = colScale · y.
//
// The bench/MIPLIB run exposed a concrete failure mode: in glass4 the
// constraint matrix has entries up to 1e8, so a residual of 1e-6 (which is
// "huge" in original units) only looks like 1e-2 in scaled space, and the
// Phase I "artificial-vars-must-be-zero" check using a flat 1e-8 absolute
// tolerance was reporting false infeasibility. Equilibrating the inf-norms
// of every row and column to ≈ 1 makes that tolerance meaningful regardless
// of the user's data scale.
type scaleFactors struct {
	rowScale []float64
	colScale []float64
}

func newIdentityScale(m, n int) scaleFactors {
	rs := make([]float64, m)
	cs := make([]float64, n)
	for i := range rs {
		rs[i] = 1
	}
	for j := range cs {
		cs[j] = 1
	}
	return scaleFactors{rowScale: rs, colScale: cs}
}

// ruizScale rescales a, b, c in place to drive row and column inf-norms of A
// toward 1. The classic Ruiz iteration applies D_r = diag(1/√max_j|a_ij|) and
// D_c = diag(1/√max_i|a_ij|) alternately; a handful of sweeps suffices in
// practice. Returns the cumulative row/col scales so the caller can unscale
// the solution.
//
// `iters` should be small (3–10). Stops early when no factor moves more than
// ~1% from 1.
func ruizScale(a [][]float64, b, c []float64, iters int) scaleFactors {
	m := len(a)
	n := len(c)
	sf := newIdentityScale(m, n)
	if m == 0 || n == 0 {
		return sf
	}

	const convergeEps = 1e-3
	for k := 0; k < iters; k++ {
		moved := false

		// Row sweep.
		for i := 0; i < m; i++ {
			maxAbs := 0.0
			for j := 0; j < n; j++ {
				if v := math.Abs(a[i][j]); v > maxAbs {
					maxAbs = v
				}
			}
			if maxAbs == 0 {
				continue
			}
			factor := 1.0 / math.Sqrt(maxAbs)
			if math.Abs(factor-1) > convergeEps {
				moved = true
			}
			sf.rowScale[i] *= factor
			for j := 0; j < n; j++ {
				a[i][j] *= factor
			}
			b[i] *= factor
		}

		// Column sweep.
		for j := 0; j < n; j++ {
			maxAbs := 0.0
			for i := 0; i < m; i++ {
				if v := math.Abs(a[i][j]); v > maxAbs {
					maxAbs = v
				}
			}
			if maxAbs == 0 {
				continue
			}
			factor := 1.0 / math.Sqrt(maxAbs)
			if math.Abs(factor-1) > convergeEps {
				moved = true
			}
			sf.colScale[j] *= factor
			for i := 0; i < m; i++ {
				a[i][j] *= factor
			}
			c[j] *= factor
		}

		if !moved {
			break
		}
	}
	return sf
}
