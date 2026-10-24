# q-opt

q-opt is a pure Go CPU solver for linear programming (LP) and mixed-integer
linear programming (MIP). Its default LP engine is a bounded-variable revised
simplex with sparse columns and sparse LU; MIP search is single-threaded.

Start with the [Go API example](quickstart.md). See the [benchmark report](benchmarks.md)
for measured results and how to reproduce them.

The implementation is still correctness-first. It supports linear objectives,
continuous, integer, and binary variables, and `<=`, `>=`, and `=` constraints.
It does not support nonlinear constraints, SOS/indicator modeling, or parallel
branch-and-bound.

For development priorities, see [solver milestones](solver-milestones.md).
q-opt is available under the [Apache-2.0 license](https://github.com/qizhanchan/qopt/blob/main/LICENSE).
