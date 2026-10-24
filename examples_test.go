package qopt_test

import (
	"context"
	"fmt"

	qopt "github.com/qizhanchan/qopt"
)

func ExampleModel_lp() {
	m := qopt.NewModel(qopt.Maximize)
	x := m.AddVar(qopt.VarSpec{Name: "x"})
	y := m.AddVar(qopt.VarSpec{Name: "y"})
	_ = m.SetObjective(map[qopt.Var]float64{x: 3, y: 2})
	_ = m.AddConstraint(map[qopt.Var]float64{x: 1, y: 1}, qopt.LessOrEqual, 4)
	_ = m.AddConstraint(map[qopt.Var]float64{x: 1}, qopt.LessOrEqual, 2)
	_ = m.AddConstraint(map[qopt.Var]float64{y: 1}, qopt.LessOrEqual, 3)

	res, _ := m.Solve(context.Background(), qopt.Options{})
	fmt.Printf("%s %.0f %.0f %.0f\n", res.Status, res.Objective, res.Values[x], res.Values[y])
	// Output: optimal 10 2 2
}

func ExampleModel_mip() {
	m := qopt.NewModel(qopt.Maximize)
	x := m.AddVar(qopt.VarSpec{Name: "x", Type: qopt.Binary})
	y := m.AddVar(qopt.VarSpec{Name: "y", Type: qopt.Binary})
	_ = m.SetObjective(map[qopt.Var]float64{x: 6, y: 5})
	_ = m.AddConstraint(map[qopt.Var]float64{x: 4, y: 3}, qopt.LessOrEqual, 5)

	res, _ := m.Solve(context.Background(), qopt.Options{})
	fmt.Printf("%s %.0f %.0f %.0f\n", res.Status, res.Objective, res.Values[x], res.Values[y])
	// Output: optimal 6 1 0
}
