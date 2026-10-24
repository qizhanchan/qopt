# Quick start

Go 1.26 is required by `go.mod`. After the repository is public, install it with:

```sh
go get github.com/qizhanchan/qopt@latest
```

```go
package main

import (
    "context"
    "fmt"

    qopt "github.com/qizhanchan/qopt"
)

func main() {
    m := qopt.NewModel(qopt.Maximize)
    x := m.AddVar(qopt.VarSpec{Name: "x", Type: qopt.Binary})
    y := m.AddVar(qopt.VarSpec{Name: "y", Type: qopt.Binary})

    _ = m.SetObjective(map[qopt.Var]float64{x: 6, y: 5})
    _ = m.AddConstraint(map[qopt.Var]float64{x: 4, y: 3}, qopt.LessOrEqual, 5)

    res, err := m.Solve(context.Background(), qopt.Options{})
    fmt.Println(res.Status, res.Objective, err)
}
```

`Options{}` selects the revised LP backend, depth-first node selection, and
most-fractional branching. The [model API source](https://github.com/qizhanchan/qopt/blob/main/model.go)
documents the opt-in search controls.
