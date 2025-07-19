// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/gradient/tf64"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// NewMultiVariateGaussian
func NewMultiVariateGaussian[T Float](cutoff, eta float64, graph, invert bool, rng *rand.Rand, name string, size int, vectors [][]T) (A, AI Matrix[T], u Matrix[T]) {
	if log {
		fmt.Println(name)
	}
	avg := make([]T, size)
	for _, measures := range vectors {
		for i, v := range measures {
			avg[i] += v
		}
	}
	if len(vectors) > 0 {
		for i := range avg {
			avg[i] /= T(len(vectors))
		}
	}
	cov := make([][]T, size)
	for i := range cov {
		cov[i] = make([]T, size)
	}
	for _, measures := range vectors {
		for i, v := range measures {
			for ii, vv := range measures {
				diff1 := avg[i] - v
				diff2 := avg[ii] - vv
				cov[i][ii] += diff1 * diff2
			}
		}
	}
	if len(vectors) > 0 {
		for i := range cov {
			for ii := range cov[i] {
				cov[i][ii] = cov[i][ii] / T(len(vectors))
			}
		}
	}
	if log {
		fmt.Println("K=")
		for i := range cov {
			fmt.Println(cov[i])
		}
		fmt.Println("u=")
		fmt.Println(avg)
		fmt.Println()
	}

	switch any(vectors).(type) {
	case [][]float64:
		set := tf64.NewSet()
		set.Add("A", size, size)
		set.Add("AI", size, size)

		for i := range set.Weights {
			w := set.Weights[i]
			if strings.HasPrefix(w.N, "b") {
				w.X = w.X[:cap(w.X)]
				w.States = make([][]float64, StateTotal)
				for ii := range w.States {
					w.States[ii] = make([]float64, len(w.X))
				}
				continue
			}
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for range cap(w.X) {
				w.X = append(w.X, rng.NormFloat64()*factor)
			}
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
		}

		others := tf64.NewSet()
		others.Add("E", size, size)
		others.Add("I", size, size)
		E := others.ByName["E"]
		for i := range cov {
			for ii := range cov[i] {
				E.X = append(E.X, float64(cov[i][ii]))
			}
		}
		I := others.ByName["I"]
		for i := range size {
			for ii := range size {
				if i == ii {
					I.X = append(I.X, 1)
				} else {
					I.X = append(I.X, 0)
				}
			}
		}

		{
			loss := tf64.Sum(tf64.Quadratic(others.Get("E"), tf64.Mul(set.Get("A"), set.Get("A"))))

			points, i := make(plotter.XYs, 0, 8), 0
			for {
				pow := func(x float64) float64 {
					y := math.Pow(x, float64(i+1))
					if math.IsNaN(y) || math.IsInf(y, 0) {
						return 0
					}
					return y
				}

				set.Zero()
				others.Zero()
				cost := tf64.Gradient(loss).X[0]
				if math.IsNaN(float64(cost)) || math.IsInf(float64(cost), 0) {
					fmt.Println(i, cost)
					break
				}

				norm := 0.0
				for _, p := range set.Weights {
					for _, d := range p.D {
						norm += d * d
					}
				}
				norm = math.Sqrt(norm)
				b1, b2 := pow(B1), pow(B2)
				scaling := 1.0
				if norm > 1 {
					scaling = 1 / norm
				}
				for _, w := range set.Weights {
					if w.N != "A" {
						continue
					}
					for ii, d := range w.D {
						g := d * scaling
						m := B1*w.States[StateM][ii] + (1-B1)*g
						v := B2*w.States[StateV][ii] + (1-B2)*g*g
						w.States[StateM][ii] = m
						w.States[StateV][ii] = v
						mhat := m / (1 - b1)
						vhat := v / (1 - b2)
						if vhat < 0 {
							vhat = 0
						}
						w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
					}
				}
				points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
				i++
				if i >= 1024 || ((cutoff != -1) && cost < cutoff) {
					break
				}
			}

			if graph {
				p := plot.New()

				p.Title.Text = "epochs vs cost"
				p.X.Label.Text = "epochs"
				p.Y.Label.Text = "cost"

				scatter, err := plotter.NewScatter(points)
				if err != nil {
					panic(err)
				}
				scatter.GlyphStyle.Radius = vg.Length(1)
				scatter.GlyphStyle.Shape = draw.CircleGlyph{}
				p.Add(scatter)

				err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("epochs_%s.png", name))
				if err != nil {
					panic(err)
				}
			}
		}

		if invert {
			loss := tf64.Sum(tf64.Quadratic(others.Get("I"), tf64.Mul(set.Get("A"), set.Get("AI"))))

			points, i := make(plotter.XYs, 0, 8), 0
			for {
				pow := func(x float64) float64 {
					y := math.Pow(x, float64(i+1))
					if math.IsNaN(y) || math.IsInf(y, 0) {
						return 0
					}
					return y
				}

				set.Zero()
				others.Zero()
				cost := tf64.Gradient(loss).X[0]
				if math.IsNaN(float64(cost)) || math.IsInf(float64(cost), 0) {
					fmt.Println(i, cost)
					break
				}

				norm := 0.0
				for _, p := range set.Weights {
					for _, d := range p.D {
						norm += d * d
					}
				}
				norm = math.Sqrt(norm)
				b1, b2 := pow(B1), pow(B2)
				scaling := 1.0
				if norm > 1 {
					scaling = 1 / norm
				}
				for _, w := range set.Weights {
					if w.N != "AI" {
						continue
					}
					for ii, d := range w.D {
						g := d * scaling
						m := B1*w.States[StateM][ii] + (1-B1)*g
						v := B2*w.States[StateV][ii] + (1-B2)*g*g
						w.States[StateM][ii] = m
						w.States[StateV][ii] = v
						mhat := m / (1 - b1)
						vhat := v / (1 - b2)
						if vhat < 0 {
							vhat = 0
						}
						w.X[ii] -= eta * mhat / (math.Sqrt(vhat) + 1e-8)
					}
				}
				points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
				if i >= 16*1024 || ((cutoff != -1) && cost < cutoff) {
					break
				}
			}

			if graph {
				p := plot.New()

				p.Title.Text = "epochs vs cost"
				p.X.Label.Text = "epochs"
				p.Y.Label.Text = "cost"

				scatter, err := plotter.NewScatter(points)
				if err != nil {
					panic(err)
				}
				scatter.GlyphStyle.Radius = vg.Length(1)
				scatter.GlyphStyle.Shape = draw.CircleGlyph{}
				p.Add(scatter)

				err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("inverse_epochs_%s.png", name))
				if err != nil {
					panic(err)
				}
			}
		}

		A = NewMatrix[T](size, size)
		for _, variance := range set.ByName["A"].X {
			A.Data = append(A.Data, T(variance))
		}
		AI = NewMatrix[T](size, size)
		for _, variance := range set.ByName["AI"].X {
			AI.Data = append(AI.Data, T(variance))
		}
		u = NewMatrix[T](size, 1)
		for _, a := range avg {
			u.Data = append(u.Data, T(a))
		}
	case [][]float32:
		set := tf32.NewSet()
		set.Add("A", size, size)
		set.Add("AI", size, size)

		for i := range set.Weights {
			w := set.Weights[i]
			if strings.HasPrefix(w.N, "b") {
				w.X = w.X[:cap(w.X)]
				w.States = make([][]float32, StateTotal)
				for ii := range w.States {
					w.States[ii] = make([]float32, len(w.X))
				}
				continue
			}
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for range cap(w.X) {
				w.X = append(w.X, float32(rng.NormFloat64()*factor))
			}
			w.States = make([][]float32, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float32, len(w.X))
			}
		}

		others := tf32.NewSet()
		others.Add("E", size, size)
		others.Add("I", size, size)
		E := others.ByName["E"]
		for i := range cov {
			for ii := range cov[i] {
				E.X = append(E.X, float32(cov[i][ii]))
			}
		}
		I := others.ByName["I"]
		for i := range size {
			for ii := range size {
				if i == ii {
					I.X = append(I.X, 1)
				} else {
					I.X = append(I.X, 0)
				}
			}
		}

		{
			loss := tf32.Sum(tf32.Quadratic(others.Get("E"), tf32.Mul(set.Get("A"), set.Get("A"))))

			points, i := make(plotter.XYs, 0, 8), 0
			for {
				pow := func(x float64) float64 {
					y := math.Pow(x, float64(i+1))
					if math.IsNaN(y) || math.IsInf(y, 0) {
						return 0
					}
					return y
				}

				set.Zero()
				others.Zero()
				cost := tf32.Gradient(loss).X[0]
				if math.IsNaN(float64(cost)) || math.IsInf(float64(cost), 0) {
					fmt.Println(i, cost)
					break
				}

				norm := 0.0
				for _, p := range set.Weights {
					for _, d := range p.D {
						norm += float64(d * d)
					}
				}
				norm = math.Sqrt(norm)
				b1, b2 := pow(B1), pow(B2)
				scaling := 1.0
				if norm > 1 {
					scaling = 1 / norm
				}
				for _, w := range set.Weights {
					if w.N != "A" {
						continue
					}
					for ii, d := range w.D {
						g := d * float32(scaling)
						m := B1*w.States[StateM][ii] + (1-B1)*g
						v := B2*w.States[StateV][ii] + (1-B2)*g*g
						w.States[StateM][ii] = m
						w.States[StateV][ii] = v
						mhat := m / (1 - float32(b1))
						vhat := v / (1 - float32(b2))
						if vhat < 0 {
							vhat = 0
						}
						w.X[ii] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
					}
				}
				points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
				i++
				if i >= 1024 || ((cutoff != -1) && float64(cost) < cutoff) {
					break
				}
			}

			if graph {
				p := plot.New()

				p.Title.Text = "epochs vs cost"
				p.X.Label.Text = "epochs"
				p.Y.Label.Text = "cost"

				scatter, err := plotter.NewScatter(points)
				if err != nil {
					panic(err)
				}
				scatter.GlyphStyle.Radius = vg.Length(1)
				scatter.GlyphStyle.Shape = draw.CircleGlyph{}
				p.Add(scatter)

				err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("epochs_%s.png", name))
				if err != nil {
					panic(err)
				}
			}
		}

		if invert {
			loss := tf32.Sum(tf32.Quadratic(others.Get("I"), tf32.Mul(set.Get("A"), set.Get("AI"))))

			points, i := make(plotter.XYs, 0, 8), 0
			for {
				pow := func(x float64) float64 {
					y := math.Pow(x, float64(i+1))
					if math.IsNaN(y) || math.IsInf(y, 0) {
						return 0
					}
					return y
				}

				set.Zero()
				others.Zero()
				cost := tf32.Gradient(loss).X[0]
				if math.IsNaN(float64(cost)) || math.IsInf(float64(cost), 0) {
					fmt.Println(i, cost)
					break
				}

				norm := 0.0
				for _, p := range set.Weights {
					for _, d := range p.D {
						norm += float64(d * d)
					}
				}
				norm = math.Sqrt(norm)
				b1, b2 := pow(B1), pow(B2)
				scaling := 1.0
				if norm > 1 {
					scaling = 1 / norm
				}
				for _, w := range set.Weights {
					if w.N != "AI" {
						continue
					}
					for ii, d := range w.D {
						g := d * float32(scaling)
						m := B1*w.States[StateM][ii] + (1-B1)*g
						v := B2*w.States[StateV][ii] + (1-B2)*g*g
						w.States[StateM][ii] = m
						w.States[StateV][ii] = v
						mhat := m / (1 - float32(b1))
						vhat := v / (1 - float32(b2))
						if vhat < 0 {
							vhat = 0
						}
						w.X[ii] -= float32(eta) * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
					}
				}
				points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
				if i >= 16*1024 || ((cutoff != -1) && float64(cost) < cutoff) {
					break
				}
			}

			if graph {
				p := plot.New()

				p.Title.Text = "epochs vs cost"
				p.X.Label.Text = "epochs"
				p.Y.Label.Text = "cost"

				scatter, err := plotter.NewScatter(points)
				if err != nil {
					panic(err)
				}
				scatter.GlyphStyle.Radius = vg.Length(1)
				scatter.GlyphStyle.Shape = draw.CircleGlyph{}
				p.Add(scatter)

				err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("inverse_epochs_%s.png", name))
				if err != nil {
					panic(err)
				}
			}
		}

		A = NewMatrix[T](size, size)
		for _, variance := range set.ByName["A"].X {
			A.Data = append(A.Data, T(variance))
		}
		AI = NewMatrix[T](size, size)
		for _, variance := range set.ByName["AI"].X {
			AI.Data = append(AI.Data, T(variance))
		}
		u = NewMatrix[T](size, 1)
		for _, a := range avg {
			u.Data = append(u.Data, T(a))
		}
	}
	return A, AI, u
}
