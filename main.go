// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"runtime"
	"strconv"
	"strings"

	"github.com/pointlander/gradient/tf64"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
	// Scale is the scale of the model
	Scale = 128
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

var Galaxies = [][]float64{
	{2.48, 00, 42, 41.877, 40, 51, 54.71}, // M32
	{2.69, 00, 40, 22.054, 41, 41, 08.04}, // M110
	{2.01, 00, 38, 57.523, 48, 20, 14.86}, // NGC 185
	{2.20, 00, 33, 12.131, 48, 30, 32.82}, // NGC 147
	{2.43, 00, 45, 39.264, 38, 02, 35.17}, // Andromeda I
	{2.13, 01, 16, 28.136, 33, 25, 50.36}, // Andromeda II
	{2.44, 00, 35, 31.777, 36, 30, 04.19}, // Andromeda III
	{2.52, 01, 10, 16.952, 47, 37, 40.12}, // Andromeda V
	{2.55, 23, 51, 46.516, 24, 34, 55.69}, // Andromeda VI
	{2.49, 23, 26, 33.321, 50, 40, 49.98}, // Andromeda VII
	{2.70, 00, 42, 06.000, 40, 37, 00.00}, // Andromeda VIII
	{2.50, 00, 52, 52.493, 43, 11, 55.66}, // Andromeda IX
	{2.90, 01, 06, 34.740, 44, 48, 23.31}, // Andromeda X
}

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Cluster  int
	Index    int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Inverse is the labels inverse map
var Inverse = [3]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

// NewMultiVariateGaussian
func NewMultiVariateGaussian(rng *rand.Rand, name string, size int, vectors [][]float64) (A, AI Matrix, u Matrix) {
	fmt.Println(name)
	avg := make([]float64, size)
	for _, measures := range vectors {
		for i, v := range measures {
			avg[i] += v
		}
	}
	for i := range avg {
		avg[i] /= float64(len(vectors))
	}
	cov := make([][]float64, size)
	for i := range cov {
		cov[i] = make([]float64, size)
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
	for i := range cov {
		for ii := range cov[i] {
			cov[i][ii] = cov[i][ii] / float64(len(vectors))
		}
	}
	fmt.Println("K=")
	for i := range cov {
		fmt.Println(cov[i])
	}
	fmt.Println("u=")
	fmt.Println(avg)
	fmt.Println()

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
			E.X = append(E.X, cov[i][ii])
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

		points := make(plotter.XYs, 0, 8)
		for i := range 1024 {
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
		}

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

	{
		loss := tf64.Sum(tf64.Quadratic(others.Get("I"), tf64.Mul(set.Get("A"), set.Get("AI"))))

		points := make(plotter.XYs, 0, 8)
		for i := range 16 * 1024 {
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
					w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				}
			}
			points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
		}

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

	A = NewMatrix(size, size)
	for _, variance := range set.ByName["A"].X {
		A.Data = append(A.Data, variance)
	}
	AI = NewMatrix(size, size)
	for _, variance := range set.ByName["AI"].X {
		AI.Data = append(AI.Data, variance)
	}
	u = NewMatrix(size, 1, avg...)
	return A, AI, u
}

// L2 is the L2 norm
func L2(a, b []float64) float64 {
	c := 0.0
	for i, v := range a {
		diff := v - b[i]
		c += diff * diff
	}
	return c
}

func main() {
	iris := Load()
	var vectors [3][][]float64
	for i := range vectors {
		vectors[i] = make([][]float64, 50)
		index := 0
		for _, flower := range iris {
			label := Labels[flower.Label]
			if i == label {
				vectors[i][index] = append(vectors[i][index], flower.Measures...)
				index++
			}
		}
	}
	rng := rand.New(rand.NewSource(1))
	var A, AI, u [3]Matrix
	cal := [][]float64{}
	for i := range vectors {
		A[i], AI[i], u[i] = NewMultiVariateGaussian(rng, Inverse[i], 4, vectors[i])
		diff := A[i].MulT(AI[i])
		c := make([]float64, 5)
		for ii, value := range diff.Data {
			if ii%5 == ii/5 {
				value -= 1
			}
			c[ii%5] += value
		}
		cal = append(cal, c)
	}
	fmt.Println("cal=")
	fmt.Println(cal)
	fmt.Println()

	{
		correct := 0
		var histogram [150][3]int
		for range 256 {
			for i := range iris {
				vector := NewMatrix(4, 1)
				vector.Data = append(vector.Data, iris[i].Measures...)
				min, index := math.MaxFloat64, 0
				for ii := range AI {
					reverse := AI[ii].T().MulT(vector.Sub(u[ii]))
					for iii := range reverse.Data {
						reverse.Data[iii] *= rng.NormFloat64()
					}
					forward := A[ii].MulT(reverse).Add(u[ii])
					fitness := L2(vector.Data, forward.Data)
					if fitness < min {
						min, index = fitness, ii
					}

				}
				histogram[i][index]++
			}
		}
		for i := range histogram {
			max, index := 0, 0
			for ii := range histogram[i] {
				if histogram[i][ii] > max {
					max, index = histogram[i][ii], ii
				}
			}
			if index == Labels[iris[i].Label] {
				correct++
			}
		}
		fmt.Println(correct, "/", len(iris), "=", float64(correct)/float64(len(iris)))
	}

	type Result struct {
		Feature int
		Fitness float64
	}

	done := make(chan [150][3]uint64, 8)
	process := func(seed int64) {
		rng := rand.New(rand.NewSource(seed))
		var histogram [150][3]uint64
		for i, flower := range iris {
			vector := flower.Measures
			min, index := math.MaxFloat64, 0
			for range 16 * 33 {
				for ii := range A {
					g := NewMatrix(4, 1)
					for iii := range 4 {
						_ = iii
						g.Data = append(g.Data, rng.NormFloat64())
					}
					s := A[ii].MulT(g).Add(u[ii])
					fitness := L2(s.Data, vector)
					if fitness < min {
						min, index = fitness, ii
					}
				}
			}
			histogram[i][index]++
		}
		done <- histogram
	}

	var histogram [150][3]uint64
	cpus := runtime.NumCPU()
	flight, i := 0, 0
	const iterations = 16
	for i < iterations && flight < cpus {
		go process(rng.Int63())
		i++
		flight++
	}
	for i < iterations {
		h := <-done
		for ii := range h {
			for iii, counts := range h[ii] {
				histogram[ii][iii] += counts
			}
		}
		flight--

		go process(rng.Int63())
		i++
		flight++
	}
	for range flight {
		h := <-done
		for ii := range h {
			for iii, counts := range h[ii] {
				histogram[ii][iii] += counts
			}
		}
	}

	correct := 0
	for i := range histogram {
		max, index := uint64(0), 0
		for ii, count := range histogram[i] {
			if count > max {
				max, index = count, ii
			}
		}
		if Labels[iris[i].Label] == index {
			correct++
		}
	}
	fmt.Println(correct, "/", len(iris), "=", float64(correct)/float64(len(iris)))
}
