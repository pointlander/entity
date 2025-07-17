// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"compress/bzip2"
	"embed"
	"encoding/csv"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/pointlander/gradient/tf32"
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

var log bool

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

// L2 is the L2 norm
func L2(a, b []float64) float64 {
	c := 0.0
	for i, v := range a {
		diff := v - b[i]
		c += diff * diff
	}
	return c
}

// IrisModel the iris model
func IrisModel() {
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
	var A, AI, u [3]Matrix[float64]
	cal := [][]float64{}
	for i := range vectors {
		A[i], AI[i], u[i] = NewMultiVariateGaussian[float64](-1, Eta, true, true, rng, Inverse[i], 4, vectors[i])
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
		start := time.Now()
		done := make(chan [150][3]uint64)
		process := func(seed int64) {
			rng := rand.New(rand.NewSource(seed))
			var histogram [150][3]uint64
			for i := range iris {
				vector := NewMatrix[float64](4, 1)
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
		elapsed := time.Since(start)
		fmt.Println(elapsed, correct, "/", len(iris), "=", float64(correct)/float64(len(iris)))
	}

	start := time.Now()
	done := make(chan [150][3]uint64, 8)
	process := func(seed int64) {
		rng := rand.New(rand.NewSource(seed))
		var histogram [150][3]uint64
		for i, flower := range iris {
			vector := flower.Measures
			min, index := math.MaxFloat64, 0
			for range 16 * 33 {
				for ii := range A {
					g := NewMatrix[float64](4, 1)
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
	elapsed := time.Since(start)
	fmt.Println(elapsed, correct, "/", len(iris), "=", float64(correct)/float64(len(iris)))
}

// Text is the text model
func Text() {
	file, err := Data.Open("books/100.txt.utf-8.bz2")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := bzip2.NewReader(file)
	data, err := io.ReadAll(reader)
	if err != nil {
		panic(err)
	}

	forward, reverse, code := make(map[rune]byte), make(map[byte]rune), byte(0)
	for _, v := range string(data) {
		if _, ok := forward[v]; !ok {
			forward[v] = code
			reverse[code] = v
			code++
			if code > 255 {
				panic("not enough codes")
			}
		}
	}
	length, datum := len(forward), []rune(string(data))

	A, AI, u := make([]Matrix[float64], length), make([]Matrix[float64], length), make([]Matrix[float64], length)
	if *FlagBuild {
		out, err := os.Create("model.bin")
		if err != nil {
			panic(err)
		}
		defer out.Close()

		rng := rand.New(rand.NewSource(1))
		for i := range length {
			vectors, index := make([][]float64, 0, 8), 8
			for _, v := range datum[8:] {
				if int(forward[v]) == i {
					vector := make([]float64, length)
					for i := 1; i < 9; i++ {
						vector[forward[datum[index-i]]]++
					}
					vectors = append(vectors, vector)
				}
				index++
			}
			A[i], AI[i], u[i] = NewMultiVariateGaussian[float64](-1.0, 1.0e-1, true, true, rng, fmt.Sprintf("%d_text", i), length, vectors)

			buffer64 := make([]byte, 8)
			for _, parameter := range u[i].Data {
				bits := math.Float64bits(parameter)
				for i := range buffer64 {
					buffer64[i] = byte((bits >> (8 * i)) & 0xFF)
				}
				n, err := out.Write(buffer64)
				if err != nil {
					panic(err)
				}
				if n != len(buffer64) {
					panic("8 bytes should be been written")
				}
			}
			for _, parameter := range A[i].Data {
				bits := math.Float64bits(parameter)
				for i := range buffer64 {
					buffer64[i] = byte((bits >> (8 * i)) & 0xFF)
				}
				n, err := out.Write(buffer64)
				if err != nil {
					panic(err)
				}
				if n != len(buffer64) {
					panic("8 bytes should be been written")
				}
			}
			for _, parameter := range AI[i].Data {
				bits := math.Float64bits(parameter)
				for i := range buffer64 {
					buffer64[i] = byte((bits >> (8 * i)) & 0xFF)
				}
				n, err := out.Write(buffer64)
				if err != nil {
					panic(err)
				}
				if n != len(buffer64) {
					panic("8 bytes should be been written")
				}
			}
		}
		return
	}

	input, err := os.Open("model.bin")
	if err != nil {
		panic(err)
	}
	defer input.Close()

	for i := range length {
		u[i] = NewMatrix[float64](length, 1)
		buffer64 := make([]byte, 8)
		for range u[i].Rows {
			for range u[i].Cols {
				n, err := input.Read(buffer64)
				if err == io.EOF {
					panic(err)
				} else if err != nil {
					panic(err)
				}
				if n != len(buffer64) {
					panic(fmt.Errorf("not all bytes read: %d", n))
				}
				value := uint64(0)
				for k := 0; k < 8; k++ {
					value <<= 8
					value |= uint64(buffer64[7-k])
				}
				u[i].Data = append(u[i].Data, math.Float64frombits(value))
			}
		}
		A[i] = NewMatrix[float64](length, length)
		for range A[i].Rows {
			for range A[i].Cols {
				n, err := input.Read(buffer64)
				if err == io.EOF {
					panic(err)
				} else if err != nil {
					panic(err)
				}
				if n != len(buffer64) {
					panic(fmt.Errorf("not all bytes read: %d", n))
				}
				value := uint64(0)
				for k := 0; k < 8; k++ {
					value <<= 8
					value |= uint64(buffer64[7-k])
				}
				A[i].Data = append(A[i].Data, math.Float64frombits(value))
			}
		}
		AI[i] = NewMatrix[float64](length, length)
		for range AI[i].Rows {
			for range AI[i].Cols {
				n, err := input.Read(buffer64)
				if err == io.EOF {
					panic(err)
				} else if err != nil {
					panic(err)
				}
				if n != len(buffer64) {
					panic(fmt.Errorf("not all bytes read: %d", n))
				}
				value := uint64(0)
				for k := 0; k < 8; k++ {
					value <<= 8
					value |= uint64(buffer64[7-k])
				}
				AI[i].Data = append(AI[i].Data, math.Float64frombits(value))
			}
		}
	}
	buffer64 := make([]byte, 8)
	_, err = input.Read(buffer64)
	if err != io.EOF {
		panic("not at the end")
	}

	for i := range length {
		x := A[i].MulT(AI[i])
		count, total := 0, 0
		for ii := range x.Rows {
			for iii := range x.Cols {
				if ii == iii {
					if x.Data[ii*x.Cols+iii] < .9 {
						count++
					}
					total++
				}
			}
		}
		fmt.Println(i, count, total, float64(count)/float64(total))
	}

	rng := rand.New(rand.NewSource(1))
	grandPrompt, grandMax := []rune{}, 0
	for range 33 {
		prompt, grand := []rune("What is the meaning of life?"), 0
		const iterations = 128
		for range 8 {
			vector := NewMatrix(length, 1, make([]float64, length)...)
			for i := 1; i < 9; i++ {
				vector.Data[forward[prompt[len(prompt)-i]]]++
			}
			histogram := make([]int, length)
			for range iterations {
				min, index := math.MaxFloat64, 0
				for i := range length {
					if i == 0 {
						continue
					}
					reverse := AI[i].T().MulT(vector.Sub(u[i]))
					for iii := range reverse.Data {
						reverse.Data[iii] *= rng.NormFloat64()
					}
					forward := A[i].MulT(reverse).Add(u[i])
					fitness := L2(vector.Data, forward.Data)
					if fitness < min {
						min, index = fitness, i
					}
				}
				histogram[index]++
			}
			sum, index, sample := 0, 0, rng.Intn(iterations)
			for i, count := range histogram {
				sum += count
				if sample < sum {
					grand += count
					index = i
					break
				}
			}
			fmt.Printf("%c %d\n", reverse[byte(index)], reverse[byte(index)])
			prompt = append(prompt, reverse[byte(index)])
		}
		fmt.Println(grand, string(prompt))
		if grand > grandMax {
			grandMax, grandPrompt = grand, prompt
		}
	}
	fmt.Println(grandMax, string(grandPrompt))
}

// Image is the image model
func Image() {
	rng := rand.New(rand.NewSource(1))
	var state [8][][]float64
	for i := range state {
		state[i] = make([][]float64, 8)
		for ii := range state[i] {
			for range 64 {
				state[i][ii] = append(state[i][ii], rng.NormFloat64())
			}
			fmt.Println(state[i])
		}
	}
	type Entity struct {
		Vector  [8]Matrix[float64]
		Fitness float64
	}
	const iterations = 256
	for i := 0; i < iterations; i++ {
		graph := i == 0 || i == iterations-1
		var a, u [8]Matrix[float64]
		for ii := range a {
			a[ii], _, u[ii] = NewMultiVariateGaussian[float64](.0001, 1.0e-1, graph, false, rng, fmt.Sprintf("entropy_%d", i), 64, state[ii])
		}
		pop := make([]Entity, 256)
		for ii := range pop {
			img := image.NewGray(image.Rect(0, 0, 8, 8))
			for v := range a {
				g := NewMatrix[float64](64, 1)
				for range 8 {
					g.Data = append(g.Data, rng.NormFloat64())
				}
				pop[ii].Vector[v] = a[v].MulT(g).Add(u[v])
				for iii := range 8 {
					for iv := range 8 {
						if pop[ii].Vector[v].Data[iii*8+iv] > 0 {
							pixel := img.GrayAt(iii, iv)
							pixel.Y |= 1 << v
							img.SetGray(iii, iv, pixel)
						}
					}
				}
			}
			buffer := bytes.Buffer{}
			err := jpeg.Encode(&buffer, img, nil)
			if err != nil {
				panic(err)
			}
			pop[ii].Fitness = float64(buffer.Len())
			/*var histogram [2]float64
			for _, value := range pop[ii].Vector.Data {
				if value < 0 {
					histogram[0]++
				} else {
					histogram[1]++
				}
			}
			fitness := 0.0
			for _, value := range histogram {
				if value == 0 {
					continue
				}
				fitness += (value / 8.0) * math.Log2(value/8.0)
			}
			pop[ii].Fitness = math.Abs(fitness)*/
		}
		sort.Slice(pop, func(i, j int) bool {
			return pop[i].Fitness < pop[j].Fitness
		})
		for v := range state {
			for ii := range 8 {
				copy(state[v][ii], pop[ii].Vector[v].Data)
				if log {
					fmt.Println(state[v][ii])
				}
			}
		}
		{
			img := image.NewGray(image.Rect(0, 0, 8, 8))
			for v := range a {
				for iii := range 8 {
					for iv := range 8 {
						if pop[0].Vector[v].Data[iii*8+iv] > 0 {
							pixel := img.GrayAt(iii, iv)
							pixel.Y |= 1 << v
							img.SetGray(iii, iv, pixel)
						}
					}
				}
			}
			output, err := os.Create(fmt.Sprintf("img_%d.jpg", i))
			if err != nil {
				panic(err)
			}
			err = jpeg.Encode(output, img, nil)
			if err != nil {
				panic(err)
			}
			output.Close()
		}
		fmt.Println(pop[0].Fitness)
	}
}

var (
	// FlagIris the iris model
	FlagIris = flag.Bool("iris", false, "the iris model")
	// FlagText text model
	FlagText = flag.Bool("text", false, "the text model")
	// FlagImage the image mode
	FlagImage = flag.Bool("image", false, "the image mode")
	// FlagBuild build the model
	FlagBuild = flag.Bool("build", false, "build the model")
)

//go:embed books/*
var Data embed.FS

func main() {
	flag.Parse()

	if *FlagIris {
		IrisModel()
		return
	}

	if *FlagText {
		Text()
		return
	}

	if *FlagImage {
		Image()
		return
	}

	rng := rand.New(rand.NewSource(1))
	type RNN struct {
		Layer   Matrix[float32]
		Bias    Matrix[float32]
		Fitness float64
	}

	const (
		size       = 256
		width      = size*size + size
		models     = width / 8
		iterations = 256
		population = 256
	)

	file, err := Data.Open("books/100.txt.utf-8.bz2")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := bzip2.NewReader(file)
	data, err := io.ReadAll(reader)
	if err != nil {
		panic(err)
	}
	forward, reverse, code := make(map[rune]byte), make(map[byte]rune), byte(0)
	for _, v := range string(data) {
		if _, ok := forward[v]; !ok {
			forward[v] = code
			reverse[code] = v
			code++
			if code > 255 {
				panic("not enough codes")
			}
		}
	}

	state := make([][]float32, 8)
	for i := range state {
		for range width {
			state[i] = append(state[i], float32(rng.NormFloat64()))
		}
	}
	pop := make([]RNN, population)
	text := []rune(string(data))
	for i := 0; i < iterations; i++ {
		graph := false //i == 0 || i == iterations-1
		translate := make([]int, width)
		for i := range translate {
			translate[i] = i % models
		}
		rng.Shuffle(width, func(i, j int) {
			translate[i], translate[j] = translate[j], translate[i]
		})
		var a, u [models]Matrix[float32]
		done := make(chan bool, 8)
		process := func(ii int, seed int64) {
			rng := rand.New(rand.NewSource(seed))
			s := make([][]float32, 8)
			for iii := range state {
				for iv, t := range translate {
					if t == ii {
						s[iii] = append(s[iii], state[iii][iv])
					}
				}
			}
			a[ii], _, u[ii] = NewMultiVariateGaussian(.0001, 1.0e-1, graph, false, rng, fmt.Sprintf("rnn_%d", i), 8, s)
			done <- true
		}
		ii, flight, cpus := 0, 0, runtime.NumCPU()
		for ii < models && flight < cpus {
			go process(ii, rng.Int63())
			flight++
			ii++
		}
		for ii < models {
			<-done
			flight--

			go process(ii, rng.Int63())
			flight++
			ii++
		}
		for range flight {
			<-done
		}

		start := rng.Intn(len(text) - 1024)
		end := start + 1024
		context := rng.Intn(128-16) + 16

		born := pop
		if i > 0 {
			born = pop[8:]
		}
		learn := func(ii int, seed int64) {
			rng := rand.New(rand.NewSource(seed))
			vector := NewMatrix[float32](width, 1)
			vector.Data = make([]float32, width)
			for iii := range a {
				g := NewMatrix[float32](a[iii].Cols, 1)
				for range a[iii].Cols {
					g.Data = append(g.Data, float32(rng.NormFloat64()))
				}
				vec, index := a[iii].MulT(g).Add(u[iii]), 0
				for iv, t := range translate {
					if t == iii {
						vector.Data[iv] = vec.Data[index]
						index++
					}
				}
			}
			born[ii].Layer = NewMatrix(size, size, vector.Data[:size*size]...)
			born[ii].Bias = NewMatrix(size, 1, vector.Data[size*size:width]...)
			born[ii].Fitness = 0.0
			input := NewMatrix[float32](size, 1)
			input.Data = make([]float32, size)
			last := -1
			for iii, symbol := range text[start:end] {
				if iii < context {
					for iv := range input.Data {
						input.Data[iv] = 0
					}
					if last >= 0 {
						input.Data[last] = 1
					}
				}
				input = born[ii].Layer.MulT(input).Add(born[ii].Bias).Sigmoid()
				target := forward[symbol]
				for iv := range len(forward) {
					var diff float32
					if iv == int(target) {
						diff = input.Data[iv] - 1
					} else {
						diff = input.Data[iv] - 0
					}
					born[ii].Fitness += float64(diff * diff)
				}
				last = int(uint(target))
			}
			done <- true
		}
		ii, flight, cpus = 0, 0, runtime.NumCPU()
		for ii < len(born) && flight < cpus {
			go learn(ii, rng.Int63())
			flight++
			ii++
		}
		for ii < len(born) {
			<-done
			flight--

			go learn(ii, rng.Int63())
			flight++
			ii++
		}
		for range flight {
			<-done
		}

		sort.Slice(pop, func(i, j int) bool {
			return pop[i].Fitness < pop[j].Fitness
		})
		for ii := range state {
			copy(state[ii], pop[ii].Layer.Data)
			copy(state[ii][16:], pop[ii].Bias.Data)
		}
		fmt.Println(pop[0].Fitness)
	}
}
