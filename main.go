// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"

	"github.com/pointlander/gradient/tf64"

	"github.com/alixaxel/pagerank"
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
				for i := range item[:4] {
					f, err := strconv.ParseFloat(item[i], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[i] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

// NewMultiVariateGaussian
func NewMultiVariateGaussian(rng *rand.Rand, name string, size int, vectors [][]float64) (A Matrix, u Matrix) {
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
		for j := range cov[i] {
			cov[i][j] = cov[i][j] / float64(len(vectors))
		}
	}
	for i := range cov {
		fmt.Println(cov[i])
	}
	fmt.Println(avg)
	fmt.Println()

	set := tf64.NewSet()
	set.Add("A", size, size)

	for i := range set.Weights {
		w := set.Weights[i]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float64, len(w.X))
		}
	}

	others := tf64.NewSet()
	others.Add("E", size, size)
	E := others.ByName["E"]
	for i := range cov {
		for j := range cov[i] {
			E.X = append(E.X, cov[i][j])
		}
	}

	loss := tf64.Sum(tf64.Quadratic(others.Get("E"), tf64.Mul(set.Get("A"), set.Get("A"))))

	points := make(plotter.XYs, 0, 8)
	for i := 0; i < 1024; i++ {
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
			for l, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][l] + (1-B1)*g
				v := B2*w.States[StateV][l] + (1-B2)*g*g
				w.States[StateM][l] = m
				w.States[StateV][l] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
		//fmt.Println(i, cost)
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

	A = NewMatrix(size, size)
	for _, v := range set.ByName["A"].X {
		A.Data = append(A.Data, float64(v))
	}
	u = NewMatrix(size, 1, avg...)
	return A, u
}

var (
	// FlagAll all in one
	FlagAll = flag.Bool("all", false, "all in one")
)

// Dot is the dot product
func Dot(a, b []float64) float64 {
	c := 0.0
	for i, v := range a {
		c += v * b[i]
	}
	return c
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
	flag.Parse()

	if *FlagAll {
		iris := Load()
		vectors := make([][]float64, len(iris))
		for i := range iris {
			vectors[i] = append(vectors[i], iris[i].Measures...)
			labels := make([]float64, 3)
			labels[Labels[iris[i].Label]] = 1
			vectors[i] = append(vectors[i], labels...)
		}

		rng := rand.New(rand.NewSource(1))
		A, u := NewMultiVariateGaussian(rng, "all", 7, vectors)

		type Result struct {
			D float64
			T [3]float64
		}

		correct := 0
		for k := range iris {
			results := make([]Result, 0, 3)
			for i := 0; i < 33; i++ {
				g := NewMatrix(7, 1)
				for j := 0; j < 7; j++ {
					g.Data = append(g.Data, rng.NormFloat64())
				}
				s := A.MulT(g).Add(u)
				result := Result{}
				for j, v := range s.Data[:4] {
					diff := v - iris[k].Measures[j]
					result.D += diff * diff
				}
				copy(result.T[:], s.Data[4:7])
				results = append(results, result)
			}
			sort.Slice(results, func(i, j int) bool {
				return results[i].D < results[j].D
			})
			index, max := 0, 0.0
			for i, v := range results[0].T {
				if v > max {
					index, max = i, v
				}
			}
			if Labels[iris[k].Label] == index {
				correct++
			}
		}
		fmt.Println(correct, float64(correct)/float64(len(iris)))
		return
	}

	iris := Load()
	var vectors [3][][]float64
	for j := range vectors {
		vectors[j] = make([][]float64, len(iris))
		for i := range iris {
			vectors[j][i] = append(vectors[j][i], iris[i].Measures...)
			labels := make([]float64, 1)
			if Labels[iris[i].Label] == j {
				labels[0] = 1
			}
			vectors[j][i] = append(vectors[j][i], labels...)
		}
	}
	rng := rand.New(rand.NewSource(1))
	var A, u [3]Matrix
	for i := range vectors {
		A[i], u[i] = NewMultiVariateGaussian(rng, fmt.Sprintf("%d", i), 5, vectors[i])
	}

	type Result struct {
		Feature int
		Value   int
		Fitness float64
	}

	correct := 0
	for k := range iris {
		results := make([]Result, 0, 3)
		vector := [][]float64{make([]float64, 0, 5), make([]float64, 0, 5)}
		vector[0] = append(vector[0], iris[k].Measures...)
		vector[0] = append(vector[0], 0)
		vector[1] = append(vector[1], iris[k].Measures...)
		vector[1] = append(vector[1], 1)
		for l := range A {
			graph := pagerank.NewGraph()
			vectors := make([][]float64, 0, 1024)
			vectors = append(vectors, vector[0])
			vectors = append(vectors, vector[1])
			for i := 0; i < 32*1024; i++ {
				g := NewMatrix(5, 1)
				for j := 0; j < 5; j++ {
					g.Data = append(g.Data, rng.NormFloat64())
				}
				s := A[l].MulT(g).Add(u[l])
				if i < 33 {
					vectors = append(vectors, s.Data)
				}
				for j := range vector {
					result := Result{
						Feature: l,
						Value:   j,
					}
					//result.Fitness = math.Sqrt(Dot(s.Data, vector)) / (math.Sqrt(Dot(s.Data, s.Data)) * math.Sqrt(Dot(vector, vector)))
					result.Fitness = L2(s.Data, vector[j])
					results = append(results, result)
				}
			}
			for i, v := range vectors {
				for j, vv := range vectors {
					vvv := math.Sqrt(L2(v, vv))
					if vvv > 0 {
						vvv = 1 / vvv
					}
					graph.Link(uint32(i), uint32(j), vvv)
				}
			}
			max, n := 0.0, uint32(0)
			graph.Rank(1.0, 1e-3, func(node uint32, rank float64) {
				if rank > max {
					max, n = rank, node
				}
			})
			fmt.Println(max, n)
		}
		avg, counts := [3]float64{}, [3]float64{}
		for i := range results {
			if results[i].Value == 1 {
				avg[results[i].Feature] += results[i].Fitness
				counts[results[i].Feature]++
			}
		}
		for i := range avg {
			avg[i] /= counts[i]
		}
		variance := [3]float64{}
		for i := range results {
			if results[i].Value == 1 {
				diff := results[i].Fitness - avg[results[i].Feature]
				variance[results[i].Feature] += diff * diff
			}
		}
		for i := range variance {
			variance[i] /= counts[i]
		}
		max, idx := 0.0, 0
		for i := range variance {
			if variance[i] > max {
				max, idx = variance[i], i
			}
		}
		fmt.Println(idx, iris[k].Label)
		sort.Slice(results, func(i, j int) bool {
			return results[i].Fitness < results[j].Fitness
		})
		index := 0
		for i := range results {
			if results[i].Value == 1 {
				index = results[i].Feature
				break
			}
		}
		if Labels[iris[k].Label] == index {
			correct++
		}
	}
	fmt.Println(correct, float64(correct)/float64(len(iris)))
}
