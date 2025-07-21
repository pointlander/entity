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
	"math/big"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
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

// L2 is the L2 norm
func L2(a, b []float64) float64 {
	c := 0.0
	for i, v := range a {
		diff := v - b[i]
		c += diff * diff
	}
	return c
}

var (
	// FlagIris the iris model
	FlagIris = flag.Bool("iris", false, "the iris model")
	// FlagText text model
	FlagText = flag.Bool("text", false, "the text model")
	// FlagImage the image model
	FlagImage = flag.Bool("image", false, "the image model")
	// FlagRNN the rnn model
	FlagRNN = flag.Bool("rnn", false, "the rnn model")
	// FlagFactor factor a number
	FlagFactor = flag.Bool("factor", false, "factor a number")
	// FlagBuild build the model
	FlagBuild = flag.Bool("build", false, "build the model")
)

//go:embed books/*
var Data embed.FS

// Factor factors a number
func Factor() {
	rng := rand.New(rand.NewSource(1))
	type Number struct {
		Number  Matrix[float32]
		Fitness float64
	}

	const (
		width      = 1024
		models     = width / 64
		iterations = 16
		population = 1024
	)

	numbers := [2]*big.Int{}
	for i := range numbers {
		for {
			n := make([]byte, 2)
			for ii := range n {
				n[ii] = byte(rng.Intn(256))
			}
			b := big.NewInt(0)
			b.SetBytes(n)
			if b.ProbablyPrime(7) {
				numbers[i] = b
				break
			}
		}
	}
	target := big.NewInt(0)
	target.Mul(numbers[0], numbers[1])
	fmt.Println(numbers[0], numbers[1], target)

	for {
		fmt.Println("_______________")
		state := make([][]float32, 8)
		for i := range state {
			for range width {
				state[i] = append(state[i], float32(rng.NormFloat64()))
			}
		}
		pop := make([]Number, population)

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
				s := make([][]float32, len(state))
				for iii := range state {
					for iv, t := range translate {
						if t == ii {
							s[iii] = append(s[iii], state[iii][iv])
						}
					}
				}
				a[ii], _, u[ii] = NewMultiVariateGaussian(.0001, 1.0e-1, graph, false, rng, fmt.Sprintf("number_%d", i), 64, s)
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
				born[ii].Number = NewMatrix(width, 1, vector.Data...)
				born[ii].Fitness = 0.0
				number := big.NewInt(0)
				for iii, bit := range born[ii].Number.Data {
					if bit > 0 {
						number.SetBit(number, iii, 1)
					}

				}
				a := big.NewInt(0)
				b := big.NewInt(0)
				a.Set(number)
				b.Set(target)
				for b.Cmp(big.NewInt(0)) != 0 {
					c := big.NewInt(0)
					c.Mod(a, b)
					a.Set(b)
					b.Set(c)
					born[ii].Fitness++
				}
				if a.Cmp(big.NewInt(1)) != 0 {
					b.Set(target)
					fmt.Println(target, "/", a, "=", b.Div(target, a))
					os.Exit(0)
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
				copy(state[ii], pop[ii].Number.Data)
			}
			fmt.Println(pop[0].Fitness)
		}
	}
}

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

	if *FlagRNN {
		RNN()
		return
	}

	if *FlagFactor {
		Factor()
		return
	}

	rng := rand.New(rand.NewSource(1))
	fitness := func(g []float32) float64 {
		fitness := 0.0
		boards := make([][]float64, 0, 8)
		for vi := 0; vi < 8; vi++ {
			board := make([]float64, 8*8)
			offset := vi * 8
			x, y := uint(0), uint(0)
			for xx := range 3 {
				x <<= 1
				if g[offset*2*3+xx] > 0 {
					x |= 1
				}
			}
			for yy := range 3 {
				y <<= 1
				if g[offset*2*3+3+yy] > 0 {
					y |= 1
				}
			}
			board[y*8+x] = 1
			for v := 1; v < 8; v++ {
				rx, ry := uint(0), uint(0)
				for xx := range 3 {
					rx <<= 1
					if g[offset*2*3+3*v+xx] > 0 {
						rx |= 1
					}
				}
				for yy := range 3 {
					ry <<= 1
					if g[offset*2*3+3+3*v+yy] > 0 {
						ry |= 1
					}
				}
				which := 0
				for board[((y+ry)%8)*8+((x+rx)%8)] == 1 {
					if which == 0 {
						ry++
						which = 1
					} else {
						rx++
						which = 0
					}
				}
				board[((y+ry)%8)*8+((x+rx)%8)] = 1
			}
			sum := 0
			for _, value := range board {
				if value > 0 {
					sum++
				}
			}
			if sum != 8 {
				fmt.Println(board)
				panic("there should be 8")
			}
			for i := range 8 {
				for ii := range 8 {
					for iii := ii + 1; iii < 8; iii++ {
						if board[i*8+iii] > 0 {
							fitness++
							break
						}
					}
					for iii := ii - 1; iii >= 0; iii-- {
						if board[i*8+iii] > 0 {
							fitness++
							break
						}
					}
					for iii := i + 1; iii < 8; iii++ {
						if board[iii*8+ii] > 0 {
							fitness++
							break
						}
					}
					for iii := i - 1; iii >= 0; iii-- {
						if board[iii*8+ii] > 0 {
							fitness++
							break
						}
					}
					x, y := ii, i
					for {
						x++
						y++
						if x > 7 || y > 7 {
							break
						}
						if board[y*8+x] > 0 {
							fitness++
							break
						}
					}
					x, y = ii, i
					for {
						x++
						y--
						if x > 7 || y < 0 {
							break
						}
						if board[y*8+x] > 0 {
							fitness++
							break
						}
					}
					x, y = ii, i
					for {
						x--
						y++
						if x < 0 || y > 7 {
							break
						}
						if board[y*8+x] > 0 {
							fitness++
							break
						}
					}
					x, y = ii, i
					for {
						x--
						y--
						if x < 0 || y < 0 {
							break
						}
						if board[y*8+x] > 0 {
							fitness++
							break
						}
					}

				}
			}
			boards = append(boards, board)
		}
		board := make([]float64, 8*8)
		sum := 0.0
		for i := 0; i < 8*8; i++ {
			for _, b := range boards {
				board[i] += b[i]
				sum += b[i]
			}
		}
		entropy := 0.0
		for _, value := range board {
			if value == 0 || sum == 0 {
				continue
			}
			entropy += (value / sum) * math.Log2(value/sum)
		}
		return fitness * -entropy
	}
	board := make([]float32, 8*8*3*2)
	for i := range board {
		board[i] = float32(rng.NormFloat64())
	}
	fmt.Println(fitness(board))

	type Number struct {
		Number  Matrix[float32]
		Fitness float64
	}

	const (
		width      = 64 * 2 * 3
		models     = width / 64
		iterations = 1024
		population = 1024
	)

	state := make([][]float32, 256)
	for i := range state {
		for range width {
			state[i] = append(state[i], float32(rng.NormFloat64()))
		}
	}
	pop := make([]Number, population)

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
			s := make([][]float32, len(state))
			for iii := range state {
				for iv, t := range translate {
					if t == ii {
						s[iii] = append(s[iii], state[iii][iv])
					}
				}
			}
			a[ii], _, u[ii] = NewMultiVariateGaussian(.0001, 1.0e-1, graph, false, rng, fmt.Sprintf("number_%d", i), 64, s)
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
			born[ii].Number = NewMatrix(width, 1, vector.Data...)
			born[ii].Fitness = float64(fitness(born[ii].Number.Data))
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
			copy(state[ii], pop[ii].Number.Data)
		}
		fmt.Println(pop[0].Fitness)
	}
}
