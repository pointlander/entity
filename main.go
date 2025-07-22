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
	"math/rand"
	"runtime"
	"sort"
	"strconv"
	"strings"
	//"github.com/pointlander/compress"
	//"github.com/texttheater/golang-levenshtein/levenshtein"
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
	// FlagQueens 8 queens problem
	FlagQueens = flag.Bool("queens", false, "8 queens problem")
	// FlagBuild build the model
	FlagBuild = flag.Bool("build", false, "build the model")
)

//go:embed books/*
var Data embed.FS

const (
	// MemorySize is the size of the working memory
	MemorySize = 1024 * 1024
	// CyclesLimit is the limit on cycles
	CyclesLimit = 1024 * 1024
)

var (
	// Genes are the genes
	Genes = [...]rune{'?', '+', '-', '>', '<', '.', '[', ']'}
)

// Program is a program
// https://github.com/cvhariharan/goBrainFuck
type Program []rune

// Execute executes a program
func (p Program) Execute(rng *rand.Rand, size int) *strings.Builder {
	var (
		memory [MemorySize]int
		pc     int
		dc     int
		i      int
		output strings.Builder
	)
	length := len(p)

	for pc < length && i < CyclesLimit {
		opcode := p[pc]
		switch opcode {
		case '?':
		case '+':
			memory[dc%MemorySize] += 1
			pc++
		case '-':
			memory[dc%MemorySize] -= 1
			pc++
		case '>':
			dc++
			pc++
		case '<':
			if dc > 0 {
				dc--
			}
			pc++
		case '.':
			m := memory[dc%MemorySize]
			if m < 0 {
				m = -m
			}
			output.WriteRune(rune(m))
			if len([]rune(output.String())) == size {
				return &output
			}
			pc++
		case ',':
			memory[dc] = rng.Intn(len(Genes))
			pc++
		case '[':
			if memory[dc] == 0 {
				pc = p.findMatchingForward(pc) + 1
			} else {
				pc++
			}
		case ']':
			if memory[dc] != 0 {
				pc = p.findMatchingBackward(pc) + 1
			} else {
				pc++
			}
		default:
			pc++
		}
		i++
	}
	return &output
}

func (p Program) findMatchingForward(position int) int {
	count, length := 1, len(p)
	for i := position + 1; i < length; i++ {
		if p[i] == ']' {
			count--
			if count == 0 {
				return i
			}
		} else if p[i] == '[' {
			count++
		}
	}

	return length - 1
}

func (p Program) findMatchingBackward(position int) int {
	count := 1
	for i := position - 1; i >= 0; i-- {
		if p[i] == '[' {
			count--
			if count == 0 {
				return i
			}
		} else if p[i] == ']' {
			count++
		}
	}

	return -1
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

	if *FlagQueens {
		Queens()
		return
	}

	rng := rand.New(rand.NewSource(1))
	fitness := func(g []float32, rng *rand.Rand) (string, float64) {
		program := Program{}
		for x := 0; x < 128; x++ {
			y := uint(0)
			for yy := range 3 {
				y <<= 1
				if g[3*x+yy] > 0 {
					y |= 1
				}
			}
			program = append(program, Genes[y])
		}
		target := []rune("Hello World!")
		output := program.Execute(rng, len(target))
		found := []rune(output.String())
		fitness := 0.0
		for i := len(found); i < len(target); i++ {
			found = append(found, 0)
		}
		for i, value := range found {
			diff := target[i] - value
			fitness += float64(diff * diff)
		}
		return output.String(), fitness
		//return float64(levenshtein.DistanceForStrings([]rune(output.String()), target, levenshtein.DefaultOptions))
		//if output.Len() > 0 && output.String()[0] == 'H' {
		//	return 0
		//}
		//target = append(target, []byte(output.String())...)
		//buffer := bytes.Buffer{}
		//compress.Mark1Compress1(target, &buffer)
		//return float64(buffer.Len()) / float64(len(target))
	}

	type Number struct {
		Number  Matrix[float32]
		Fitness float64
		Output  string
	}

	const (
		width      = 128 * 3
		models     = width / (32 * 3)
		iterations = 1024
		population = 1024
		cut        = 256
	)

	state := make([][]float32, cut)
	for i := range state {
		for range width {
			state[i] = append(state[i], float32(rng.NormFloat64()))
		}
	}
	pop := make([]Number, population)

	last := 0.0
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
			a[ii], _, u[ii] = NewMultiVariateGaussian(.0001, 1.0e-1, graph, false, rng, fmt.Sprintf("bf_%d", i), 32*3, s)
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
			born = pop[cut:]
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
			output, fitness := fitness(born[ii].Number.Data, rng)
			born[ii].Fitness = float64(fitness)
			born[ii].Output = output
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
		if pop[0].Fitness == 0 {
			for x := 0; x < 128; x++ {
				y := uint(0)
				for yy := range 3 {
					y <<= 1
					if pop[0].Number.Data[3*x+yy] > 0 {
						y |= 1
					}
				}
				fmt.Printf("%c, ", Genes[y])
			}
			fmt.Println()
			fmt.Println([]byte(pop[0].Output))
			break
		}
		if pop[0].Fitness != last {
			for x := 0; x < 128; x++ {
				y := uint(0)
				for yy := range 3 {
					y <<= 1
					if pop[0].Number.Data[3*x+yy] > 0 {
						y |= 1
					}
				}
				fmt.Printf("%c, ", Genes[y])
			}
			fmt.Println()
			fmt.Println(pop[0].Output)
			last = pop[0].Fitness
		}
	}
}
