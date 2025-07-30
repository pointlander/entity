// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math/rand"
	"runtime"
	"sort"
)

// T is a transformer
func T() {
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

	rng := rand.New(rand.NewSource(1))

	coded := make([]byte, 0, 8)
	for _, v := range string(data) {
		coded = append(coded, forward[v])
	}
	type Sample struct {
		Input  []byte
		Output []byte
		Target byte
	}
	samples := make([]Sample, 1000)
	for i := range samples {
		begin := rng.Intn(len(coded) - 100)
		end := begin + 1 + rng.Intn(99)
		samples[i].Input = coded[begin:end]
		begin, end = end, end+1+rng.Intn(99)
		samples[i].Output = coded[begin:end]
		samples[i].Target = coded[end]
	}

	set := Set[float32]{
		Sizes: []Size{
			{"itags", 8, 100},
			{"otags", 8, 100},
			{"lembeddingIn", 8 + 256, 32},
			{"bembeddingIn", 32, 1},
			{"inQ", 32, 32},
			{"inK", 32, 32},
			{"inV", 32, 32},
			{"l1In", 32, 32},
			{"b1In", 32, 1},
			{"lembeddingOut", 8 + 256, 32},
			{"bembeddingOut", 32, 1},
			{"outQ1", 32, 32},
			{"outK1", 32, 32},
			{"outV1", 32, 32},
			{"outQ2", 32, 32},
			{"outK2", 32, 32},
			{"outV2", 32, 32},
			{"l1Out", 32, 32},
			{"b1Out", 32, 1},
			{"linear", 32, 256},
		},
	}
	width := set.Size()
	models := width / 32
	const (
		iterations = 1024
		population = 1024
		cut        = 512
	)
	fmt.Println(width, set.Size())

	g := make([]float32, set.Size())
	for i := range g {
		g[i] = float32(rng.NormFloat64())
	}

	fitness := func(g []float32, set Set[float32]) float64 {
		fitness := 0.0
		inputs := NewMatrix[float32](256, 100)
		for range inputs.Cols * inputs.Rows {
			inputs.Data = append(inputs.Data, 0)
		}
		outputs := NewMatrix[float32](256, 100)
		for range outputs.Cols * outputs.Rows {
			outputs.Data = append(outputs.Data, 0)
		}
		s := NewMatrices(set, g)
		for _, sample := range samples {
			for i := range inputs.Data {
				inputs.Data[i] = 0
			}
			for i := range inputs.Rows {
				if i < len(sample.Input) {
					inputs.Data[i*inputs.Cols+int(sample.Input[i])] = 1
				}
			}
			for i := range outputs.Data {
				outputs.Data[i] = 0
			}
			for i := range outputs.Rows {
				if i < len(sample.Output) {
					outputs.Data[i*outputs.Cols+int(sample.Output[i])] = 1
				}
			}
			output := Transformer(s, inputs, outputs)
			diff := output.Data[sample.Target] - 1
			fitness += float64(diff * diff)
		}
		return fitness
	}

	type Number struct {
		Number  Matrix[float32]
		Fitness float64
	}

	fmt.Println(fitness(g, set))
	state := make([][]float32, cut)
	for i := range state {
		for range width {
			state[i] = append(state[i], float32(rng.NormFloat64()))
		}
	}
	pop := make([]Number, population)

	for i := 0; i < iterations; i++ {
		translate := make([]int, width)
		for i := range translate {
			translate[i] = i % models
		}
		rng.Shuffle(width, func(i, j int) {
			translate[i], translate[j] = translate[j], translate[i]
		})
		a, u := make([]Matrix[float32], models), make([]Matrix[float32], models)
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
			a[ii], _, u[ii] = NewMultiVariateGaussian(.0001, 1.0e-1, false, false, rng, fmt.Sprintf("transformer_%d", i), 32, s)
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
			fit := fitness(born[ii].Number.Data, set)
			born[ii].Fitness = fit
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
		if pop[0].Fitness <= 100 {
			break
		}
	}
}
