// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math/rand"
	"os"
	"runtime"
	"sort"
)

// RNN is the rnn model
func RNN() {
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

	if *FlagBuild {
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

			born := pop
			if i > 0 {
				born = pop[8:]
			}
			learn := func(ii int, seed int64) {
				start := rng.Intn(len(text) - 1024)
				end := start + 1024

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
				for _, symbol := range text[start:end] {
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
					for iv := range input.Data[:len(forward)] {
						input.Data[iv] = 0
					}
					input.Data[target] = 1
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
				copy(state[ii][size*size:], pop[ii].Bias.Data)
			}
			fmt.Println(pop[0].Fitness)
		}

		output, err := os.Create("model.bin")
		if err != nil {
			panic(err)
		}
		defer output.Close()
		err = pop[0].Layer.Write(output)
		if err != nil {
			panic(err)
		}
		pop[0].Bias.Write(output)
		if err != nil {
			panic(err)
		}
		return
	}

	input, err := os.Open("model.bin")
	if err != nil {
		panic(err)
	}
	defer input.Close()
	layer := NewMatrix[float32](size, size)
	bias := NewMatrix[float32](size, 1)
	err = layer.Read(input)
	if err != nil {
		panic(err)
	}
	err = bias.Read(input)
	if err != nil {
		panic(err)
	}

	{
		prompt := []rune("What color is the sky?")
		input := NewMatrix[float32](size, 1)
		input.Data = make([]float32, size)
		last := -1
		type Result struct {
			Result string
			Cost   float32
		}
		results := []Result{}
		for range 256 {
			output := []rune{}
			result := Result{}
			for _, s := range prompt {
				for iv := range input.Data[:len(forward)] {
					input.Data[iv] = 0
				}
				if last >= 0 {
					input.Data[last] = 1
				}
				input = layer.MulT(input).Add(bias).Sigmoid()
				last = int(uint(forward[s]))
				output = append(output, reverse[byte(last)])
			}
			for range 256 {
				input = layer.MulT(input).Add(bias).Sigmoid()
				dist := make([]float32, len(input.Data[:len(forward)]))
				copy(dist, input.Data[:len(forward)])
				sum := float32(0.0)
				for _, value := range dist {
					sum += value
				}
				for iv := range dist {
					dist[iv] /= sum
				}
				total, selected, symbol := float32(0.0), rng.Float32(), 0
				for iv := range len(forward) {
					total += dist[iv]
					if selected < total {
						symbol = iv
						break
					}
				}
				result.Cost += dist[symbol]
				output = append(output, reverse[byte(symbol)])
				for iv := range input.Data[:len(forward)] {
					input.Data[iv] = 0
				}
				input.Data[symbol] = 1
			}
			result.Result = string(output)
			results = append(results, result)
		}
		sort.Slice(results, func(i, j int) bool {
			return results[i].Cost > results[j].Cost
		})
		fmt.Println(results[0].Cost)
		fmt.Println("'" + results[0].Result + "'")
	}

}
