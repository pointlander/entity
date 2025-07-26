// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"sort"
)

// FF is the feed forward mode
func FF() {
	iris := Load()
	rng := rand.New(rand.NewSource(1))
	fitness := func(g []float32) (int, float64) {
		fitness := 0.0 //150.0
		correct := 0
		l1 := NewMatrix[float32](4, 4, g[:16]...)
		b1 := NewMatrix[float32](4, 1, g[16:20]...)
		l2 := NewMatrix[float32](4, 3, g[20:32]...)
		b2 := NewMatrix[float32](3, 1, g[32:35]...)
		for _, flower := range iris {
			input := NewMatrix[float32](4, 1)
			for _, measure := range flower.Measures {
				input.Data = append(input.Data, float32(measure))
			}
			output := l1.MulT(input).Add(b1).Sigmoid()
			output = l2.MulT(output).Add(b2).Softmax(1)
			/*target := [3]float32{}
			target[Labels[flower.Label]] = 1.0
			for i, value := range output.Data {
				diff := value - target[i]
				fitness += float64(diff * diff)
			}*/
			diff := output.Data[Labels[flower.Label]] - 1
			fitness += float64(diff * diff)
			max, index := float32(0.0), 0
			for i, value := range output.Data {
				if value > max {
					max, index = value, i
				}
			}
			if Labels[flower.Label] == index {
				correct++
			}
		}
		return correct, fitness
	}

	type Number struct {
		Number  Matrix[float32]
		Fitness float64
		Correct int
	}

	const (
		width      = 4*4 + 4 + 4*3 + 3
		models     = width / width
		iterations = 1024
		population = 8 * 1024
		cut        = 512
	)

	state := make([][]float32, cut)
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
			a[ii], _, u[ii] = NewMultiVariateGaussian(.0001, 1.0e-1, graph, false, rng, fmt.Sprintf("number_%d", i), width, s)
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
			correct, fit := fitness(born[ii].Number.Data)
			born[ii].Fitness = fit
			born[ii].Correct = correct
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
		fmt.Println(pop[0].Fitness, pop[0].Correct)
		if pop[0].Correct >= 149 {
			g := pop[0].Number.Data
			correct := 0
			l1 := NewMatrix[float32](4, 4, g[:16]...)
			b1 := NewMatrix[float32](4, 1, g[16:20]...)
			l2 := NewMatrix[float32](4, 3, g[20:32]...)
			b2 := NewMatrix[float32](3, 1, g[32:35]...)
			for _, flower := range iris {
				input := NewMatrix[float32](4, 1)
				for _, measure := range flower.Measures {
					input.Data = append(input.Data, float32(measure))
				}
				output := l1.MulT(input).Add(b1).Sigmoid()
				output = l2.MulT(output).Add(b2).Softmax(1)
				max, index := float32(0.0), 0
				for i, value := range output.Data {
					if value > max {
						max, index = value, i
					}
				}
				if Labels[flower.Label] == index {
					correct++
				}
				fmt.Println(flower.Label, index)
			}

			break
		}
	}

}
