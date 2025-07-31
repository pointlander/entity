// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
)

// Entropy is the entropy mode
func Entropy() {
	rng := rand.New(rand.NewSource(1))
	fitness := func(g []float32, set Set[float32]) float64 {
		fitness := 0.0 //150.0
		h1 := [2]float64{}
		s := NewMatrices(set, g)
		ss := SelfAttention(s.ByIndex[0], s.ByIndex[0], s.ByIndex[0])
		for _, value := range ss.Data {
			if value > 0 {
				h1[0]++
			} else {
				h1[1]++
			}
		}
		h2 := [2]float64{}
		for _, value := range g {
			if value > 0 {
				h2[0]++
			} else {
				h2[1]++
			}
		}
		sum := 0.0
		for _, value := range h1 {
			sum += value
		}
		a := 0.0
		for _, value := range h1 {
			if value == 0 || sum == 0 {
				continue
			}
			a -= (value / sum) * math.Log2(value/sum)
		}
		sum = 0.0
		for _, value := range h2 {
			sum += value
		}
		b := 0.0
		for _, value := range h2 {
			if value == 0 || sum == 0 {
				continue
			}
			b -= (value / sum) * math.Log2(value/sum)
		}
		diff := b / a
		_ = diff
		//fitness += diff * diff
		for i, value := range g {
			diff := value - ss.Data[i]
			fitness += float64(diff * diff)
		}
		return fitness
	}

	type Number struct {
		Number  Matrix[float32]
		Fitness float64
	}

	set := Set[float32]{
		Sizes: []Size{
			{"e", 8, 8},
		},
	}
	width := set.Size()
	models := width / width
	const (
		iterations = 1024
		population = 1024
		cut        = 128
	)

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
			a[ii], _, u[ii] = NewMultiVariateGaussian(.0001, 1.0e-1, false, false, rng, fmt.Sprintf("ff_%d", i), width, s)
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
		if pop[0].Fitness < .01 {
			g := pop[0].Number.Data
			s := NewMatrices(set, g)
			fmt.Println("input")
			for ii := range s.ByIndex[0].Rows {
				for iii := range s.ByIndex[0].Cols {
					if s.ByIndex[0].Data[ii*s.ByIndex[0].Cols+iii] > 0 {
						fmt.Printf(" 1")
					} else {
						fmt.Printf(" 0")
					}
				}
				fmt.Println()
			}
			fmt.Println()

			fmt.Println("output")
			ss := SelfAttention(s.ByIndex[0], s.ByIndex[0], s.ByIndex[0])
			for ii := range ss.Rows {
				for iii := range ss.Cols {
					if ss.Data[ii*ss.Cols+iii] > 0 {
						fmt.Printf(" 1")
					} else {
						fmt.Printf(" 0")
					}
				}
				fmt.Println()
			}
			break
		}
	}

}
