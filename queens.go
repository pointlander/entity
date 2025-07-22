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

// Queens is the 8 queens problem
func Queens() {
	rng := rand.New(rand.NewSource(1))
	fitness := func(g []float32) float64 {
		fitness := 0.0
		board := make([]float64, 8*8)
		for x := 0; x < 8; x++ {
			y := uint(0)
			for yy := range 3 {
				y <<= 1
				if g[3*x+yy] > 0 {
					y |= 1
				}
			}
			board[y*8+uint(x)] = 1
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
				if board[i*8+ii] == 0 {
					continue
				}
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
		return fitness
	}
	board := make([]float32, 8*3)
	for i := range board {
		board[i] = float32(rng.NormFloat64())
	}
	fmt.Println(fitness(board))

	type Number struct {
		Number  Matrix[float32]
		Fitness float64
	}

	const (
		width      = 8 * 3
		models     = width / (8 * 3)
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
			a[ii], _, u[ii] = NewMultiVariateGaussian(.0001, 1.0e-1, graph, false, rng, fmt.Sprintf("number_%d", i), 8*3, s)
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
		if pop[0].Fitness == 0 {
			for x := 0; x < 8; x++ {
				y := uint(0)
				for yy := range 3 {
					y <<= 1
					if pop[0].Number.Data[3*x+yy] > 0 {
						y |= 1
					}
				}
				fmt.Printf("%d, ", y)
			}
			fmt.Println()
			break
		}
	}

}
