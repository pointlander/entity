// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/big"
	"math/rand"
	"os"
	"runtime"
	"sort"
)

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
