// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"
)

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
