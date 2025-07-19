// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"math/rand"
	"os"
	"sort"
)

// Image is the image model
func Image() {
	rng := rand.New(rand.NewSource(1))
	var state [8][][]float64
	for i := range state {
		state[i] = make([][]float64, 8)
		for ii := range state[i] {
			for range 64 {
				state[i][ii] = append(state[i][ii], rng.NormFloat64())
			}
			fmt.Println(state[i])
		}
	}
	type Entity struct {
		Vector  [8]Matrix[float64]
		Fitness float64
	}
	const iterations = 256
	for i := 0; i < iterations; i++ {
		graph := i == 0 || i == iterations-1
		var a, u [8]Matrix[float64]
		for ii := range a {
			a[ii], _, u[ii] = NewMultiVariateGaussian[float64](.0001, 1.0e-1, graph, false, rng, fmt.Sprintf("entropy_%d", i), 64, state[ii])
		}
		pop := make([]Entity, 256)
		for ii := range pop {
			img := image.NewGray(image.Rect(0, 0, 8, 8))
			for v := range a {
				g := NewMatrix[float64](64, 1)
				for range 8 {
					g.Data = append(g.Data, rng.NormFloat64())
				}
				pop[ii].Vector[v] = a[v].MulT(g).Add(u[v])
				for iii := range 8 {
					for iv := range 8 {
						if pop[ii].Vector[v].Data[iii*8+iv] > 0 {
							pixel := img.GrayAt(iii, iv)
							pixel.Y |= 1 << v
							img.SetGray(iii, iv, pixel)
						}
					}
				}
			}
			buffer := bytes.Buffer{}
			err := jpeg.Encode(&buffer, img, nil)
			if err != nil {
				panic(err)
			}
			pop[ii].Fitness = float64(buffer.Len())
			/*var histogram [2]float64
			for _, value := range pop[ii].Vector.Data {
				if value < 0 {
					histogram[0]++
				} else {
					histogram[1]++
				}
			}
			fitness := 0.0
			for _, value := range histogram {
				if value == 0 {
					continue
				}
				fitness += (value / 8.0) * math.Log2(value/8.0)
			}
			pop[ii].Fitness = math.Abs(fitness)*/
		}
		sort.Slice(pop, func(i, j int) bool {
			return pop[i].Fitness < pop[j].Fitness
		})
		for v := range state {
			for ii := range 8 {
				copy(state[v][ii], pop[ii].Vector[v].Data)
				if log {
					fmt.Println(state[v][ii])
				}
			}
		}
		{
			img := image.NewGray(image.Rect(0, 0, 8, 8))
			for v := range a {
				for iii := range 8 {
					for iv := range 8 {
						if pop[0].Vector[v].Data[iii*8+iv] > 0 {
							pixel := img.GrayAt(iii, iv)
							pixel.Y |= 1 << v
							img.SetGray(iii, iv, pixel)
						}
					}
				}
			}
			output, err := os.Create(fmt.Sprintf("img_%d.jpg", i))
			if err != nil {
				panic(err)
			}
			err = jpeg.Encode(output, img, nil)
			if err != nil {
				panic(err)
			}
			output.Close()
		}
		fmt.Println(pop[0].Fitness)
	}
}
