// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math/rand"
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
	width := set.Size() / 4760
	fmt.Println(width, set.Size())
	inputs := NewMatrix[float32](256, 100)
	for range inputs.Cols * inputs.Rows {
		inputs.Data = append(inputs.Data, 0)
	}
	outputs := NewMatrix[float32](256, 100)
	for range outputs.Cols * outputs.Rows {
		outputs.Data = append(outputs.Data, 0)
	}
	g := make([]float32, set.Size())
	for i := range g {
		g[i] = float32(rng.NormFloat64())
	}
	s := NewMatrices(set, g)
	Transformer(s, inputs, outputs)
}
