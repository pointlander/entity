// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
)

// Text is the text model
func Text() {
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
	length, datum := len(forward), []rune(string(data))

	A, AI, u := make([]Matrix[float64], length), make([]Matrix[float64], length), make([]Matrix[float64], length)
	if *FlagBuild {
		out, err := os.Create("model.bin")
		if err != nil {
			panic(err)
		}
		defer out.Close()

		rng := rand.New(rand.NewSource(1))
		for i := range length {
			vectors, index := make([][]float64, 0, 8), 8
			for _, v := range datum[8:] {
				if int(forward[v]) == i {
					vector := make([]float64, length)
					for i := 1; i < 9; i++ {
						vector[forward[datum[index-i]]]++
					}
					vectors = append(vectors, vector)
				}
				index++
			}
			A[i], AI[i], u[i] = NewMultiVariateGaussian[float64](-1.0, 1.0e-1, true, true, rng, fmt.Sprintf("%d_text", i), length, vectors)

			buffer64 := make([]byte, 8)
			for _, parameter := range u[i].Data {
				bits := math.Float64bits(parameter)
				for i := range buffer64 {
					buffer64[i] = byte((bits >> (8 * i)) & 0xFF)
				}
				n, err := out.Write(buffer64)
				if err != nil {
					panic(err)
				}
				if n != len(buffer64) {
					panic("8 bytes should be been written")
				}
			}
			for _, parameter := range A[i].Data {
				bits := math.Float64bits(parameter)
				for i := range buffer64 {
					buffer64[i] = byte((bits >> (8 * i)) & 0xFF)
				}
				n, err := out.Write(buffer64)
				if err != nil {
					panic(err)
				}
				if n != len(buffer64) {
					panic("8 bytes should be been written")
				}
			}
			for _, parameter := range AI[i].Data {
				bits := math.Float64bits(parameter)
				for i := range buffer64 {
					buffer64[i] = byte((bits >> (8 * i)) & 0xFF)
				}
				n, err := out.Write(buffer64)
				if err != nil {
					panic(err)
				}
				if n != len(buffer64) {
					panic("8 bytes should be been written")
				}
			}
		}
		return
	}

	input, err := os.Open("model.bin")
	if err != nil {
		panic(err)
	}
	defer input.Close()

	for i := range length {
		u[i] = NewMatrix[float64](length, 1)
		buffer64 := make([]byte, 8)
		for range u[i].Rows {
			for range u[i].Cols {
				n, err := input.Read(buffer64)
				if err == io.EOF {
					panic(err)
				} else if err != nil {
					panic(err)
				}
				if n != len(buffer64) {
					panic(fmt.Errorf("not all bytes read: %d", n))
				}
				value := uint64(0)
				for k := 0; k < 8; k++ {
					value <<= 8
					value |= uint64(buffer64[7-k])
				}
				u[i].Data = append(u[i].Data, math.Float64frombits(value))
			}
		}
		A[i] = NewMatrix[float64](length, length)
		for range A[i].Rows {
			for range A[i].Cols {
				n, err := input.Read(buffer64)
				if err == io.EOF {
					panic(err)
				} else if err != nil {
					panic(err)
				}
				if n != len(buffer64) {
					panic(fmt.Errorf("not all bytes read: %d", n))
				}
				value := uint64(0)
				for k := 0; k < 8; k++ {
					value <<= 8
					value |= uint64(buffer64[7-k])
				}
				A[i].Data = append(A[i].Data, math.Float64frombits(value))
			}
		}
		AI[i] = NewMatrix[float64](length, length)
		for range AI[i].Rows {
			for range AI[i].Cols {
				n, err := input.Read(buffer64)
				if err == io.EOF {
					panic(err)
				} else if err != nil {
					panic(err)
				}
				if n != len(buffer64) {
					panic(fmt.Errorf("not all bytes read: %d", n))
				}
				value := uint64(0)
				for k := 0; k < 8; k++ {
					value <<= 8
					value |= uint64(buffer64[7-k])
				}
				AI[i].Data = append(AI[i].Data, math.Float64frombits(value))
			}
		}
	}
	buffer64 := make([]byte, 8)
	_, err = input.Read(buffer64)
	if err != io.EOF {
		panic("not at the end")
	}

	for i := range length {
		x := A[i].MulT(AI[i])
		count, total := 0, 0
		for ii := range x.Rows {
			for iii := range x.Cols {
				if ii == iii {
					if x.Data[ii*x.Cols+iii] < .9 {
						count++
					}
					total++
				}
			}
		}
		fmt.Println(i, count, total, float64(count)/float64(total))
	}

	rng := rand.New(rand.NewSource(1))
	grandPrompt, grandMax := []rune{}, 0
	for range 33 {
		prompt, grand := []rune("What is the meaning of life?"), 0
		const iterations = 128
		for range 8 {
			vector := NewMatrix(length, 1, make([]float64, length)...)
			for i := 1; i < 9; i++ {
				vector.Data[forward[prompt[len(prompt)-i]]]++
			}
			histogram := make([]int, length)
			for range iterations {
				min, index := math.MaxFloat64, 0
				for i := range length {
					if i == 0 {
						continue
					}
					reverse := AI[i].T().MulT(vector.Sub(u[i]))
					for iii := range reverse.Data {
						reverse.Data[iii] *= rng.NormFloat64()
					}
					forward := A[i].MulT(reverse).Add(u[i])
					fitness := L2(vector.Data, forward.Data)
					if fitness < min {
						min, index = fitness, i
					}
				}
				histogram[index]++
			}
			sum, index, sample := 0, 0, rng.Intn(iterations)
			for i, count := range histogram {
				sum += count
				if sample < sum {
					grand += count
					index = i
					break
				}
			}
			fmt.Printf("%c %d\n", reverse[byte(index)], reverse[byte(index)])
			prompt = append(prompt, reverse[byte(index)])
		}
		fmt.Println(grand, string(prompt))
		if grand > grandMax {
			grandMax, grandPrompt = grand, prompt
		}
	}
	fmt.Println(grandMax, string(grandPrompt))
}
