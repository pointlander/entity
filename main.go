// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"flag"
	"io"
	"strconv"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
	// Scale is the scale of the model
	Scale = 128
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

var Galaxies = [][]float64{
	{2.48, 00, 42, 41.877, 40, 51, 54.71}, // M32
	{2.69, 00, 40, 22.054, 41, 41, 08.04}, // M110
	{2.01, 00, 38, 57.523, 48, 20, 14.86}, // NGC 185
	{2.20, 00, 33, 12.131, 48, 30, 32.82}, // NGC 147
	{2.43, 00, 45, 39.264, 38, 02, 35.17}, // Andromeda I
	{2.13, 01, 16, 28.136, 33, 25, 50.36}, // Andromeda II
	{2.44, 00, 35, 31.777, 36, 30, 04.19}, // Andromeda III
	{2.52, 01, 10, 16.952, 47, 37, 40.12}, // Andromeda V
	{2.55, 23, 51, 46.516, 24, 34, 55.69}, // Andromeda VI
	{2.49, 23, 26, 33.321, 50, 40, 49.98}, // Andromeda VII
	{2.70, 00, 42, 06.000, 40, 37, 00.00}, // Andromeda VIII
	{2.50, 00, 52, 52.493, 43, 11, 55.66}, // Andromeda IX
	{2.90, 01, 06, 34.740, 44, 48, 23.31}, // Andromeda X
}

//go:embed iris.zip
var Iris embed.FS

var log bool

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Cluster  int
	Index    int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Inverse is the labels inverse map
var Inverse = [3]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

// L2 is the L2 norm
func L2(a, b []float64) float64 {
	c := 0.0
	for i, v := range a {
		diff := v - b[i]
		c += diff * diff
	}
	return c
}

var (
	// FlagIris the iris model
	FlagIris = flag.Bool("iris", false, "the iris model")
	// FlagText text model
	FlagText = flag.Bool("text", false, "the text model")
	// FlagImage the image model
	FlagImage = flag.Bool("image", false, "the image model")
	// FlagRNN the rnn model
	FlagRNN = flag.Bool("rnn", false, "the rnn model")
	// FlagFactor factor a number
	FlagFactor = flag.Bool("factor", false, "factor a number")
	// FlagQueens 8 queens problem
	FlagQueens = flag.Bool("queens", false, "8 queens problem")
	// FlagBF bf mode
	FlagBF = flag.Bool("bf", false, "bf mode")
	// FlagFF feed forward mode
	FlagFF = flag.Bool("ff", false, "feed forward mode")
	// FlagTransformer transformer mode
	FlagTransformer = flag.Bool("t", false, "transformer mode")
	// FlagBuild build the model
	FlagBuild = flag.Bool("build", false, "build the model")
)

//go:embed books/*
var Data embed.FS

func main() {
	flag.Parse()

	// +
	if *FlagIris {
		IrisModel()
		return
	}

	// -
	if *FlagText {
		Text()
		return
	}

	// +-
	if *FlagImage {
		Image()
		return
	}

	// -
	if *FlagRNN {
		RNN()
		return
	}

	// +-
	if *FlagFactor {
		Factor()
		return
	}

	// +
	if *FlagQueens {
		Queens()
		return
	}

	// -
	if *FlagBF {
		BF()
		return
	}

	// +
	if *FlagFF {
		FF()
		return
	}

	//
	if *FlagTransformer {
		T()
		return
	}
}
