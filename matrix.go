// Copyright 2024 The Entity Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"

	"github.com/pointlander/entity/vector"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

// Number is a number
type Float interface {
	float32 | float64
}

// Matrix is a float64 matrix
type Matrix[T Float] struct {
	Cols int
	Rows int
	Data []T
}

// NewMatrix creates a new float64 matrix
func NewMatrix[T Float](cols, rows int, data ...T) Matrix[T] {
	if data == nil {
		data = make([]T, 0, cols*rows)
	}
	return Matrix[T]{
		Cols: cols,
		Rows: rows,
		Data: data,
	}
}

// MulT multiplies two matrices and computes the transpose
func (m Matrix[T]) MulT(n Matrix[T]) Matrix[T] {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix[T]{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]T, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, dot(mm, nn))
		}
	}
	return o
}

// Add adds two float64 matrices
func (m Matrix[T]) Add(n Matrix[T]) Matrix[T] {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix[T]{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Sub subtracts two float64 matrices
func (m Matrix[T]) Sub(n Matrix[T]) Matrix[T] {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix[T]{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value-n.Data[i%lenb])
	}
	return o
}

// Hadamard multiples two float64 matrices
func (m Matrix[T]) Hadamard(n Matrix[T]) Matrix[T] {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix[T]{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value*n.Data[i%lenb])
	}
	return o
}

// Softmax calculates the softmax of the matrix rows
func (m Matrix[T]) Softmax(t T) Matrix[T] {
	output := NewMatrix[T](m.Cols, m.Rows)
	max := T(0.0)
	for _, v := range m.Data {
		v /= t
		if v > max {
			max = v
		}
	}
	s := max * S
	for i := 0; i < len(m.Data); i += m.Cols {
		sum := T(0.0)
		values := make([]T, m.Cols)
		for j, value := range m.Data[i : i+m.Cols] {
			values[j] = T(math.Exp(float64(value/t - s)))
			sum += values[j]
		}
		for _, value := range values {
			output.Data = append(output.Data, value/sum)
		}
	}
	return output
}

// Sigmoid computes the sigmoid of a matrix
func (m Matrix[T]) Sigmoid() Matrix[T] {
	o := Matrix[T]{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, 1/(1+T(math.Exp(float64(-value)))))
	}
	return o
}

// Entropy calculates the entropy of the matrix rows
func (m Matrix[T]) Entropy() Matrix[T] {
	output := NewMatrix[T](m.Rows, 1)
	for i := 0; i < len(m.Data); i += m.Cols {
		entropy := T(0.0)
		for _, value := range m.Data[i : i+m.Cols] {
			entropy += value * T(math.Log(float64(value)))
		}
		output.Data = append(output.Data, -entropy)
	}
	return output
}

// Sum sums the rows of a matrix
func (m Matrix[T]) Sum() Matrix[T] {
	o := Matrix[T]{
		Cols: m.Cols,
		Rows: 1,
		Data: make([]T, m.Cols),
	}
	for i := 0; i < m.Rows; i++ {
		offset := i * m.Cols
		for j := range o.Data {
			o.Data[j] += m.Data[offset+j]
		}
	}
	return o
}

// T tramsposes a matrix
func (m Matrix[T]) T() Matrix[T] {
	o := Matrix[T]{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

func dot[T Float](x, y []T) (z T) {
	switch x := any(x).(type) {
	case []float64:
		switch y := any(y).(type) {
		case []float64:
			for i := range x {
				z += T(x[i] * y[i])
			}
		}
	case []float32:
		switch y := any(y).(type) {
		case []float32:
			z = T(vector.Dot(x, y))
		}
	}
	return z
}

func softmax[T Float](values []T) {
	max := T(0.0)
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := T(0.0)
	for j, value := range values {
		values[j] = T(math.Exp(float64(value - s)))
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfAttention computes the self attention of Q, K, V
func SelfAttention[T Float](Q, K, V Matrix[T]) Matrix[T] {
	o := Matrix[T]{
		Cols: V.Cols,
		Rows: K.Rows,
		Data: make([]T, 0, V.Rows*K.Rows),
	}
	outputs, values := make([]T, V.Cols), make([]T, Q.Rows)
	V = V.T()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			outputs[j] = dot(values, V)
		}
		o.Data = append(o.Data, outputs...)
	}
	return o
}
