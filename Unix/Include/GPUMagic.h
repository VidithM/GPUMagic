#pragma once
// ============= Utils/public structs ============
#include "matrix.cuh"
#include "timer.cuh"
#include <random>

template<typename T, bool is_float>
struct DistrFactory;

template<typename T>
struct DistrFactory<T, true> {
    using DistrType = std::uniform_real_distribution<T>;
};

template<typename T>
struct DistrFactory<T, false> {
    using DistrType = std::uniform_int_distribution<T>;
};

template<typename T>
constexpr bool is_float = std::is_floating_point<T>::value;

template<typename T>
using DistrType = typename DistrFactory<T, is_float<T>>::DistrType;

// ====================================================

// ===============  Kernel declarations ================
template <typename T>
void matmul_dense (
	matrix<T> **res,
	matrix<T> *A,
	matrix<T> *B,
	size_t block_dim_rows,
	size_t block_dim_cols
);

template <typename T>
void cumsum (
    matrix<T> **res,
    matrix<T> *arr,
    size_t chunk_size,
	size_t block_size
);

template <typename T>
void cumsum_span_efficient (
    matrix<T> **res,
    matrix<T> *arr,
    size_t block_size
);
// =====================================================
