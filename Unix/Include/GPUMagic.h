// Main include file for all sources
#include "common.cuh"

// ============= Utils/types/typedefs, etc. ============
#include "matrix.cuh"
#include "timer.cuh"
// =====================================================


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
void matmul_rowscale (
    matrix<T> **res,
    matrix<T> *A,
    matrix<T> *B,
    size_t block_size
);

template <typename T>
void cumsum (
    matrix<T> **res,
    matrix<T> *arr,
    size_t chunk_size,
	size_t block_size
);
// =====================================================
