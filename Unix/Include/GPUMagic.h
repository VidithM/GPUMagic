// Main include file for all sources
#include "common.h"

// ============= Utils/types/typedefs, etc. ============
#include "matrix.cuh"
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
// =====================================================
