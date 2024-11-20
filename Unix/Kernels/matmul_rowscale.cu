#include "GPUMagic.h"

template <typename T>
void matmul_rowscale (
    matrix<T> **res,
    matrix<T> *A,
    matrix<T> *B,
    size_t block_size
)
{
    ASSERT(A->is_init());
    ASSERT(B->is_init());

    ASSERT(A->is_diag());
    ASSERT((A->get_storage_type() == CSR) || (A->get_storage_type() == CSC));
    ASSERT(B->get_storage_type() == CSR);

    ASSERT(A->get_ncols() == B->get_nrows());

    ASSERT(res != NULL);

    matrix<T> *d_A = to_gpu(A);
    matrix<T> *d_B = to_gpu(B);

}

template void matmul_rowscale<float>(matrix<float> **res, matrix<float> *A, matrix<float> *B, size_t block_size);
template void matmul_rowscale<double>(matrix<double> **res, matrix<double> *A, matrix<double> *B, size_t block_size);
template void matmul_rowscale<int>(matrix<int> **res, matrix<int> *A, matrix<int> *B, size_t block_size);