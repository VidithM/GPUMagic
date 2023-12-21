#include "GPUMagic.h"

template <typename T>
__global__ void mult_kernel(matrix<T> *res, matrix<T> *A, matrix<T> *B)
{

}

template <typename T>
void matmul_dense (
    matrix<T> **res,
    matrix<T> *A,
    matrix<T> *B
)
{
    assert(A->curr_type == storage_type::DENSE);
    assert(B->curr_type == storage_type::DENSE);
}


template void matmul_dense<float>(matrix<float> **res, matrix<float> *A, matrix<float> *B);
template void matmul_dense<double>(matrix<double> **res, matrix<double> *A, matrix<double> *B);
template void matmul_dense<int>(matrix<int> **res, matrix<int> *A, matrix<int> *B);
