#include "GPUMagic.h"

template <typename T>
__global__ void mult_kernel(matrix<T> *res, matrix<T> *A, matrix<T> *B)
{

}

template <typename T>
void matmul_dense (
    matrix<T> **res,
    matrix<T> *A,
    matrix<T> *B,
    size_t block_dim_rows,
    size_t block_dim_cols
)
{
    assert(A->get_storage_type() == DENSE);
    assert(B->get_storage_type() == DENSE);

    assert(A->get_ncols() == B->get_nrows());

    assert(res == NULL);

    
    matrix<T> *d_tmp = NULL;
    dim3 block_dim(block_dim_rows, block_dim_cols);
    // dim3 grid_dim()

    // mult_kernel<<<grid_dim, block_dim>>>

    matrix<T> *res_val = new matrix<T>();
    res_val->set_storage_location(CPU);


}


template void matmul_dense<float>(matrix<float> **res, matrix<float> *A, matrix<float> *B, size_t block_dim_rows, size_t block_dim_cols);
template void matmul_dense<double>(matrix<double> **res, matrix<double> *A, matrix<double> *B, size_t block_dim_rows, size_t block_dim_cols);
template void matmul_dense<int>(matrix<int> **res, matrix<int> *A, matrix<int> *B, size_t block_dim_rows, size_t block_dim_cols);
