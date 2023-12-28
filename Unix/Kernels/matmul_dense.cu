#include "GPUMagic.h"

template <typename T>
__global__ void mult_kernel(matrix<T> **res, matrix<T> *A, matrix<T> *B)
{
    printf("Passed in the kernel\n");
    printf("%d\n", A->get_nrows());

    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    printf("in kernel w/ (tid_x, tid_y): (%d, %d)\n", tid_x, tid_y);

    matrix<float> *test_mat = new matrix<float>();
    
    // __syncthreads();

    if(tid_x == 0 && tid_y == 0){
        // test_mat.set_storage_type(DENSE);
        // test_mat.set_storage_location(GPU);
    }
    // test_mat->init(NULL, NULL, NULL, 0, 1, 1);
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

    assert(A->is_init());
    assert(B->is_init());

    assert(A->get_ncols() == B->get_nrows());

    assert(res == NULL);

    
    matrix<T> *d_A = to_gpu(A);
    matrix<T> *d_B = to_gpu(B);
    matrix<T> *d_res;

    dim3 block_dim(block_dim_rows, block_dim_cols);
    dim3 grid_dim(1, 1);

    mult_kernel<<<grid_dim, block_dim>>>(&d_res, d_A, d_B);
    CU_TRY(cudaPeekAtLastError());
    CU_TRY(cudaDeviceSynchronize());

}


template void matmul_dense<float>(matrix<float> **res, matrix<float> *A, matrix<float> *B, size_t block_dim_rows, size_t block_dim_cols);
template void matmul_dense<double>(matrix<double> **res, matrix<double> *A, matrix<double> *B, size_t block_dim_rows, size_t block_dim_cols);
template void matmul_dense<int>(matrix<int> **res, matrix<int> *A, matrix<int> *B, size_t block_dim_rows, size_t block_dim_cols);
