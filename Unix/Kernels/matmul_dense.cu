#include "GPUMagic_internal.cuh"
#include "matrix.cuh"

template <typename T>
__global__ void mult_kernel(matrix<T> *res, matrix<T> *A, matrix<T> *B)
{
    size_t nrows = res->get_nrows();
    size_t ncols = res->get_ncols();

    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(tid_x >= nrows || tid_y >= ncols){
        return;
    }
    T sum = 0;
    bool found = false;
    for(int k = 0; k < A->get_ncols(); k++){
        if(A->exists(tid_x, k) && B->exists(k, tid_y)){
            sum += A->at(tid_x, k) * B->at(k, tid_y);
            found = true;
        }
    }
    if(found){
        res->put(tid_x, tid_y, sum);
    }
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
    ASSERT(A->is_init());
    ASSERT(B->is_init());

    ASSERT(A->get_storage_type() == DENSE);
    ASSERT(B->get_storage_type() == DENSE);

    ASSERT(A->get_ncols() == B->get_nrows());

    ASSERT(res != NULL);

    
    matrix<T> *d_A = to_gpu(A);
    matrix<T> *d_B = to_gpu(B);

    size_t res_nrows = A->get_nrows();
    size_t res_ncols = B->get_ncols();

    matrix<T> *h_res = new matrix<T>(DENSE);
    h_res->init(NULL, NULL, NULL, 0, res_nrows, res_ncols);
    matrix<T> *d_res = to_gpu(h_res);
    delete h_res; h_res = NULL;

    dim3 block_dim(block_dim_rows, block_dim_cols);
    dim3 grid_dim(
        ((res_nrows + block_dim_rows - 1) / block_dim_rows),
        ((res_ncols + block_dim_cols - 1) / block_dim_cols)
    );
    mult_kernel<<<grid_dim, block_dim>>>(d_res, d_A, d_B);
    CU_TRY(cudaPeekAtLastError());
    CU_TRY(cudaDeviceSynchronize());

    h_res = to_cpu(d_res);
    (*res) = h_res;
    gpu_del(d_res);
    gpu_del(d_A);
    gpu_del(d_B);
}

#define MAKE_PROTO(type) template void matmul_dense<type>(matrix<type> **res, matrix<type> *A, matrix<type> *B, size_t block_dim_rows, size_t block_dim_cols)

MAKE_PROTO(float);
MAKE_PROTO(double);
MAKE_PROTO(int);