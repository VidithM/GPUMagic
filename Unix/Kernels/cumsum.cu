#include "GPUMagic.h"

template <typename T>
__global__ void cumsum_kernel_phase1(matrix<T> *res, matrix<T> *partial, matrix<T> *arr, size_t *chunk_size)
{
    size_t n = arr->get_ncols();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = partial->get_ncols(); 
    size_t len = *chunk_size;

    T sum = 0;
    size_t start = len * tid, end = len * tid + len - 1;
    for(size_t i = start; i <= end; i++){
        sum += arr->at(0, i);
    }
    partial->put(0, tid, sum);
}

template <typename T>
__global__ void cumsum_kernel_phase2(matrix<T> *res, matrix<T> *partial, matrix<T> *arr, size_t *chunk_size)
{
    size_t n = arr->get_ncols();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = partial->get_ncols();
    size_t len = *chunk_size;

    T prev_sum = 0;
    for(int i = 0; i < tid; i++){
        prev_sum += partial->at(0, i);
    }
    size_t start = len * tid, end = len * tid + len - 1;
    T curr_sum = prev_sum;
    for(int i = start; i <= end; i++){
        curr_sum += arr->at(0, i);
        res->put(0, i, curr_sum);
    }
}

template <typename T>
void cumsum (
    matrix<T> **res,
    matrix<T> *arr,
    size_t chunk_size
)
{
    assert(arr->get_storage_type() == DENSE);
    assert(res != NULL);

    int nthreads = ((arr->get_ncols() + chunk_size - 1) / chunk_size);

    size_t *d_chunk_size;
    cudaMalloc(&d_chunk_size, sizeof(size_t));
    cudaMemcpy(d_chunk_size, &chunk_size, sizeof(size_t), cudaMemcpyHostToDevice);

    matrix<T> *h_partial = new matrix<T>(DENSE, CPU);
    h_partial->init(NULL, NULL, NULL, 0, 1, nthreads);

    matrix<T> *h_res = new matrix<T>(DENSE, CPU);
    h_res->init(NULL, NULL, NULL, 0, 1, arr->get_ncols());

    matrix<T> *d_partial = to_gpu(h_partial), *d_res = to_gpu(h_res);
    delete h_partial; delete h_res; 

    matrix<T> *d_arr = to_gpu(arr);

    cumsum_kernel_phase1<<<1, nthreads>>>(d_res, d_partial, d_arr, d_chunk_size);
    CU_TRY(cudaPeekAtLastError());
    CU_TRY(cudaDeviceSynchronize());

    cumsum_kernel_phase2<<<1, nthreads>>>(d_res, d_partial, d_arr, d_chunk_size);
    CU_TRY(cudaPeekAtLastError());
    CU_TRY(cudaDeviceSynchronize());

    h_res = to_cpu(d_res);
    (*res) = h_res;

    gpu_del(d_res);
    gpu_del(d_arr);
    gpu_del(d_partial);
    cudaFree(d_chunk_size);
}

template void cumsum<float>(matrix<float> **res, matrix<float> *arr, size_t chunk);
template void cumsum<double>(matrix<double> **res, matrix<double> *arr, size_t chunk);
template void cumsum<int>(matrix<int> **res, matrix<int> *arr, size_t chunk);