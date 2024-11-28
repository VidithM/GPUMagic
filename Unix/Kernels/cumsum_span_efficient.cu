#include "GPUMagic_internal.cuh"
#include "matrix.cuh"
#include "timer.cuh"

#define MAX_BLOCK_SIZE 512
#define MAX_NCHUNKS (1 << 10)

template <typename T>
__global__ void cumsum_kernel(matrix<T> *res, matrix<T> *arr, int stage)
{
    size_t n = arr->get_ncols();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    for(int i = tid + (1 << stage); i < n; i += nthreads){
        res->put(0, i, res->at(0, i) + arr->at(0, i - (1 << stage)));
    }
}

template <typename T>
void cumsum_span_efficient (
    matrix<T> **res,
    matrix<T> *arr,
    size_t block_size
)
{
    CU_INIT(0);
    ASSERT(arr->get_storage_type() == DENSE);
    ASSERT(arr->is_init());
    ASSERT(res != NULL);

    int number_of_sms = props0.multiProcessorCount;
    int n = arr->get_ncols();

    matrix<T> *h_res = new matrix<T>(DENSE);
    h_res->init(NULL, NULL, NULL, 0, 1, arr->get_ncols());
    // initialize result to arr
    matrix<T> *d_arr = to_gpu(arr), *d_res = to_gpu(arr);

    gpu_timer t;
    t.start("GPU kernels start");
    int stage = 0;
    for (int i = 1; i < n; i *= 2){
        int grid_size = ((n - i + block_size - 1) / block_size);
        grid_size = min(grid_size, 10 * number_of_sms);

        cumsum_kernel<<<grid_size, block_size>>>(d_res, d_arr, stage);
        
        CU_TRY(cudaPeekAtLastError());
        CU_TRY(cudaDeviceSynchronize());
        stage++;
    }
    t.end("GPU kernels end");

    h_res = to_cpu(d_res);
    (*res) = h_res;

    gpu_del(d_res);
    gpu_del(d_arr);
}

#define MAKE_PROTO(type) template void cumsum_span_efficient<type>(matrix<type> **res, matrix<type> *arr, size_t block_size)

MAKE_PROTO(float);
MAKE_PROTO(double);
MAKE_PROTO(int);