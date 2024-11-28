#include "GPUMagic_internal.cuh"
#include "matrix.cuh"
#include "timer.cuh"

#define MAX_BLOCK_SIZE 512
#define MAX_NCHUNKS (1 << 10)

template <typename T>
__global__ void cumsum_kernel_phase1_old(matrix<T> *res, matrix<T> *partial, matrix<T> *arr, size_t chunk_size)
{
    size_t n = arr->get_ncols();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    int nchunks = partial->get_ncols();
    for(int chunk_id = tid; chunk_id < nchunks; chunk_id += nthreads){
        T sum = 0;
        int start = chunk_id * chunk_size;
        int end = start + chunk_size - 1;
        if(end >= n){
            end = n - 1;
        }
        for(int i = start; i <= end; i++){
            sum += arr->at(0, i);
        }
        partial->put(0, chunk_id, sum);
    }
}

template <typename T>
__global__ void cumsum_kernel_phase1_new(matrix<T> *res, matrix<T> *partial, matrix<T> *arr, size_t chunk_size)
{
    size_t n = arr->get_ncols();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nchunks = partial->get_ncols();
    __shared__ T sums[MAX_BLOCK_SIZE];
    for(int chunk_id = blockIdx.x; chunk_id < nchunks; chunk_id += gridDim.x){
        T my_sum = 0;
        for(int offset = threadIdx.x; offset < chunk_size; offset += blockDim.x){
            my_sum += arr->at(0, chunk_id * chunk_size + offset);
        }
        sums[threadIdx.x] = my_sum;
        __syncthreads();
        if(threadIdx.x == 0){
            T tot_sum = 0;
            for(int i = 0; i < blockDim.x; i++){
                tot_sum += sums[i];
                sums[i] = 0;
            }
            partial->put(0, chunk_id, tot_sum);
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void cumsum_kernel_phase2(matrix<T> *res, matrix<T> *partial, matrix<T> *arr, size_t chunk_size)
{
    // Next iteration: Make max_nchunks bigger and build sparse table
    size_t n = arr->get_ncols();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    int nchunks = partial->get_ncols();
    __shared__ T partial_cumsum[MAX_NCHUNKS];

    if(threadIdx.x == 0){
        T curr_sum = 0;
        for(int i = 0; i < partial->get_ncols(); i++){
            curr_sum += partial->at(0, i);
            partial_cumsum[i] = curr_sum;
        }
    }
    __syncthreads();

    for(int chunk_id = tid; chunk_id < nchunks; chunk_id += nthreads){
        T prev_sum = 0;
        if(chunk_id > 0){
            prev_sum = partial_cumsum[chunk_id - 1];
        }
        int start = chunk_id * chunk_size;
        int end = start + chunk_size - 1;
        if(end >= n){
            end = n - 1;         
        }
        T chunk_sum = 0;
        for(int i = start; i <= end; i++){
            chunk_sum += arr->at(0, i);
            res->put(0, i, prev_sum + chunk_sum);
        }
    }
}

template <typename T>
void cumsum (
    matrix<T> **res,
    matrix<T> *arr,
    size_t chunk_size,
    size_t block_size
)
{
    CU_INIT(0);
    ASSERT(block_size <= props0.maxThreadsPerBlock && block_size <= MAX_BLOCK_SIZE);
    ASSERT(arr->get_storage_type() == DENSE);
    ASSERT(arr->is_init());
    ASSERT(res != NULL);

    size_t nchunks = ((arr->get_ncols() + chunk_size - 1) / chunk_size);
    ASSERT(nchunks <= MAX_NCHUNKS);

    block_size = std::min(nchunks + (32 - (nchunks % 32)), block_size);
    int number_of_sms = props0.multiProcessorCount;
    int grid_size = ((nchunks + block_size - 1) / block_size);
    grid_size = min(grid_size, 10 * number_of_sms);

    matrix<T> *h_partial = new matrix<T>(DENSE);
    h_partial->init(NULL, NULL, NULL, 0, 1, nchunks);

    matrix<T> *h_res = new matrix<T>(DENSE);
    h_res->init(NULL, NULL, NULL, 0, 1, arr->get_ncols());

    matrix<T> *d_partial = to_gpu(h_partial), *d_res = to_gpu(h_res);
    delete h_partial; delete h_res; 

    matrix<T> *d_arr = to_gpu(arr);
    gpu_timer t;
    t.start("phase1");
    cumsum_kernel_phase1_old<<<grid_size, block_size>>>(d_res, d_partial, d_arr, chunk_size);
    t.end("phase1");
    CU_TRY(cudaPeekAtLastError());
    CU_TRY(cudaDeviceSynchronize());

    t.start("phase2");
    cumsum_kernel_phase2<<<grid_size, block_size>>>(d_res, d_partial, d_arr, chunk_size);
    t.end("phase2");
    CU_TRY(cudaPeekAtLastError());
    CU_TRY(cudaDeviceSynchronize());

    h_res = to_cpu(d_res);
    (*res) = h_res;

    gpu_del(d_res);
    gpu_del(d_arr);
    gpu_del(d_partial);
}

#define MAKE_PROTO(type) template void cumsum<type>(matrix<type> **res, matrix<type> *arr, size_t chunk, size_t block_size)

MAKE_PROTO(float);
MAKE_PROTO(double);
MAKE_PROTO(int);