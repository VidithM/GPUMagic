#include "GPUMagic.h"

template <typename T>
__global__ void cumsum_kernel(matrix<T> *res, matrix<T> *partial, matrix<T> *arr)
{
    size_t n = arr->get_ncols();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = partial->get_ncols();

}

template <typename T>
void cumsum(
    matrix<T> **res,
    matrix<T> *arr,
    size_t chunk_size
)
{
    assert(arr->get_storage_type == DENSE);
    assert(res != NULL);

    int nthreads = ((arr->ncols() + chunk_size - 1) / chunk_size);
}