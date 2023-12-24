#include "matrix.cuh"

template <typename T>
static __global__ void unpack(T **Ax, size_t **Ap, size_t **Ai, size_t **Aj, bool **Ab, matrix<T> *mat){

}

template <typename T>
__host__ matrix<T>* matrix<T>::toGPU(matrix<T> *mat){
    if(mat->get_storage_location() != GPU){
        CU_ERROR("Attempt to call toGPU() with a non-GPU stored matrix\n", "");
    }
    T *d_Ax;
    size_t *d_Ap, *d_Ai, *d_Aj;
    bool *d_Ab;
    if(mat->get_storage_type() == DENSE){
        // cudaMalloc(&d_Ax, mat->)

    } else {
        // TODO: Implement this
        CU_ERROR("Unsupported method\n", "");
    }
}

template <typename T>
__host__ matrix<T>* matrix<T>::toCPU(matrix<T> *d_mat){
    // TODO: Implement this
    return NULL;
}

template matrix<float>* matrix<float>::toGPU(matrix<float> *mat);
template matrix<float>* matrix<float>::toCPU(matrix<float> *d_mat);