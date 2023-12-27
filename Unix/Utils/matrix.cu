#include "matrix.cuh"

template <typename T>
static __global__ void unpack(T **Ax, size_t **Ap, size_t **Ai, size_t **Aj, bool **Ab, matrix<T> *mat){
    mat->unpack(Ax, Ap, Ai, Aj, Ab);
}

template <typename T>
__host__ matrix<T>* matrix<T>::toGPU(matrix<T> *mat){
    if(mat->get_storage_location() == GPU){
        CU_ERROR("Attempt to call toGPU() with a GPU stored matrix\n", "");
    }
    T *d_Ax;
    size_t *d_Ap, *d_Ai, *d_Aj;
    bool *d_Ab;
    if(mat->get_storage_type() == DENSE){
        T *h_Ax;
        bool *h_Ab;
        mat->unpack(&h_Ax, NULL, NULL, NULL, &h_Ab);

        matrix<T> *d_mat = new matrix<T>();
        memcpy(d_mat, mat, sizeof(matrix<T>));
        d_mat->set_storage_location(GPU);
        // pack Ax, Ab, memcpy to GPU
    } else {
        // TODO: Implement this
        CU_ERROR("Unsupported method\n", "");
    }
}

template <typename T>
__host__ matrix<T>* matrix<T>::toCPU(matrix<T> *d_mat){
    if(d_mat->get_storage_location() == CPU){
        CU_ERROR("Attempt to call toCPUY() with a GPU stored matrix\n", "");
    }
    // TODO: Implement this
    return NULL;
}

template matrix<float>* matrix<float>::toGPU(matrix<float> *mat);
template matrix<float>* matrix<float>::toCPU(matrix<float> *d_mat);