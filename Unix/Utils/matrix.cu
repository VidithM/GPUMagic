#include "matrix.cuh"

template <typename T>
__host__ matrix<T>* to_gpu(matrix<T> *mat){
    if(mat->get_storage_location() == GPU){
        CU_ERROR("Attempt to call to_gpu() with a GPU stored matrix\n", "");
    }
    if(!mat->is_init()){
        CU_ERROR("Attempt to use uninitialized matrix\n", "");
    }
    T *d_Ax;
    size_t *d_Ap, *d_Ai, *d_Aj;
    bool *d_Ab;
    if(mat->get_storage_type() == DENSE){
        size_t nrows = mat->nrows;
        size_t ncols = mat->ncols;
        size_t nvals = mat->nvals;
    
        cudaMalloc(&d_Ax, nrows * ncols * sizeof(T));
        cudaMalloc(&d_Ab, nrows * ncols * sizeof(bool));
        cudaMemcpy(d_Ax, mat->Ax, nrows * ncols * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ab, mat->Ab, nrows * ncols * sizeof(bool), cudaMemcpyHostToDevice);

        matrix<T> d_mat_proto;
        memcpy(&d_mat_proto, mat, sizeof(matrix<T>));
        d_mat_proto.Ax = d_Ax;
        d_mat_proto.Ab = d_Ab;
        d_mat_proto.location = GPU;

        matrix<T> *d_mat;
        cudaMalloc(&d_mat, sizeof(matrix<T>));
        cudaMemcpy(d_mat, &d_mat_proto, sizeof(matrix<T>), cudaMemcpyHostToDevice);

        // prevent arrays from being freed upon destruction
        d_mat_proto.Ax = NULL;
        d_mat_proto.Ab = NULL;

        return d_mat;

    } else {
        // TODO: Implement this
        CU_ERROR("Unsupported method\n", "");
    }
}

template <typename T>
__host__ matrix<T>* to_cpu(matrix<T> *d_mat){
    if(d_mat->get_storage_location() == CPU){
        CU_ERROR("Attempt to call to_cpuY() with a GPU stored matrix\n", "");
    }
    // TODO: Implement this
    // cudaMemcpy d_mat from device to host, then the host can examine its contents
    // then, cudaMemcpy the arrays to the host
    // build host matrix and return
    return NULL;
}

template __host__ matrix<float>* to_gpu(matrix<float> *mat);
template __host__ matrix<float>* to_cpu(matrix<float> *d_mat);

template __host__ matrix<int>* to_gpu(matrix<int> *mat);
template __host__ matrix<int>* to_cpu(matrix<int> *d_mat);

template __host__ matrix<double>* to_gpu(matrix<double> *mat);
template __host__ matrix<double>* to_cpu(matrix<double> *d_mat);