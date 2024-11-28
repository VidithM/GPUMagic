#include "matrix.cuh"

template <typename T>
__host__ matrix<T>* to_gpu(matrix<T> *mat){
    if(mat->get_storage_location() == GPU){
        ERROR("Attempt to call to_gpu() with a GPU stored matrix\n", __FILE__, __LINE__);
    }
    if(!mat->is_init()){
        ERROR("Attempt to use uninitialized matrix\n", __FILE__, __LINE__);
    }
    T *d_Ax;
    // size_t *d_Ap, *d_Ai, *d_Aj;
    bool *d_Ab;
    if(mat->get_storage_type() == DENSE){
        size_t nrows = mat->nrows;
        size_t ncols = mat->ncols;
        size_t nvals = mat->nvals;
    
        cudaMalloc(&d_Ax, nrows * ncols * sizeof(T));
        cudaMalloc(&d_Ab, nrows * ncols * sizeof(bool));
        cudaMemcpy(d_Ax, mat->Ax, nrows * ncols * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ab, mat->Ab, nrows * ncols * sizeof(bool), cudaMemcpyHostToDevice);

        matrix<T> d_mat_proto(DENSE);
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

        CU_TRY(cudaPeekAtLastError());
        CU_TRY(cudaDeviceSynchronize());

        return d_mat;

    } else {
        // TODO: Implement this
        ERROR("Unsupported method\n", __FILE__, __LINE__);
    }
    return NULL;
}

template <typename T>
__host__ matrix<T>* to_cpu(matrix<T> *d_mat){
    // cudaMemcpy d_mat from device to host, then the host can examine its contents
    // then, cudaMemcpy the arrays to the host
    // build host matrix and return
    matrix<T> *h_mat = new matrix<T>(DENSE);
    cudaMemcpy(h_mat, d_mat, sizeof(matrix<T>), cudaMemcpyDeviceToHost);
    h_mat->location = CPU;
    T *h_Ax;
    // size_t *h_Ap, *h_Ai, *h_Aj;
    bool *h_Ab;
    if(h_mat->get_storage_type() == DENSE){
        size_t nrows = h_mat->nrows;
        size_t ncols = h_mat->ncols;
        
        h_Ax = new T[nrows * ncols * sizeof(T)];
        h_Ab = new bool[nrows * ncols * sizeof(bool)];
        cudaMemcpy(h_Ax, h_mat->Ax, nrows * ncols * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ab, h_mat->Ab, nrows * ncols * sizeof(bool), cudaMemcpyDeviceToHost);
        h_mat->Ax = h_Ax;
        h_mat->Ab = h_Ab;

        CU_TRY(cudaPeekAtLastError());
        CU_TRY(cudaDeviceSynchronize());

        return h_mat;
    } else {
        delete h_mat;
        ERROR("Unsupported method\n", __FILE__, __LINE__);
    }
    return NULL;
}

template <typename T>
__host__ void gpu_del(matrix<T> *d_mat){
    // similar to to_cpu, but deallocate instead
    matrix<T> h_mat(DENSE);
    cudaMemcpy(&h_mat, d_mat, sizeof(matrix<T>), cudaMemcpyDeviceToHost);

    cudaFree(h_mat.Ax);
    cudaFree(h_mat.Ab);
    cudaFree(h_mat.Ai);
    cudaFree(h_mat.Aj);
    cudaFree(h_mat.Ap);

    cudaFree(d_mat);

    CU_TRY(cudaPeekAtLastError());
    CU_TRY(cudaDeviceSynchronize());

    // prevent from freeing once h_mat gets destroyed
    h_mat.Ax = NULL;
    h_mat.Ab = NULL;
    h_mat.Ai = NULL;
    h_mat.Aj = NULL;
    h_mat.Ap = NULL;
}


template __host__ matrix<float>* to_gpu(matrix<float> *mat);
template __host__ matrix<int>* to_gpu(matrix<int> *mat);
template __host__ matrix<double>* to_gpu(matrix<double> *mat);

template __host__ matrix<float>* to_cpu(matrix<float> *d_mat);
template __host__ matrix<int>* to_cpu(matrix<int> *d_mat);
template __host__ matrix<double>* to_cpu(matrix<double> *d_mat);

template __host__ void gpu_del(matrix<float> *d_mat);
template __host__ void gpu_del(matrix<int> *d_mat);
template __host__ void gpu_del(matrix<double> *d_mat);