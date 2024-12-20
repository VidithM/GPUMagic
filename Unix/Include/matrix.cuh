#pragma once

#include "GPUMagic_internal.cuh"

#include <map>
#include <set>
#include <cstring>
#include <sstream>

enum storage_type       { DENSE, CSR, CSC, UNKNOWN_TYPE };
enum storage_location   { CPU, GPU, UNKNOWN_LOCATION };

template <typename T>
class matrix {
    private:
        storage_type curr_type;
        storage_location location;
        size_t *Ap;                 // start index of k-th row/column fo`r sparse
        size_t *Ai, *Aj;            // rows/columns and values for sparse
        bool *Ab;                   // pattern of where entries are for dense
        T *Ax;                      // values of entries for both dense and sparse
        size_t nrows, ncols, nvals;
        bool did_init;

        __host__ void* allocate(int nbytes){
            if(location == UNKNOWN_LOCATION){
                ERROR("Attempt to use uninitialized matrix\n");
            }
            return malloc(nbytes);
        }

        __host__ void scrap(void **block){
            if(location == UNKNOWN_LOCATION){
                ERROR("Attempt to use uninitialized matrix\n");
            }
            free(*block);
            *block = NULL;
        }

    public:
        matrix() = delete;
        __host__ matrix(storage_type type) : did_init(false), Ap(NULL), Ai(NULL), Aj(NULL), Ab(NULL), Ax(NULL) {
            this->curr_type = type;
            this->location = CPU;
        }
        __host__ ~matrix() {
            if(!did_init){
                // Nothing to do
                return;
            }
            scrap((void**) &Ap); scrap((void**) &Ai); scrap((void**) &Aj); scrap((void**) &Ab); scrap((void**) &Ax); 
        }

        __host__ __device__ size_t get_nrows(){ return nrows; }
        __host__ __device__ size_t get_ncols(){ return ncols; }
        __host__ __device__ size_t get_nvals(){ return nvals; }

        __host__ __device__ bool exists(size_t row, size_t col){
            if(!did_init){
                ERROR("Attempt to use uninitialized matrix\n");
            }
            if(row >= nrows || col >= ncols){
                ERROR("Out of bounds matrix access: tried (%ld, %ld) for dim (%ld, %ld)\n", row, col, nrows, ncols);
            }
            if(curr_type == DENSE){
                return Ab[row * ncols + col];
            } else {
                // TODO: Implement this
                ERROR("Unsupported method\n");
            }
            return false;
        }

        __host__ __device__ T at(size_t row, size_t col){
            if(!did_init){
                ERROR("Attempt to use uninitialized matrix\n");
            }
            if(row >= nrows || col >= ncols){
                ERROR("Out of bounds matrix access: tried (%ld, %ld) for dim (%ld, %ld)\n", row, col, nrows, ncols);
            }
            if(curr_type == DENSE){
                return Ax[row * ncols + col];
            } else {
                // TODO: Implement this
                ERROR("Unsupported method\n");
            }
            return Ax[0];
        }

        __host__ __device__ void put(size_t row, size_t col, T val){
            if(!did_init){
                ERROR("Attempt to use uninitialized matrix\n");
            }
            if(row >= nrows || col >= ncols){
                ERROR("Out of bounds matrix access: tried (%ld, %ld) for dim (%ld, %ld)\n", row, col, nrows, ncols);
            }
            if(curr_type == DENSE){
                Ax[row * ncols + col] = val;
                Ab[row * ncols + col] = true;
            } else {
                // TODO: Implement this
                ERROR("Unsupported method\n");
            }
        }

        __host__ __device__ void del(size_t row, size_t col){
            if(!did_init){
                ERROR("Attempt to use uninitialized matrix\n");
            }
            if(row >= nrows || col >= ncols){
                ERROR("Out of bounds matrix access: tried (%ld, %ld) for dim (%ld, %ld)\n", row, col, nrows, ncols);
            }
            if(curr_type == DENSE){
                Ab[row * ncols + col] = false;
            } else {
                // TODO: Implement this
                ERROR("Unsupported method\n");
            }
        }

        __host__ __device__ void set_storage_type(storage_type type){
            if(type == UNKNOWN_TYPE){
                ERROR("Cannot set storage type to unkown\n");
            }
            this->curr_type = type;
            if(did_init){
                // TODO: Change existing contents
            }
        }

        __host__ __device__ storage_type get_storage_type(){
            return curr_type;
        }

        __host__ __device__ storage_location get_storage_location(){
            return location;
        }

        __host__ __device__ bool is_diag(){
            if(!did_init){
                ERROR("Attempt to use uninitialized matrix\n");
            }
            if(curr_type == DENSE){
                for(size_t i = 0; i < nrows * ncols; i++){
                    if(!Ab[i]){ continue; }
                    if(i % (ncols + 1)){ return false; }
                }
                return true;
            } else {
                // ERROR("Unsupported method\n");
                // TODO: Implement for CSR/CSC
                if(curr_type == CSR){
                    for(size_t i = 0; i < nrows; i++){
                        size_t num_entries = Ap[i + 1] - Ap[i];
                        if(num_entries > 1){
                            return false;
                        }
                        if(num_entries == 0){
                            continue;
                        }
                        if(Aj[Ap[i]] != i){
                            return false;
                        }
                    }
                    return true;
                } else {
                    for(size_t i = 0; i < ncols; i++){
                        size_t num_entries = Ap[i + 1] - Ap[i];
                        if(num_entries > 1){
                            return false;
                        }
                        if(num_entries == 0){
                            continue;
                        }
                        if(Ai[Ap[i]] != i){
                            return false;
                        }
                    }
                    return true;
                }
            }
        }

        __host__ __device__ bool is_init(){
            return did_init;
        }
        
        __host__ void init(
            T *entries = NULL,
            size_t *rows = NULL,
            size_t *cols = NULL,
            size_t nvals = 0,
            size_t nrows = 0,
            size_t ncols = 0
        ){
            if(location == UNKNOWN_LOCATION || curr_type == UNKNOWN_TYPE){
                ERROR("Attempt to initialize matrix with unknown storage location/type.\n");
            }
            this->nrows = nrows;
            this->ncols = ncols;
            this->nvals = nvals;

            if(curr_type == DENSE){
                if(Ab != NULL){
                    scrap((void**) &Ab); scrap((void**) &Ax);
                }
                Ab = (bool*) allocate(nrows * ncols * sizeof(bool));
                Ax = (T*) allocate(nrows * ncols * sizeof(T));

                memset(Ab, false, (nrows * ncols) * sizeof(bool));

                for(size_t i = 0; i < nvals; i++){
                    Ab[rows[i] * ncols + cols[i]] = true;
                    Ax[rows[i] * ncols + cols[i]] = entries[i];
                }
                did_init = true;
            } else {
                size_t *indices = (curr_type == CSR) ? Aj : Ai;
                size_t vdim = (curr_type == CSR) ? nrows : ncols;
                
                if(Ap != NULL){
                    scrap((void**) &Ap);
                    scrap((void**) &indices);
                    scrap((void**) &Ax);
                }

                std::map<size_t, std::set<std::pair<size_t, T>>> ord;

                for(size_t i = 0; i < nvals; i++){
                    size_t bucket = (curr_type == CSR) ? rows[i] : cols[i];
                    size_t index = (curr_type == CSR) ? cols[i] : rows[i];
                    ord[bucket].insert({index, entries[i]});
                }
                
                Ap = (size_t*) allocate((vdim + 1) * sizeof(size_t));
                for(int i = 0; i < vdim + 1; i++){
                    Ap[i] = -1;
                }
                indices = (size_t*) allocate(nvals * sizeof(size_t));
                Ax = (T*) allocate(nvals * sizeof(T));
                int Ap_at = 0;
                int Ap_put = 0;
                int Ax_at = 0;
                for(auto &vec : ord){
                    int vec_idx = vec.first;
                    int nvals_this_vec = vec.second.size();
                    while(Ap_at <= vec_idx){
                        Ap[Ap_at] = Ap_put;
                        Ap_at++;
                    }
                    Ap_put += nvals_this_vec;
                    for(auto &entry : vec.second){
                        indices[Ax_at] = entry.first;
                        Ax[Ax_at] = entry.second;
                        Ax_at++;
                    }
                }
                while(Ap_at <= vdim){
                    Ap[Ap_at] = Ap_put;
                    Ap_at++;
                }
                if(curr_type == CSR){
                    Aj = indices;
                } else {
                    Ai = indices;
                }
                did_init = true;
            }
        }

        __host__ void print(){
            if(!did_init){
                ERROR("Attempt to use uninitialized matrix\n");
            }
            if(curr_type == DENSE){
                std::ostringstream oss;
                oss << "\nDENSE (" << nrows << "-by-" << ncols << ") MATRIX" << std::endl;
                oss << "PATTERN: (Ab)" << std::endl;
                for(int i = 0; i < nrows; i++){
                    for(int j = 0; j < ncols; j++){
                        oss << Ab[i * ncols + j] << " ";
                    }
                    oss << std::endl;
                }
                oss << "ENTRIES: (Ax)" << std::endl;
                for(size_t i = 0; i < nrows; i++){
                    for(int j = 0; j < ncols; j++){
                        if(!Ab[i * ncols + j]){
                            oss << ". ";
                        } else {
                            oss << Ax[i * ncols + j] << " ";
                        }
                    }
                    oss << std::endl;
                }
                DBG(oss.str().c_str(), "");
            } else {
                std::ostringstream oss;
                std::string mat_type = (curr_type == CSR) ? "CSR" : "CSC";
                std::string index_array_name = (curr_type == CSR) ? "Aj" : "Ai";
                size_t *indices = (curr_type == CSR) ? Aj : Ai;
                size_t vdim = (curr_type == CSR) ? nrows : ncols;
                oss << "\n" << mat_type << " (" << nrows << "-by-" << ncols << ") MATRIX" << std::endl;
                oss << "POINTER (Ap)" << std::endl;
                for(int i = 0; i < vdim; i++){
                    oss << Ap[i] << " ";
                }
                oss << std::endl;
                oss << "INDICES (" << index_array_name << ")" << std::endl;
                for(int i = 0; i < nvals; i++){
                    oss << indices[i] << " ";
                }
                oss << std::endl;
                oss << "ENTRIES (Ax)" << std::endl;
                for(int i = 0; i < nvals; i++){
                    oss << Ax[i] << " ";
                }
                oss << std::endl;
                DBG(oss.str().c_str(), "");
            }
        }

        __host__ __device__ const size_t*   get_Ap(){ return Ap; }
        __host__ __device__ const size_t*   get_Ai(){ return Ai; }
        __host__ __device__ const size_t*   get_Aj(){ return Aj; }
        __host__ __device__ const bool*     get_Ab(){ return Ab; }
        __host__ __device__ const T*        get_Ax(){ return Ax; }

        template <typename V>
        friend __host__ matrix<V>* to_gpu(matrix<V> *mat);

        template <typename V>
        friend __host__ matrix<V>* to_cpu(matrix<V> *d_mat);

        template <typename V>
        friend __host__ void gpu_del(matrix<V> *d_mat);
};