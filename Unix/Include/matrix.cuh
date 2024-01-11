#pragma once

#include "common.cuh"

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
                ERROR("Attempt to use uninitialized matrix\n", __FILE__, __LINE__);
            }
            return malloc(nbytes);
        }

        __host__ void scrap(void **block){
            if(location == UNKNOWN_LOCATION){
                ERROR("Attempt to use uninitialized matrix\n", __FILE__, __LINE__);
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
                ERROR("Attempt to use uninitialized matrix\n", __FILE__, __LINE__);
            }
            if(row >= nrows || col >= ncols){
                ERROR("Out of bounds matrix access\n", __FILE__, __LINE__);
            }
            if(curr_type == DENSE){
                return Ab[row * ncols + col];
            } else {
                // TODO: Implement this
                ERROR("Unsupported method\n", __FILE__, __LINE__);
            }
            return false;
        }

        __host__ __device__ T at(size_t row, size_t col){
            if(!did_init){
                ERROR("Attempt to use uninitialized matrix\n", __FILE__, __LINE__);
            }
            if(row >= nrows || col >= ncols){
                ERROR("Out of bounds matrix access\n", __FILE__, __LINE__);
            }
            if(curr_type == DENSE){
                return Ax[row * ncols + col];
            } else {
                // TODO: Implement this
                ERROR("Unsupported method\n", __FILE__, __LINE__);
            }
            return Ax[0];
        }

        __host__ __device__ void put(size_t row, size_t col, T val){
            if(!did_init){
                ERROR("Attempt to use uninitialized matrix\n", __FILE__, __LINE__);
            }
            if(row >= nrows || col >= ncols){
                ERROR("Out of bounds matrix access\n", __FILE__, __LINE__);
            }
            if(curr_type == DENSE){
                Ax[row * ncols + col] = val;
                Ab[row * ncols + col] = true;
            } else {
                // TODO: Implement this
                ERROR("Unsupported method\n", __FILE__, __LINE__);
            }
        }

        __host__ __device__ void del(size_t row, size_t col){
            if(!did_init){
                ERROR("Attempt to use uninitialized matrix\n", __FILE__, __LINE__);
            }
            if(row >= nrows || col >= ncols){
                ERROR("Out of bounds matrix access\n", __FILE__, __LINE__);
            }
            if(curr_type == DENSE){
                Ab[row * ncols + col] = false;
            } else {
                // TODO: Implement this
                ERROR("Unsupported method\n", __FILE__, __LINE__);
            }
        }

        __host__ __device__ void set_storage_type(storage_type type){
            if(type == UNKNOWN_TYPE){
                ERROR("Cannot set storage type to unkown\n", __FILE__, __LINE__);
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
                ERROR("Attempt to use uninitialized matrix\n", __FILE__, __LINE__);
            }
            if(curr_type == DENSE){
                for(size_t i = 0; i < nrows * ncols; i++){
                    if(!Ab[i]){ continue; }
                    if(i % (ncols + 1)){ return false; }
                }
                return true;
            } else {
                ERROR("Unsupported method\n", __FILE__, __LINE__);
                // TODO: Implement for CSR/CSC
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
                ERROR("Attempt to initialize matrix with unknown storage location/type.\n", __FILE__, __LINE__);
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

                if(location == CPU){
                    memset(Ab, false, (nrows * ncols) * sizeof(bool));
                } else {
                    cudaMemset(Ab, false, (nrows * ncols) * sizeof(bool));
                }

                for(size_t i = 0; i < nvals; i++){
                    Ab[rows[i] * ncols + cols[i]] = true;
                    Ax[rows[i] * ncols + cols[i]] = entries[i];
                }
                did_init = true;
            } else {
                ERROR("Unsupported method\n", __FILE__, __LINE__);
                #if 0
                    // TODO: Finish implementing CSR/CSC init
                    // Cannot use map in device code
                    if(Ap != NULL){
                        scrap((void**) &Ap); scrap((void**) &Aj); scrap((void**) &Ai); scrap((void**) &Ax);
                    }
                    std::map<size_t, std::set<std::pair<size_t, T>>> ord;

                    for(size_t i = 0; i < nvals; i++){
                        ord[rows[i]].insert({cols[i], entries[i]});
                    }
                    
                    Ap = new size_t[nrows + 1];
                    Aj = new size_t[nvals];
                    Ax = new size_t[nvals];
                    Ap[0] = 0;
                    int Ap_at = 1;
                    int Ax_at;     
                    for(auto &row : ord){
                        int row_idx = row.first;
                        int nvals_this_row = row.second.size();
                        Ap[at] = Ap[at - 1] + nvals_this_row;
                        for(auto &entry : row.second){

                        }
                    }
                #endif
            }
        }

        __host__ void print(){
            if(!did_init){
                ERROR("Attempt to use uninitialized matrix\n", __FILE__, __LINE__);
            }
            if(get_storage_type() == DENSE){
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
                ERROR("Unsupported method\n", __FILE__, __LINE__);
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