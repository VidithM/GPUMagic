#pragma once

#include "common.h"

#include <map>
#include <set>
#include <cstring>

enum storage_type       { DENSE, CSR, CSC, UNKNOWN_TYPE };
enum storage_location   { CPU, GPU, UNKNOWN_LOCATION };

template <typename T>
class matrix {
    private:
        storage_type curr_type;
        storage_location location;
        size_t *Ap;                 // start index of k-th row/column for sparse
        size_t *Ai, *Aj;            // rows/columns and values for sparse
        bool *Ab;                   // pattern of where entries are for dense
        T *Ax;                      // values of entries for both dense and sparse
        size_t nrows, ncols, nvals;
        bool did_init;

        void* allocate(int nbytes){
            if(location == UNKNOWN_LOCATION){
                CU_ERROR("Attempt to use uninitialized matrix\n", "");
            }
            if(location == CPU){
                return malloc(nbytes);
            } else {
                void *res = NULL;
                cudaMalloc(&res, nbytes);
                return res;
            }
        }

        void scrap(void **block){
            if(location == UNKNOWN_LOCATION){
                CU_ERROR("Attempt to use uninitialized matrix\n", "");
            }
            if(location == CPU){
                free(*block);
                *block = NULL;
            } else {
                cudaFree(*block);
                *block = NULL;
            }
        }

    public:
        __host__ __device__ matrix() : did_init(false), curr_type(UNKNOWN_TYPE), location(UNKNOWN_LOCATION), Ap(NULL), Ai(NULL), Aj(NULL), Ab(NULL), Ax(NULL) {}
        __host__ __device__ ~matrix() { scrap((void**) &Ap); scrap((void**) &Ai); scrap((void**) &Aj); scrap((void**) &Ab); scrap((void**) &Ax); }

        __host__ __device__ size_t get_nrows(){ return nrows; }
        __host__ __device__ size_t get_ncols(){ return ncols; }
        __host__ __device__ size_t get_nvals(){ return nvals; }

        __host__ __device__ bool exists(size_t row, size_t col){
            if(!did_init){
                CU_ERROR("Attempt to use uninitialized matrix\n", "");
            }
            if(curr_type == DENSE){
                return Ab[row * ncols + col];
            } else {
                // TODO: Implement this
                CU_ERROR("Unsupported method\n", "");
            }
            return false;
        }

        __host__ __device__ T at(size_t row, size_t col){
            if(!did_init){
                CU_ERROR("Attempt to use uninitialized matrix\n", "");
            }
            if(curr_type == DENSE){
                return Ax[row * ncols + col];
            } else {
                // TODO: Implement this
                CU_ERROR("Unsupported method\n", "");
            }
            return Ax[0];
        }

        __host__ __device__ void put(size_t row, size_t col, T val){
            if(!did_init){
                CU_ERROR("Attempt to use uninitialized matrix\n", "");
            }
            if(curr_type == DENSE){
                Ax[row * ncols + col] = val;
                Ab[row * ncols + col] = true;
            } else {
                // TODO: Implement this
            }
        }

        __host__ __device__ void del(size_t row, size_t col){
            if(!did_init){
                CU_ERROR("Attempt to use uninitialized matrix\n", "");
            }
            if(curr_type == DENSE){
                Ab[row * ncols + col] = false;
            } else {
                // TODO: Implement this
            }
        }

        __host__ __device__ void set_storage_type(storage_type type){
            this->curr_type = type;
            // TODO: Change existing contents of array
        }

        storage_type get_storage_type(){
            return curr_type;
        }

        void set_storage_location(storage_location location){
            // Note: Does NOT actually move the contents of the matrix. This simply
            // makes it known that the matrix is supposed to be stored in a certain location,
            // which is checked for assertions and memory allocation/de-allocation.

            // It is the responsibility of the user to make sure that the actual location of
            // the matrix is consistent with this setting. Static factory functions to convert
            // to a new storage location are available with matrix<T>::toGPU() and matrix<T>::toCPU().
            this->location = location;
        }

        storage_location get_storage_location(){
            return location;
        }

        static __host__ matrix<T>* toGPU(matrix<T> *mat);
        static __host__ matrix<T>* toCPU(matrix<T> *d_mat);

        __host__ __device__ bool is_diag(){
            if(!did_init){
                CU_ERROR("Attempt to use uninitialized matrix\n", "");
            }
            if(curr_type == DENSE){
                for(size_t i = 0; i < nrows * ncols; i++){
                    if(!Ab[i]){ continue; }
                    if(i % (ncols + 1)){ return false; }
                }
                return true;
            } else {
                CU_ERROR("Unsupported method\n", "");
                // TODO: Implement for CSR/CSC
            }
        }

         __host__ __device__ void init(
            T *entries, 
            size_t *rows, 
            size_t *cols, 
            size_t nvals, 
            size_t nrows, 
            size_t ncols,
            storage_location location,
            storage_type type
        ){
            if(location == UNKNOWN_LOCATION || type == UNKNOWN_TYPE){
                CU_ERROR("Attempt to initialize matrix with unknown storage location/type\n", "");
            }

            this->curr_type = type;
            this->location = location;
            this->nrows = nrows;
            this->ncols = ncols;
            this->nvals = nvals;

            CU_PRINT("%d\n", get_storage_location());
            CU_PRINT("%d\n", get_storage_type());

            if(curr_type == DENSE){
                if(Ab != NULL){
                    scrap((void**) &Ab); scrap((void**) &Ax);
                }
                Ab = (bool*) allocate(nrows * ncols);
                Ax = (T*) allocate(nrows * ncols);

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
                CU_ERROR("Unsupported method\n", "");
                #if 0
                    // TODO: Finish implementing CSR/CSC init
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
};
