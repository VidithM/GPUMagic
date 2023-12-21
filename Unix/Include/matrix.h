#pragma once

#include <map>
#include <set>
#include <cstring>

enum storage_type { DENSE, CSR, CSC, UNKNOWN };

template <typename T>
struct matrix {
    storage_type curr_type;
    size_t *Ap;        // start index of k-th row/column for sparse
    size_t *Ai, *Aj;   // rows/columns and values for sparse
    bool *Ab;          // pattern of where entries are for dense
    T *Ax;             // values of entries for both dense and sparse

    size_t nrows, ncols, nvals;

    matrix() : curr_type(storage_type::UNKNOWN), Ap(NULL), Ai(NULL), Aj(NULL), Ab(NULL), Ax(NULL) {}
    ~matrix() { delete Ap; delete Ai; delete Aj; delete Ab; delete Ax; }

    void set_storage_type(storage_type type){
        this->curr_type = type;
    }

    void init(
        T *entries, 
        size_t *rows, 
        size_t *cols, 
        size_t nvals, 
        size_t nrows, 
        size_t ncols
    ){
        this->nrows = nrows;
        this->ncols = ncols;
        this->nvals = nvals;

        if(curr_type == storage_type::DENSE){
            delete Ab; delete Ax;
            Ab = new bool[nrows * ncols];
            Ax = new bool[nrows * ncols];

            memset(Ab, false, (nrows * ncols) * sizeof(bool));

            for(size_t i = 0; i < nvals; i++){
                Ab[rows[i] * ncols + cols[i]] = true;
                Ax[rows[i] * ncols + cols[i]] = entries[i];
            }
        } else {
            #if 0
                delete Ap; delete Aj; delete Ai; delete Ax;

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
        
        for(size_t i = 0; i < nvals; i++){ 

        }
    }
};