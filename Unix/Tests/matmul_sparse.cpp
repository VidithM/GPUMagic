#include "GPUMagic.h"

using MatrixType = float;

// methods (for now): 1: rowscale, 2: colscale

int main(int argc, char **argv){
    if(argc <= 1){
        ERROR("Must specify multiplication method\n", __FILE__, __LINE__);
        return 0;
    }
    if(atoi(argv[1]) == 1){
        matrix<MatrixType> mat(DENSE);
        size_t arr_rows[6] = {0, 2, 4, 5};
        size_t arr_cols[6] = {0, 2, 4, 5};
        MatrixType arr_vals[6] = {1.43, 2.0, 4.0, 3.0, 3.14, 2.89};
        mat.init(arr_vals, arr_rows, arr_cols, 6, 10, 10);
        matrix<MatrixType> *res = NULL;
        matmul_rowscale(&res, &mat, &mat, 1);
        mat.print();
    } else {
        ERROR("colscale test not implemented\n", __FILE__, __LINE__);
    }
}