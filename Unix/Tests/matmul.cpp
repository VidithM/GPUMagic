#include "GPUMagic.h"

int main(){
    matrix<float> *A, *B;

    A = new matrix<float>(DENSE);
    B = new matrix<float>(DENSE);

    size_t A_rows[4] = {0, 0, 1, 1}; size_t A_cols[4] = {1, 3, 1, 3};
    size_t B_rows[4] = {1, 3, 1, 3}; size_t B_cols[4] = {0, 0, 1, 1};
    float  A_vals[4] = {1, 2, 3, 4}; float  B_vals[4] = {1, 2, 3, 4};

    A->init(A_vals, A_rows, A_cols, 4, 2, 4);
    B->init(B_vals, B_rows, B_cols, 4, 4, 2);
    A->print();
    B->print();
    std::cout << A->exists(0, 0) << std::endl;
    // A->put(0, 0, 3.14f);

    std::cout << A->exists(0, 3) << std::endl;
    std::cout << A->at(0, 3) << std::endl;

    matrix<float> *res = NULL;
    matmul_dense(&res, A, B, 10, 10);

    res->print();
    delete res;

    delete A;
    delete B;
}