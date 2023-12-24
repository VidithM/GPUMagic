#include "GPUMagic.h"

int main(){
    matrix<float> **res = NULL, *A, *B;

    A = new matrix<float>();
    B = new matrix<float>();

    A->init(
        NULL,
        NULL,
        NULL,
        0,
        5,
        5,
        CPU,
        DENSE
    );
    std::cout << A->exists(0, 0) << std::endl;
    A->put(0, 0, 3.14f);

    std::cout << A->exists(0, 0) << std::endl;
    std::cout << A->at(0, 0) << std::endl;

    // matmul_dense<float>(res, A, B); // fails
    delete A;
    // delete B;
}