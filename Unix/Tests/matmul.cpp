#include "GPUMagic.h"

int main(){
    matrix<float> **res = NULL, *A, *B;

    A = new matrix<float>(DENSE, CPU);
    B = new matrix<float>(DENSE, CPU);
    A->init(NULL, NULL, NULL, 0, 13, 1);
    B->init(NULL, NULL, NULL, 0, 1, 1);
    std::cout << A->exists(0, 0) << std::endl;
    A->put(0, 0, 3.14f);

    std::cout << A->exists(0, 0) << std::endl;
    std::cout << A->at(0, 0) << std::endl;

    matmul_dense((matrix<float>**) NULL, A, B, 10, 10);

    delete A;
    delete B;
}