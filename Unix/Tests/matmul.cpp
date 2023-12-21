#include "GPUMagic.h"

int main(){
    matrix<float> **res = NULL, *A, *B;

    A = new matrix<float>();
    B = new matrix<float>();

    matmul_dense<float>(res, A, B); // fails
}