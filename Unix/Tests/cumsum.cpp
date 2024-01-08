#include "GPUMagic.h"
#include <random>

using MatrixType = double;

int main(){
    matrix<MatrixType> *arr = NULL;

    int N = 100'000'000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<MatrixType> distr(1, 1000);

    size_t *arr_rows = new size_t[N];
    memset(arr_rows, 0, sizeof(arr_rows));
    size_t *arr_cols = new size_t[N];
    for(int i = 0; i < N; i++){
        arr_cols[i] = i;
    }
    MatrixType *arr_entries = new MatrixType[N];
    for(int i = 0; i < N; i++){
        arr_entries[i] = distr(gen);
    }

    arr = new matrix<MatrixType>(DENSE, CPU);
    arr->init(arr_entries, arr_rows, arr_cols, N, 1, N);
    // arr->print();

    timer t;

    CU_PRINT("GPU Time\n", "");
    matrix<MatrixType> *res = NULL;
    t.start();
    cumsum(&res, arr, 1000);
    t.end();
    // res->print();

    CU_PRINT("CPU Time\n", "");
    t.start();
    matrix<MatrixType> *cpu_res = new matrix<MatrixType>(DENSE, CPU);
    // std::cout << "Passed here" << std::endl;
    cpu_res->init(NULL, NULL, NULL, 0, 1, N);
    MatrixType curr = 0;
    for(int i = 0; i < N; i++){
        curr += arr->at(0, i);
        cpu_res->put(0, i, curr);
    }
    t.end();
    
    delete res;
    delete[] arr_rows;
    delete[] arr_cols;
    delete[] arr_entries;
}