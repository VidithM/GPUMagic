#include "GPUMagic.h"

using MatrixType = double;
#define eps 1

int main(){
    matrix<MatrixType> *arr = NULL;

    int N = 100'000'000;
    std::random_device rd;
    std::mt19937 gen(rd());
    // #if std::is_floati
    DistrType<MatrixType> distr(1, 1000);

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

    arr = new matrix<MatrixType>(DENSE);
    arr->init(arr_entries, arr_rows, arr_cols, N, 1, N);
    // arr->print();

    timer t;

    printf("GPU Time (span efficient)\n");
    matrix<MatrixType> *res = NULL;
    t.start();
    cumsum_span_efficient(&res, arr, 512);
    t.end();

    delete res;
    res = NULL;

    printf("GPU Time (basic)\n");
    t.start();
    cumsum(&res, arr, (1 << 18), 512);
    t.end();

    printf("CPU Time\n");
    t.start();
    matrix<MatrixType> *cpu_res = new matrix<MatrixType>(DENSE);
    // std::cout << "Passed here" << std::endl;
    cpu_res->init(NULL, NULL, NULL, 0, 1, N);
    MatrixType curr = 0;
    for(int i = 0; i < N; i++){
        curr += arr->at(0, i);
        cpu_res->put(0, i, curr);
    }
    t.end();

    for(int i = 0; i < N; i++){
        if(fabs(cpu_res->at(0, i) - res->at(0, i)) > 0.1) {
            printf("Verification failed %d: Host: %0.5f, Device: %0.5f\n", i, cpu_res->at(0, i), res->at(0, i));
            exit(-1);
        }
    }

    delete res;
    delete[] arr_rows;
    delete[] arr_cols;
    delete[] arr_entries;
}