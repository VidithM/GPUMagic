#include "GPUMagic.h"

int main(){
    matrix<float> *arr = NULL;

    float arr_entries[4] = {1, 2, 3, 4};
    size_t arr_rows[4] = {0, 0, 0, 0};
    size_t arr_cols[4] = {0, 1, 2, 3};

    arr = new matrix<float>(DENSE, CPU);
    arr->init(arr_entries, arr_rows, arr_cols, 4, 1, 4);

    matrix<float> *res = NULL;
    cumsum(&res, arr, 2);
    res->print();
}