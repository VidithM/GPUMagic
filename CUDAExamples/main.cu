
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <random>
#include <cassert>

#define CU_TRY(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define N 2000
#define eps 1e-6


__global__ void daxpyKernel(double *a, double *mat, double *query, int *j)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N * N) {
		return;
	}
	int row = tid / N;
	if (row <= *j) {
		return;
	}
	int dif = row - *j;
	mat[tid] -= (a[row]) * mat[tid - dif * N];

	if (tid % N == 0) {
		// first thread in each row is responsible for doing daxpy on the query vector
		query[row] -= (a[row]) * query[*j];
	}
}


__global__ void swapRowsKernel(double* x, double* y, double* qx, double* qy) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	if (tid == 0) {
		// zeroth thread is responsible for swapping query entries
		double tmp = *qx;
		*qx = *qy;
		*qy = tmp;
	}
	double tmp = x[tid];
	x[tid] = y[tid];
	y[tid] = tmp;
}

/*
For now, doing with a single thread (need kernel since need access to device matrix)
TODO: Figure out how to parallelize
*/
__global__ void findSwapRowKernel(double* mat, int *j, int* ans) {
	for (int i = *j; i < N; i++) {
		if (fabs(mat[i * N + *j]) > eps) {
			*ans = i;
			return;
		}
	}
	*ans = -1;
}

/*
* Collects all coefficients needed for daxpy computations on a column.
* Performed after row swap.
*/
__global__ void collectCoeffsKernel(double* mat, double* out, int *j) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	out[tid] = (tid < *j ? 1 : mat[tid * N + *j] / mat[*j * N + *j]);
}

void pr(double** mat) {
	printf("Printing matrix of dim: (%d, %d):\n", N, N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("(%d, %d): %0.3f\n", i, j, mat[i][j]);
		}
	}
}

void pr(double* vec) {
	printf("Printng vector of length: %d\n", N);
	for (int i = 0; i < N; i++) {
		printf("(%d): %0.3f\n", i, vec[i]);
	}
}

/*
* Serial implementation provided by: ...
*/
void serialBenchmark() {

}

int main() {
	double** mat, *d_mat;
	// we will solve Ax = b, where A = mat, b = query
	double* query, *query_orig, *d_query;
	// allocate host matrix
	mat = (double**)malloc(N * sizeof(double*));
	// allocate device matrix
	CU_TRY(cudaMalloc((void**)(&d_mat), N * N * sizeof(double)));
	// allocate device query vector
	CU_TRY(cudaMalloc((void**)(&d_query), N * sizeof(double)));
	// allocate host query vector
	query = (double*)malloc(N * sizeof(double));
	query_orig = (double*)malloc(N * sizeof(double));
	// allocate rows of host matrix
	for (int i = 0; i < N; i++) {
		mat[i] = (double*)malloc(N * sizeof(double));
	}

	std::uniform_real_distribution<double> distr(1.0f, 100.0f);
	std::random_device rd;
	std::mt19937 gen(rd());

	for (int i = 0; i < N; i++) {
		double query_entry = distr(gen);
		// copy into ith query entry
		CU_TRY(cudaMemcpy(d_query + i, &query_entry, sizeof(double), cudaMemcpyHostToDevice));
		query_orig[i] = query_entry;
		for (int j = 0; j < N; j++) {
			// populate host matrix
			mat[i][j] = distr(gen);
		}
		// memcpy rows into device matrix
		CU_TRY(cudaMemcpy((void*)(d_mat + (i * N)), (void*)mat[i], N * sizeof(double), cudaMemcpyHostToDevice));
	}

	double* daxpy_coeffs;
	CU_TRY(cudaMalloc((void**)(&daxpy_coeffs), N * sizeof(double)));

	int* non_zero_row;	// first non-zero row; used for row swap step
	int* curr_col;		// current column used for kernel computations

	// allocate result variables on device
	CU_TRY(cudaMalloc((void**)(&non_zero_row), sizeof(int)));
	CU_TRY(cudaMalloc((void**)(&curr_col), sizeof(int)));

	cudaEvent_t start, end;
	CU_TRY(cudaEventCreate(&start));
	CU_TRY(cudaEventCreate(&end));

	CU_TRY(cudaEventRecord(start));
	for (int j = 0; j < N - 1; j++) {
		// printf("At column %d\n", j);
		// jth column
		// find first i s.t. mat[i][j] != 0, swap ith row and jth row, perform reduction
		// memcpy column into device memory
		CU_TRY(cudaMemcpy(curr_col, &j, sizeof(int), cudaMemcpyHostToDevice));

		// need to parallelize this later; read comments on kernel
		findSwapRowKernel << <1, 1 >> > (d_mat, curr_col, non_zero_row);
		CU_TRY(cudaPeekAtLastError());
		CU_TRY(cudaDeviceSynchronize());

		int targ;
		CU_TRY(cudaMemcpy(&targ, non_zero_row, sizeof(int), cudaMemcpyDeviceToHost));
		if (targ == -1) {
			printf("[ERROR]: Matrix is non-invertible!\n");
			exit(0);
		}
		if (targ != j) {
			swapRowsKernel <<<10, 256>>> (d_mat + targ * N, d_mat + j * N, d_query + targ, d_query + j);
			CU_TRY(cudaPeekAtLastError());
			CU_TRY(cudaDeviceSynchronize());
		}
		collectCoeffsKernel << <10, 256 >> > (d_mat, daxpy_coeffs, curr_col);
		CU_TRY(cudaPeekAtLastError());
		CU_TRY(cudaDeviceSynchronize());
		// double* daxpy_raw = (double*)malloc(N * sizeof(double));
		// CU_TRY(cudaMemcpy(daxpy_raw, daxpy_coeffs, N * sizeof(double), cudaMemcpyDeviceToHost));
		
		daxpyKernel << <20000, 256 >> > (daxpy_coeffs, d_mat, d_query, curr_col);
		CU_TRY(cudaPeekAtLastError());
		CU_TRY(cudaDeviceSynchronize());
	}
	double* reduced = (double*)malloc(N * N * sizeof(double));
	CU_TRY(cudaMemcpy(reduced, d_mat, N * N * sizeof(double), cudaMemcpyDeviceToHost));

	double* result = (double*)malloc(N * sizeof(double));
	CU_TRY(cudaMemcpy(query, d_query, N * sizeof(double), cudaMemcpyDeviceToHost));

	// final solving step is serial for now.
	// parallelization idea: one block per row, and some threads responsible for subtraction step
	// use __syncthreads() for barrier to wait for all subtraction threads to finish, then do division

	for (int i = N - 1; i >= 0; i--) {
		double sum = 0;
		for (int j = N - 1; j > i; j--) {
			sum += reduced[i * N + j] * result[j];
		}
		double rem = query[i] - sum;
		result[i] = rem / reduced[i * N + i];
	}
#ifdef dbg
	pr(result);
	pr(mat);
	pr(query_orig);
#endif
	CU_TRY(cudaEventRecord(end));
	CU_TRY(cudaEventSynchronize(end));
	float millis = 0;
	CU_TRY(cudaEventElapsedTime(&millis, start, end));
	printf("Runtime (ms): %0.3f\n", millis);

	for (int i = 0; i < N; i++) {
		double sum = 0;
		for (int j = 0; j < N; j++) {
			sum += mat[i][j] * result[j];
		}
		if (fabs(query_orig[i] - sum) > eps) {
			printf("[ERROR]: Produced solution is incorrect\n");
			exit(0);
		}
	}
	printf("Verification passed.\n");
#ifdef dbg
	pr(mat);
#endif
}