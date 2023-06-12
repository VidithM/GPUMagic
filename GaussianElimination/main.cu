
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <random>
#include <cassert>

#define N 4096
#define eps 1e-6
#define tol 1e-5

// #define DBG
#define SHOW_PROGRESS

#define THREADS_PER_BLOCK 512

// #define USE_TRANSPOSE

#define FREE_ALL							\
{									\
	cudaFree((void*)d_mat);						\
	cudaFree((void*)d_query);					\
	cudaFree((void*)non_zero_row);					\
	cudaFree((void*)curr_col);					\
									\
	for (int i = 0; i < N; i++) {					\
		free((void*)mat[i]);					\
	}								\
									\
	free((void*)mat);						\
	free((void*)column);					\
	free((void*)reduced);						\
	free((void*)result);						\
}									\

#define CU_TRY(ans)							\
{									\
	if (ans != cudaSuccess) {					\
		fprintf(						\
			stderr,						\
			"GPUassert: %s %s %d\n",			\
			cudaGetErrorString(ans),			\
			__FILE__,					\
			__LINE__					\
		);							\
		FREE_ALL;						\
		if (abort) exit(ans);					\
	}								\
}									\


__global__ void daxpyKernel(double *mat, double *query, double *a, int *j)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N * N) {
		return;
	}
#ifdef USE_TRANSPOSE
	// make the blocks go left-right in the transpose
	int row = tid / N;
	int col = tid % N;

	if (col <= *j) {
		return;
	} 
	__shared__ double to_scale;
	to_scale = mat[row * N + *j];

	int dif = col - *j;
	mat[tid] -= (a[col] * mat[tid - dif]);

	if (row == 0) {
		// first thread in each row is responsible for doing daxpy on the query vector
		query[col] -= (a[col]) * query[*j];
	}
#else
	// make the blocks go left-right
	int row = tid / N;
	if (row <= *j) {
		return;
	}
	// printf("%d %d\n", row, tid % N);
	int dif = row - *j;
	mat[tid] -= (a[row]) * mat[tid - dif * N];

	if (tid % N == 0) {
		// first thread in each row is responsible for doing daxpy on the query vector
		query[row] -= (a[row]) * query[*j];
	}
#endif
}


__global__ void swapRowsKernel(double *mat, double *query, int *swap_top, int *swap_bottom) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	if (tid == 0) {
		// zeroth thread is responsible for swapping query entries
		double tmp = query[*swap_top];
		query[*swap_top] = query[*swap_bottom];
		query[*swap_bottom] = tmp;
	}
#ifdef USE_TRANSPOSE
	double tmp = mat[tid * N + *swap_top];
	mat[tid * N + *swap_top] = mat[tid * N + *swap_bottom];
	mat[tid * N + *swap_bottom] = tmp;
#else
	double tmp = mat[*swap_top * N + tid];
	mat[*swap_top * N + tid] = mat[*swap_bottom * N + tid];
	mat[*swap_bottom * N + tid] = tmp;
#endif
}

/*
For now, doing with a single thread (need kernel since need access to device matrix)
TODO: Figure out how to parallelize
*/
__global__ void findSwapRowKernel(double *mat, int *j, int *ans) {
	for (int i = *j; i < N; i++) {
		int idx;
#ifdef USE_TRANSPOSE
		idx = *j * N + i;
#else
		idx = i * N + *j;
#endif
		if (fabs(mat[idx]) > eps) {
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
__global__ void collectCoeffsKernel(double *mat, double *out, int *j) { 
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	// this is constant across all threads in this block; no need for separate reads
	// in fact, this is constant across all blocks - if there was an L1 cache like shared memory that ALL blocks had access to,
	// we could use that
	// __shared__ double div;
	// double div = mat[*j * N + *j];

	// old code:
#ifdef USE_TRANSPOSE
	__shared__ double div;
	div = mat[*j * N + *j];
	out[tid] = (tid < *j ? 1 : mat[*j * N + tid] / div);
#else
	out[tid] = (tid < *j ? 1 : mat[tid * N + *j] / mat[*j * N + *j]);
#endif
	// out[tid] = (tid < *j ? 1 : mat[tid * N + *j] / div);
}

void pr(double** mat) {
	printf("Printing matrix of dim: (%d, %d):\n", N, N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("(%d, %d): %0.3f\n", i, j, mat[i][j]);
		}
	}
}

void pr(double *vec) {
	printf("Printng vector of length: %d\n", N);
	for (int i = 0; i < N; i++) {
		printf("(%d): %0.3f\n", i, vec[i]);
	}
}

/*
Idea:
Lets store the transpose of mat as d_mat
This will let us get good spatial locality on all operations, in addition to letting us do the
shared memory technique for daxpy quickly
*/

// host and device matrices (host matrix stores the original matrix, d_mat will become the reduced matrix)
// note: if USE_TRANSPOSE is set, d_mat is stored as the transpose of mat
double **mat, *d_mat;

// build each column of the mat matrix if USE_TRANSPOSE to easily transfer to d_mat 
double *column;

// we will solve Ax = b, where A = mat, b = query
double *query, *query_orig, *d_query;

int *non_zero_row;		// first non-zero row; used for row swap step
int *curr_col;			// current column used for kernel computations
double *daxpy_coeffs;   	// row scalars needed for daxpy step; computed after row swap

double *reduced;		// reduced matrix copied to host
double *result;			// resulting solution computed on host

/*
* Serial implementation provided by ...
* 
*/
void serialBenchmark(double **a, double *ans) {}

int main() {

	// allocate host matrix
	mat = (double**)malloc(N * sizeof(double*));
	// allocate device matrix
	CU_TRY(cudaMalloc((void**)(&d_mat), N * N * sizeof(double)));
#ifdef USE_TRANSPOSE
	column = (double*)malloc(N * sizeof(double));
#endif
	// allocate host query vector
	query = (double*)malloc(N * sizeof(double));
	// original query vector for validation at end
	query_orig = (double*)malloc(N * sizeof(double));
	// allocate device query vector
	CU_TRY(cudaMalloc((void**)(&d_query), N * sizeof(double)));
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
#ifndef USE_TRANSPOSE
		CU_TRY(cudaMemcpy((void*)(d_mat + i * N), (void*)mat[i], N * sizeof(double), cudaMemcpyHostToDevice));
#endif
	}

#ifdef USE_TRANSPOSE
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			column[i] = mat[i][j];
		}
		CU_TRY(cudaMemcpy((void*)(d_mat + j * N), (void*)column, N * sizeof(double), cudaMemcpyHostToDevice));
	}
#endif

	// allocate result variables on device
	CU_TRY(cudaMalloc((void**)(&non_zero_row), sizeof(int)));
	CU_TRY(cudaMalloc((void**)(&curr_col), sizeof(int)));
	CU_TRY(cudaMalloc((void**)(&daxpy_coeffs), N * sizeof(double)));

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

		// find first non-zero row
		findSwapRowKernel << <1, 1 >> > (d_mat, curr_col, non_zero_row);
		CU_TRY(cudaPeekAtLastError());
		CU_TRY(cudaDeviceSynchronize());

		int targ;
		CU_TRY(cudaMemcpy(&targ, non_zero_row, sizeof(int), cudaMemcpyDeviceToHost));
		if (targ == -1) {
			// no row with non-zero entry was found
			printf("[ERROR]: Matrix is non-invertible!\n");
			FREE_ALL;
			exit(0);
		}
		if (targ != j) {
			// only swap if found row != j
			swapRowsKernel << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_mat, d_query, curr_col, non_zero_row);
			CU_TRY(cudaPeekAtLastError());
			CU_TRY(cudaDeviceSynchronize());
		}
		// collect daxpy coefficients for the jth column
		collectCoeffsKernel << < (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_mat, daxpy_coeffs, curr_col);
		CU_TRY(cudaPeekAtLastError());
		CU_TRY(cudaDeviceSynchronize());
		
		// perform daxpy step
		daxpyKernel << <(N * N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_mat, d_query, daxpy_coeffs, curr_col);
		CU_TRY(cudaPeekAtLastError());
		CU_TRY(cudaDeviceSynchronize());

#ifdef SHOW_PROGRESS
		if ((j & 127) == 0) {
			// for progress checking
			printf("[%0.3f%%] complete\n", j * 100.0 / N);
		}
#endif
	}
	reduced = (double*)malloc(N * N * sizeof(double));
	// copy reduced matrix from device to host
	CU_TRY(cudaMemcpy(reduced, d_mat, N * N * sizeof(double), cudaMemcpyDeviceToHost));

#ifdef USE_TRANSPOSE
	// flip entries across diagonal
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i < j) {
				std::swap(reduced[i * N + j], reduced[j * N + i]);
			}
		}
	}
#endif

	if (fabs(reduced[(N - 1) * N + (N - 1)]) < eps) {
		// last row is all zeroes; non-invertible
		printf("[ERROR]: Matrix is non-invertible!\n");
		FREE_ALL;
		exit(0);
	}
	
	result = (double*)malloc(N * sizeof(double));
	
	// copy reduced query vector to host
	CU_TRY(cudaMemcpy(query, d_query, N * sizeof(double), cudaMemcpyDeviceToHost));

	// solve for result
	for (int i = N - 1; i >= 0; i--) {
		double sum = 0;
		for (int j = N - 1; j > i; j--) {
			sum += reduced[i * N + j] * result[j];
		}
		double rem = query[i] - sum;
		result[i] = rem / reduced[i * N + i];
	}
#ifdef DBG
	pr(result);
	pr(mat);
	pr(query_orig);
#endif
	CU_TRY(cudaEventRecord(end));
	CU_TRY(cudaEventSynchronize(end));

	float gpu_millis = 0;
	CU_TRY(cudaEventElapsedTime(&gpu_millis, start, end));
	printf("GPU runtime (ms): %0.3f\n", gpu_millis);

	// verify that result is correct
	double mx_error = 0;
	for (int i = 0; i < N; i++) {
		double sum = 0;
		for (int j = 0; j < N; j++) {
			sum += mat[i][j] * result[j];
		}
		if (sum != sum) {
			// sum is NaN, something went wrong
			printf("[ERROR]: Obtained NaN value when solving\n");
			FREE_ALL;
			exit(0); 
		}
		mx_error = std::max(mx_error, fabs(query_orig[i] - sum));
	}
	if (mx_error > tol) {
		printf("[ERROR]: Produced solution is incorrect w/ residual %0.8f\n", mx_error);
		FREE_ALL;
		exit(0);
	}
	printf("Verification passed.\n");

#if 0
	printf("Measuring serial performance...\n");
	CU_TRY(cudaEventRecord(start));
	serialBenchmark(mat, result);
	CU_TRY(cudaEventRecord(end));
	CU_TRY(cudaEventSynchronize(end));

	float cpu_millis = 0;
	CU_TRY(cudaEventElapsedTime(&cpu_millis, start, end));
	printf("CPU runtime (ms): %0.3f\n", cpu_millis);
#endif

#ifdef DBG
	pr(mat);
#endif
	FREE_ALL;
}
