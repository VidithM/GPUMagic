
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <random>

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

/*
* Arguments are pointers to the rows; see below
*/
__global__ void saxpyKernel(float *a, float *x, float *y)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N * N) {
		return;
	}
	int row = tid / N;
	y[tid] -= (a[row]) * x[tid];
}

/*
* Arguments here are pointers to the rows; to fetch a certain entry in the row,
* need to do *x[i]
*/
__global__ void swapRowsKernel(float* x, float* y) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	float tmp = x[tid];
	x[tid] = y[tid];
	y[tid] = tmp;
}

/*
For now, doing with a single thread (need kernel since need access to device matrix)
TODO: Figure out how to parallelize
*/
__global__ void findSwapRowKernel(float* mat, int *j, int* ans) {
	for (int i = *j; i < N; i++) {
		if (fabs(mat[i * N + *j]) > eps) {
			*ans = i;
			return;
		}
	}
	*ans = -1;
}

/*
* Collects all coefficients needed for saxpy computations on a column.
* Performed after row swap.
*/
__global__ void collectCoeffsKernel(float* mat, float* out, int *j) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) {
		return;
	}
	out[tid] = (tid < *j ? 1 : mat[tid * N + *j] / mat[*j * N + *j]);
}

void saxpySerial(float a, float* x, float* y) {
	for (int i = 0; i < N; i++) {
		y[i] = a * x[i];
	}
}

void pr(float** mat) {
	printf("Printing matrix of dim: (%d, %d):\n", N, N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("(%d, %d): %0.3f\n", i, j, mat[i][j]);
		}
	}
}

int main() {
	float** mat, *d_mat;
	// allocate host matrix
	mat = (float**)malloc(N * sizeof(float*));
	// allocate device matrix
	CU_TRY(cudaMalloc((void**)(&d_mat), N * N * sizeof(float)));

	// allocate rows of host matrix
	for (int i = 0; i < N; i++) {
		mat[i] = (float*)malloc(N * sizeof(float));
	}

	std::uniform_real_distribution<float> distr(1.0f, 100.0f);
	std::random_device rd;
	std::mt19937 gen(rd());

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			// populate host matrix
			mat[i][j] = distr(gen);
		}
		// memcpy rows into device matrix
		CU_TRY(cudaMemcpy((void*)(d_mat + (i * N)), (void*)mat[i], N * sizeof(float), cudaMemcpyHostToDevice));
	}

	float* saxpy_coeffs;
	CU_TRY(cudaMalloc((void**)(&saxpy_coeffs), N * sizeof(float)));

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
		printf("At column %d\n", j);
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
			swapRowsKernel <<<10, 256>>> (d_mat + targ * N, d_mat + j * N);
			CU_TRY(cudaPeekAtLastError());
			CU_TRY(cudaDeviceSynchronize());
		}
		collectCoeffsKernel << <10, 256 >> > (d_mat, saxpy_coeffs, curr_col);
		CU_TRY(cudaPeekAtLastError());
		CU_TRY(cudaDeviceSynchronize());
		float* saxpy_raw = (float*)malloc(N * sizeof(float));
		CU_TRY(cudaMemcpy(saxpy_raw, saxpy_coeffs, N * sizeof(float), cudaMemcpyDeviceToHost));
		
		// N threads operate on separate columns of the row
		saxpyKernel << <20000, 256 >> > (saxpy_coeffs, d_mat, d_mat);
		CU_TRY(cudaPeekAtLastError());
		CU_TRY(cudaDeviceSynchronize());
	}
	CU_TRY(cudaEventRecord(end));
	CU_TRY(cudaEventSynchronize(end));
	float millis = 0;
	CU_TRY(cudaEventElapsedTime(&millis, start, end));
	printf("Runtime (ms): %0.3f\n", millis);
	/*
	float* raw = (float*)malloc(N * N * sizeof(float));
	CU_TRY(cudaMemcpy(raw, d_mat, N * N * sizeof(float), cudaMemcpyDeviceToHost));
	int at = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			mat[i][j] = raw[at];
			at++;
		}
	}
	pr(mat);
	*/
}