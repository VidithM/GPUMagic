
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <random>
#include <cassert>

#define FREE_ALL							\
{											\
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
}	

#define N 5
#define THREADS_PER_BLOCK 32

__global__ void transposeKernel(double *mat) {
	int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
	int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
	if (tid_x <= tid_y || (tid_x >= N || tid_y >= N)) {
		// on or below diagonal, or out of bounds
		return;
	}
	double tmp = mat[tid_x * N + tid_y];
	mat[tid_x * N + tid_y] = mat[tid_y * N + tid_x];
	mat[tid_y * N + tid_x] = tmp;
}

int main() {
	double *d_mat;
	double *row;	// single row of the matrix on host to memcpy to device; don't need to store entire matrix
	double *result; // result matrix on host for output

	// allocate device matrix
	CU_TRY(cudaMalloc((void**)(&d_mat), N * N * sizeof(double)));
	// allocate result matrix on host
	result = (double*)malloc(N * N * sizeof(double));
	// allocate row vector on host
	row = (double*)malloc(N * sizeof(double));

	std::uniform_real_distribution<double> distr(1.0f, 100.0f);
	std::random_device rd;
	std::mt19937 gen(rd());

	printf("Printing initial matrix:\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			row[j] = distr(gen);
			printf("(%d, %d) entry: %0.3f\n", i, j, row[j]);
		}
		CU_TRY(cudaMemcpy(d_mat + i * N, row, N * sizeof(double), cudaMemcpyHostToDevice));
	}
	free(row);
	dim3 gridDim((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
	dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
	transposeKernel << <gridDim, blockDim >> > (d_mat);
	CU_TRY(cudaPeekAtLastError());
	CU_TRY(cudaDeviceSynchronize());
	
	CU_TRY(cudaMemcpy(result, d_mat, N * N * sizeof(double), cudaMemcpyDeviceToHost));

	printf("Printing transposed matrix:\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("(%d, %d) entry: %0.3f\n", i, j, result[i * N + j]);
		}
	}
}