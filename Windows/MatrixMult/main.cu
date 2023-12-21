
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>

#include <random>

// #define USE_TILING
#define TILE_N 4

#define N 1024
#define N_LOG2 7
#define IGNORE_P2 0

#define tol 1e-6

#define SHOW_PROGRESS

#define FREE_ALL {}

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

constexpr inline int static_popcount(const int x) {
	int ans = 0;
	int xx = x;
	while (xx) {
		ans += (xx & 1);
		xx >>= 1;
	}
	return ans;
}

__global__ void multKernel(double *res, double *a)
{
	int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
	int tid_y = blockDim.y * blockIdx.y + threadIdx.y;

	// for now, no need to check that tid_x, tid_y are in bounds since N must be a multiple of TILE_N
	// in other words, the matrix is perfectly tiled

	int l_row = tid_y;
	int r_col = tid_x;
	// printf("in kernel, (r, c): (%d, %d)\n", l_row, r_col);

#ifdef USE_TILING
	// cache all rows/cols needed for this tile
	__shared__ double need[TILE_N + TILE_N][N];
	// compute ranges of rows/cols that this tile covers
	int row_start = blockDim.y * blockIdx.y, row_end = row_start + blockDim.y - 1;
	int col_start = blockDim.x * blockIdx.x, col_end = col_start + blockDim.x - 1;

	// threads along the top and left edges of the tile will be responsible
	// for caching rows/cols
	if (r_col == col_start) {
		// cache the row
		int pos = l_row - row_start;
		for (int i = 0; i < N; i++) {
			need[pos][i] = a[l_row * N + i];
		}
	}
	if (l_row == row_start) {
		// cache the column
		int pos = r_col - col_start + 1 + (row_end - row_start);
		for (int i = 0; i < N; i++) {
			need[pos][i] = a[i * N + r_col];
		}
	}
	// need to syncrhonize all tile threads since need[] might not be
	// computed yet
	__syncthreads();
	// multiply
	for (int i = 0; i < N; i++) {
		double x = need[l_row - row_start][i];
		double y = need[(r_col - col_start + row_end - row_start + 1)][i];
		// printf("for (%d, %d), multiplying (%0.3f, %0.3f)\n", l_row, r_col, x, y);
		res[l_row * N + r_col] += x * y;
	}
#else 
	for (int i = 0; i < N; i++) {
		double x = a[l_row * N + i];
		double y = a[i * N + r_col];
		res[l_row * N + r_col] += x * y;
	}
	// printf("for (%d, %d), answer is (%0.3f)\n", l_row, r_col, res[l_row * N + r_col]);
#endif
}

void pr2D(double** mat) {
	printf("Printing matrix of dim: (%d, %d):\n", N, N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%0.3f\t", mat[i][j]);
		}
		printf("\n");
	}
}

double **mat, **true_result, *d_mat;
double *d_res_mat, *result;

int main()
{
	static_assert(IGNORE_P2 || static_popcount(N) == 1, "N should be a power of 2");
	static_assert((N & (TILE_N - 1)) == 0, "N should be a multiple of tile length");

	// allocate host matrix
	mat = (double**) malloc(N * sizeof(double*));
	// allocate device matrix
	CU_TRY(cudaMalloc((void**)(&d_mat), N * N * sizeof(double)));
	// allocate result device matrix
	CU_TRY(cudaMalloc((void**)(&d_res_mat), N * N * sizeof(double)));
	CU_TRY(cudaMemset((void*)(d_res_mat), 0, N * N * sizeof(double)));

	// allocate rows of host matrix
	for (int i = 0; i < N; i++) {
		mat[i] = (double*)malloc(N * sizeof(double));
	}

	std::uniform_real_distribution<double> distr(1.0f, 100.0f);
	std::random_device rd;
	std::mt19937 gen(rd());

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			// populate host matrix
			mat[i][j] = distr(gen);
		}
		CU_TRY(cudaMemcpy((void*)(d_mat + i * N), (void*)mat[i], N * sizeof(double), cudaMemcpyHostToDevice));
	}

	// pr2D(mat);

	// block will use tile dimensions
	dim3 blockDim(TILE_N, TILE_N);
	// grid will use N (for now, N must be a multiple of tile length)
	dim3 gridDim(N / TILE_N, N / TILE_N);

	cudaEvent_t start, end;

	CU_TRY(cudaEventCreate(&start));
	CU_TRY(cudaEventCreate(&end));

	CU_TRY(cudaEventRecord(start));
	multKernel << <gridDim, blockDim >> > (d_res_mat, d_mat);
	CU_TRY(cudaPeekAtLastError());
	CU_TRY(cudaEventRecord(end));

	CU_TRY(cudaDeviceSynchronize());

	float gpu_millis = 0;
	CU_TRY(cudaEventElapsedTime(&gpu_millis, start, end));
	printf("GPU runtime (ms): %0.3f\n", gpu_millis);

	result = (double*)malloc(N * N * sizeof(double));
	CU_TRY(cudaMemcpy(result, d_res_mat, N * N * sizeof(double), cudaMemcpyDeviceToHost));

	clock_t cpu_start = clock();
	true_result = (double**)malloc(N * sizeof(double*));

	for (int i = 0; i < N; i++) {
		true_result[i] = (double*)malloc(N * sizeof(double));
		memset(true_result[i], 0, N * sizeof(double));

		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				true_result[i][j] += mat[i][k] * mat[k][j];
			}
		}
	}

	clock_t cpu_end = clock();

	float cpu_secs = ((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC);
	printf("CPU runtime (ms): %0.3f\n", 1000.0f * cpu_secs);


	printf("Starting verification...\n");
	double mx_error = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			// printf("(%d, %d), (%0.3f, %0.3f)\n", i, j, true_result[i][j], result[i * N + j]);
			mx_error = std::max(mx_error, fabs(true_result[i][j] - result[i * N + j]));
		}
	}
	if (mx_error > tol) {
		printf("[ERROR]: Produced solution is incorrect w/ error %0.8f\n", mx_error);
		exit(0);
	}
	printf("Verification passed\n");
}