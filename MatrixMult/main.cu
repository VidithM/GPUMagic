
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>

#define USE_TILING
#ifdef USE_TILING
	#define TILE_N 8
#endif

#define N 2048
#define N_LOG2 11 
#define IGNORE_P2 0

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

	int l_row = tid_y;
	int r_col = tid_x;

#ifdef USE_TILING
	// cache all rows/cols needed for this tile
	__shared__ double need[TILE_N + TILE_N][N];
	// compute ranges of rows/cols that this tile covers
	int row_start = blockDim.y * blockIdx.y, row_end = row_start + blockDim.y - 1;
	int col_start = blockDim.x * blockIdx.x, col_end = col_start + blockDim.x - 1;
	int at = 0; // which entry in need
	if (l_row == row_start && r_col == col_start) {
		// only run this code for one thread per tile (the top left thread)
		// rows
		for (int i = row_start; i <= row_end; i++) {
			for (int j = 0; j < N; j++) {
				need[at][j] = a[i * N + j];
				at++;
			}
		}
		// cols
		for (int i = col_start; i <= col_end; i++) {
			for (int j = 0; j < N; j++) {
				need[at][j] = a[j * N + i];
				at++;
			}
		}
	}
	// need to syncrhonize all tile threads since need[] might not be
	// computed yet
	__syncthreads();
	// multiply
	for (int i = 0; i < N; i++) {
		double x = need[l_row - row_start][i];
		double y = need[(r_col + col_start - row_start) - row_start][i];
		res[l_row * N + r_col] += x * y;
	}
#else 
	for (int i = 0; i < N; i++) {
		double x = a[l_row][i];
		double y = a[r_col][i];
		res[l_row * N + r_col] += x * y;
	}
#endif
}

double** mat, *d_mat;

int main()
{
	static_assert(IGNORE_P2 || static_popcount(N) == 1, "N should be a power of 2");
	static_assert((N & (TILE_N - 1)) == 0, "N should be a multiple of tile length");

	// allocate host matrix
	mat = (double**) malloc(N * sizeof(double*));
	// allocate device matrix
	CU_TRY(cudaMalloc((void**)(&d_mat), N * N * sizeof(double)));

}