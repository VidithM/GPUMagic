// Main include file for all sources

// ================== CUDA headers =====================
#define CU_TRY(ans)					    \
{								    	\
	if (ans != cudaSuccess) {		    \
		fprintf(					    \
			stderr,					    \
			"GPUassert: %s %s %d\n",    \
			cudaGetErrorString(ans),	\
			__FILE__,					\
			__LINE__					\
		);							    \
		FREE_ALL;						\
		if (abort) exit(ans);			\
	}								    \
}

#ifndef FREE_ALL
    #define FREE_ALL {}
#endif
// =====================================================


// ======  Utils/types/common headers, typedefs, etc.===
#include <iostream>
#include <cassert>

#include "matrix.h"
// =====================================================


// ===============  Kernel declarations ================
template <typename T>
void matmul_dense (
	matrix<T> **res,
	matrix<T> *A,
	matrix<T> *B
);
// =====================================================