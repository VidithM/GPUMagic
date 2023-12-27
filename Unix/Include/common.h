#pragma once

// ================= Common headers ====================
#include <iostream>
#include <cassert>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// =====================================================


// ================== CUDA headers =====================
#ifndef FREE_ALL
    #define FREE_ALL {}
#endif

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
		exit(ans);						\
	}								    \
}

#define CU_PRINT(fmt, args...)                  \
{                                               \
    printf("[CU_PRINT]: ");                     \
    printf(fmt, args);                          \
}  								                \

#define CU_ERROR(fmt, args...)  		             \
{                               		             \
    printf("[CU_ERROR]: File: %s, Line: %d: ", 	 	 \
            __FILE__,                                \
            __LINE__                                 \
    );                                               \
    printf(fmt, args);                         		 \
	FREE_ALL;							             \
	exit(-1);							             \
}  										             \
// =====================================================