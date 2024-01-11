#pragma once

// ================= Common headers ====================
#include <iostream>
#include <cassert>
#include <cstring>
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

const int MAX_MSG_LEN = 256;

template <typename... T>
inline __host__ __device__ void DBG(const char msg[MAX_MSG_LEN], T... args){
	char real_msg[MAX_MSG_LEN + 7] = "[DBG]: ";
	std::memcpy(real_msg + 7, msg, MAX_MSG_LEN * sizeof(char));
	printf(real_msg, args...);
}

template <typename... T>
inline __host__ __device__ void ERROR(const char msg[MAX_MSG_LEN], const char *file, size_t line, T... args){
	char real_msg[MAX_MSG_LEN + 29] = "[ERROR]: File: %s, Line: %d: ";
	std::memcpy(real_msg + 29, msg, MAX_MSG_LEN * sizeof(char));
	printf(real_msg, file, line, args...);
	FREE_ALL;
}
// =====================================================