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
inline __host__ __device__ void _DBG(const char msg[MAX_MSG_LEN], T... args){
	char real_msg[MAX_MSG_LEN + 7] = "[DBG]: ";
	for(size_t i = 0; i < MAX_MSG_LEN; i++){
		real_msg[7 + i] = msg[i];
	}
	printf(real_msg, args...);
}

template <typename... T>
inline __host__ __device__ void _ERROR(const char msg[MAX_MSG_LEN], const char *file, size_t line, T... args){
	char real_msg[MAX_MSG_LEN + 29] = "[ERROR]: File: %s, Line: %d: ";
	for(size_t i = 0; i < MAX_MSG_LEN; i++){
		real_msg[29 + i] = msg[i];
	}
	printf(real_msg, file, line, args...);
	FREE_ALL;
}

inline __host__ __device__ void _ASSERT(bool val, const char *file, size_t line){
	if(!val){
		_ERROR("Assertion failed\n", file, line);
	}
}


#define DBG 					_DBG
#define ERROR 					_ERROR// _ERROR(msg, __FILE__, __LINE__, args)
#define ASSERT(val)				_ASSERT(val, __FILE__, __LINE__)
// =====================================================