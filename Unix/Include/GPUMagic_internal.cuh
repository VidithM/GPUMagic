#pragma once

// ================= Utils/common headers =================
#ifndef FREE_ALL
    #define FREE_ALL {}
#endif

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstring>
#include <cassert>

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
inline __host__ __device__ void _DBG(const char msg[MAX_MSG_LEN], const char *file, size_t line, T... args){
	char real_msg[26 + MAX_MSG_LEN] = "[DBG]: File: %s, Line %d: ";
	memcpy(real_msg + 26, msg, MAX_MSG_LEN * sizeof(char));
	printf(real_msg, file, line, args...);
}

template <typename... T>
inline __host__ __device__ void _ERROR(const char msg[MAX_MSG_LEN], const char *file, size_t line, T... args){
	char real_msg[28 + MAX_MSG_LEN] = "[ERROR]: File: %s, Line %d: ";
	memcpy(real_msg + 28, msg, MAX_MSG_LEN * sizeof(char));
	printf(real_msg, file, line, args...);
	FREE_ALL;
	assert(0);
}

inline __host__ __device__ void _ASSERT(bool val, const char *file, size_t line){
	if(!val){
		_ERROR("Assertion failed\n", file, line);
	}
}



#define DBG(msg, args...)		_DBG(msg, __FILE__, __LINE__, ##args)
#define ERROR(msg, args...) 	_ERROR(msg, __FILE__, __LINE__, ##args)
#define ASSERT(val)				_ASSERT(val, __FILE__, __LINE__)

#define CU_INIT(device)								            \
cudaDeviceProp props##device;									\
{																\
	int deviceCount;											\
	CU_TRY(cudaGetDeviceCount(&deviceCount));					\
	if (deviceCount <= device){								    \
		ERROR("Device not present\n");                          \
	}													        \
	CU_TRY(cudaGetDeviceProperties(&props##device, device));    \
}
// ========================================================