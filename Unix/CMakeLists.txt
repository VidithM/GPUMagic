cmake_minimum_required ( VERSION 3.20 )
project ( GPUMagic LANGUAGES CXX CUDA )

set ( CMAKE_VERBOSE_MAKEFILE ON )

find_package ( CUDA REQUIRED )

include_directories ( Include )

set ( CMAKE_CUDA_SEPARABLE_COMPILATION ON )
set ( CMAKE_POSITION_INDEPENDENT_CODE ON )

set ( CUDA_KERNEL_SOURCES 
    Kernels/matmul_dense.cu
    Utils/matrix.cu
)

cuda_add_library ( kernels SHARED ${CUDA_KERNEL_SOURCES} )

# Make device code relocatable (allows a kernel to call a __device__ defined in other compilation units)
# set_target_properties (kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options( kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -rdc=true )

add_executable ( matmul Tests/matmul.cpp )
set_target_properties ( matmul PROPERTIES LINKER_LANGUAGE CUDA )
target_link_libraries ( matmul PRIVATE kernels ${CUDA_LIBRARIES} )