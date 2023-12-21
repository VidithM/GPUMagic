cmake_minimum_required ( VERSION 3.20 )
project ( GPUMagic LANGUAGES CXX CUDA )

find_package ( CUDA REQUIRED )

include_directories ( Include )

set ( CUDA_SOURCES Kernels/matmul_dense.cu )
cuda_add_library ( kernels SHARED ${CUDA_SOURCES} )

add_executable ( matmul Tests/matmul.cpp )
set_target_properties (matmul PROPERTIES LINKER_LANGUAGE CXX )
target_link_libraries ( matmul PRIVATE kernels ${CUDA_LIBRARIES} )