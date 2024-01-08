cmake_minimum_required ( VERSION 3.20 )
project ( GPUMagic LANGUAGES CXX CUDA )

set ( CMAKE_VERBOSE_MAKEFILE OFF )

find_package ( CUDA REQUIRED )

include_directories ( Include )

set ( CMAKE_CUDA_SEPARABLE_COMPILATION ON )
set ( CMAKE_POSITION_INDEPENDENT_CODE ON )

set ( CUDA_KERNEL_SOURCES 
    Kernels/matmul_dense.cu
    Kernels/cumsum.cu
    Utils/matrix.cu
)

cuda_add_library ( kernels SHARED ${CUDA_KERNEL_SOURCES} )

# Make device code relocatable (allows a kernel to call a __device__ defined in other compilation units)
# set_target_properties (kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options( kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -rdc=true )

file ( GLOB tests "Tests/*.cpp" )

foreach ( file ${tests} )
    get_filename_component ( EXC_NAME ${file} NAME_WE )
    add_executable ( ${EXC_NAME} ${file} )
    set_target_properties ( ${EXC_NAME} PROPERTIES LINKER_LANGUAGE CUDA )
    target_link_libraries ( ${EXC_NAME} PRIVATE kernels ${CUDA_LIBRARIES} )
endforeach ()