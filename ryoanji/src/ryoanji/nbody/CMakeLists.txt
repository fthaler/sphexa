if(CMAKE_HIP_COMPILER)
    set_source_files_properties(traversal_gpu.cu upsweep_gpu.cu rtfmm_mpole.cpp PROPERTIES LANGUAGE HIP)
elseif(CMAKE_CUDA_COMPILER)
    set_source_files_properties(rtfmm_mpole.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_library(ryoanji_rtfmm OBJECT rtfmm_mpole.cpp)
target_include_directories(ryoanji_rtfmm PUBLIC ${PROJECT_SOURCE_DIR}/src ${CSTONE_DIR} ${BLAS_INCLUDE_DIR})

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(ryoanji_kernels OBJECT traversal_gpu.cu upsweep_gpu.cu)
    target_include_directories(ryoanji_kernels PUBLIC ${PROJECT_SOURCE_DIR}/src ${CSTONE_DIR})
endif()
