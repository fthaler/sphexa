include(cstone_add_test)

function(addMpiTest source exename testname ranks)
    add_executable(${exename} ${source})
    target_include_directories(${exename} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(${exename} PRIVATE ${PROJECT_SOURCE_DIR}/test)
    target_include_directories(${exename} PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(${exename} PRIVATE -Wno-unknown-pragmas)
    target_link_libraries(${exename} PRIVATE ${MPI_CXX_LIBRARIES} GTest::gtest_main)
    cstone_add_test(${testname} EXECUTABLE ${exename} RANKS ${ranks})
    install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/integration_mpi)
endfunction()

function(addCstoneMpiTest source exename testname ranks)
    list(APPEND source test_main.cpp)
    addMpiTest("${source}" ${exename} ${testname} ${ranks})
endfunction()

addCstoneMpiTest(globaloctree.cpp globaloctree GlobalTreeTests 2)
addCstoneMpiTest(exchange_halos.cpp exchange_halos GlobalHaloExchange 2)
addCstoneMpiTest(box_mpi.cpp box_mpi GlobalBox 2)
addCstoneMpiTest(exchange_focus.cpp exchange_focus GlobalFocusExchange 2)
addCstoneMpiTest(exchange_keys.cpp exchange_keys GlobalKeyExchange 2)
addCstoneMpiTest(domain_2ranks.cpp domain_2ranks GlobalDomain2Ranks 2)
addCstoneMpiTest(domain_resize.cpp domain_resize GlobalDomainResize 2)
addCstoneMpiTest(exchange_ptype.cpp exchange_ptype GlobalPtypeExchange 2)
addCstoneMpiTest(fluid_halos.cpp fluid_halos GlobalFluidHalos 2)
addCstoneMpiTest(zero_fluid_rank.cpp zero_fluid_rank GlobalZeroFluidRank 2)

# ve-pt unconverged-particle removal: full-propagator test, needs main/src + sph + ryoanji headers.
# Guarded on being part of the sphexa build: SPH_EXA_WITH_CUDA/HIP are defined by the sphexa root
# project only, so standalone cstone builds (domain/-only) skip the test instead of failing on the
# missing headers.
if (DEFINED SPH_EXA_WITH_CUDA AND EXISTS "${CMAKE_SOURCE_DIR}/main/src/propagator/ve_hydro_pt.hpp")
    addCstoneMpiTest(unconverged_removal.cpp unconverged_removal GlobalUnconvergedRemoval 2)
    target_include_directories(unconverged_removal PRIVATE ${CMAKE_SOURCE_DIR}/main/src
                              ${CMAKE_SOURCE_DIR}/sph/include ${CMAKE_SOURCE_DIR}/ryoanji/src)
else ()
    message(STATUS "Skipping GlobalUnconvergedRemoval: not building inside the sphexa tree")
endif ()

addCstoneMpiTest(treedomain.cpp treedomain GlobalDomainTreeIntregration 12)
addCstoneMpiTest(exchange_general.cpp exchange_general GeneralFocusExchange 12)
addCstoneMpiTest(exchange_domain.cpp exchange_domain GlobalDomainExchange 12)
addCstoneMpiTest(focus_tree.cpp focus_tree FocusTreeIntregration 12)
addCstoneMpiTest(domain_nranks.cpp domain_nranks GlobalDomainNRanks 12)

# workaround for compiling MPI with nvcc on cray systems
if (CSTONE_WITH_CUDA AND MPI_CXX_INCLUDE_PATH STREQUAL "" AND DEFINED ENV{CRAY_MPICH2_DIR} OR DEFINED ENV{CRAY_MPICH_DIR})
    message(STATUS "Applying MPI include workaround on CRAY")
    set(MPI_CXX_INCLUDE_PATH $ENV{MPICH_DIR}/include)
endif ()

function(addCstoneGpuMpiTest source exename testname ranks)
    addCstoneMpiTest(${source} ${exename} ${testname} ${ranks})
    target_link_libraries(${exename} PRIVATE cstone_gpu)
    if (CSTONE_WITH_CUDA)
        target_link_libraries(${exename} PRIVATE CUDA::cudart)
    endif ()
    if (CSTONE_WITH_HIP)
        target_link_libraries(${exename} PRIVATE hip::host)
        target_compile_definitions(${exename} PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
        set_target_properties(${exename} PROPERTIES LINKER_LANGUAGE CXX)
    endif ()
endfunction()

if (CSTONE_WITH_CUDA OR CSTONE_WITH_HIP)
    addCstoneGpuMpiTest(exchange_general_gpu.cpp exchange_general_gpu ExchangeGeneralGpu 5)
    addCstoneGpuMpiTest(exchange_halos_gpu.cpp exchange_halos_gpu GlobalHaloExchangeGpu 2)
    addCstoneGpuMpiTest(exchange_domain_gpu.cpp exchange_domain_gpu GlobalDomainExchangeGpu 2)
    addCstoneGpuMpiTest(assignment_gpu.cpp assignment_gpu AssignmentGpu 2)
    addCstoneGpuMpiTest(domain_gpu.cpp domain_gpu DomainGpu 2)
    addCstoneGpuMpiTest(exchange_ptype.cpp exchange_ptype_gpu GlobalPtypeExchangeGpu 2)
    target_compile_definitions(exchange_ptype_gpu PRIVATE SPH_EXA_PTYPE_GPU_TEST)
    addCstoneGpuMpiTest(fluid_halos.cpp fluid_halos_gpu GlobalFluidHalosGpu 2)
    target_compile_definitions(fluid_halos_gpu PRIVATE SPH_EXA_PTYPE_GPU_TEST)
    addCstoneGpuMpiTest(zero_fluid_rank.cpp zero_fluid_rank_gpu GlobalZeroFluidRankGpu 2)
    target_compile_definitions(zero_fluid_rank_gpu PRIVATE SPH_EXA_PTYPE_GPU_TEST)
    # sph_gpu/ryoanji only exist when the sphexa root project enables CUDA/HIP; CSTONE_WITH_CUDA can
    # be ON here without them (standalone cstone GPU build), which would fail at link time
    if (DEFINED SPH_EXA_WITH_CUDA AND (SPH_EXA_WITH_CUDA OR SPH_EXA_WITH_HIP)
        AND EXISTS "${CMAKE_SOURCE_DIR}/main/src/propagator/ve_hydro_pt.hpp")
        addCstoneGpuMpiTest(unconverged_removal.cpp unconverged_removal_gpu GlobalUnconvergedRemovalGpu 2)
        target_compile_definitions(unconverged_removal_gpu PRIVATE SPH_EXA_PTYPE_GPU_TEST)
        target_include_directories(unconverged_removal_gpu PRIVATE ${CMAKE_SOURCE_DIR}/main/src
                                  ${CMAKE_SOURCE_DIR}/sph/include ${CMAKE_SOURCE_DIR}/ryoanji/src)
        target_link_libraries(unconverged_removal_gpu PRIVATE sph_gpu ryoanji)
    else ()
        message(STATUS "Skipping GlobalUnconvergedRemovalGpu: not building inside the sphexa tree")
    endif ()
endif ()

