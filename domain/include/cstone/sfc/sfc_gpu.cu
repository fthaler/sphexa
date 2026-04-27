/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  SFC encoding/decoding in 32- and 64-bit on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/errorcheck.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/sfc/sfc_gpu.h"

namespace cstone
{

template<class KeyType, class T>
__global__ void
computeSfcKeysKernel(KeyType* keys, const T* x, const T* y, const T* z, size_t numKeys, const Box<T> box)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys)
    {
        KeyType rmKey = removeKey<KeyType>{};
        if (keys[tid] != rmKey) { keys[tid] = sfc3D<KeyType>(x[tid], y[tid], z[tid], box); }
    }
}

template<class KeyType, class T>
void computeSfcKeysGpu(const T* x, const T* y, const T* z, KeyType* keys, size_t numKeys, const Box<T>& box)
{
    if (numKeys == 0) { return; }

    constexpr int threadsPerBlock = 256;
    computeSfcKeysKernel<<<iceil(numKeys, threadsPerBlock), threadsPerBlock>>>(keys, x, y, z, numKeys, box);
    checkGpuErrors(cudaGetLastError());
}

template void
computeSfcKeysGpu(const float*, const float*, const float*, MortonKey<unsigned>*, size_t, const Box<float>&);
template void
computeSfcKeysGpu(const double*, const double*, const double*, MortonKey<unsigned>*, size_t, const Box<double>&);
template void
computeSfcKeysGpu(const float*, const float*, const float*, MortonKey<uint64_t>*, size_t, const Box<float>&);
template void
computeSfcKeysGpu(const double*, const double*, const double*, MortonKey<uint64_t>*, size_t, const Box<double>&);

template void
computeSfcKeysGpu(const float*, const float*, const float*, HilbertKey<unsigned>*, size_t, const Box<float>&);
template void
computeSfcKeysGpu(const double*, const double*, const double*, HilbertKey<unsigned>*, size_t, const Box<double>&);
template void
computeSfcKeysGpu(const float*, const float*, const float*, HilbertKey<uint64_t>*, size_t, const Box<float>&);
template void
computeSfcKeysGpu(const double*, const double*, const double*, HilbertKey<uint64_t>*, size_t, const Box<double>&);

template<class KeyType, class T>
__global__ void
computeSfcKeysClampKernel(KeyType* keys, const Vec3<T>* xyz, size_t numKeys, const Box<T> box, const IBox lbox)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys)
    {
        KeyType rmKey = removeKey<KeyType>{};
        if (keys[tid] != rmKey) { keys[tid] = sfc3D<KeyType>(xyz[tid][0], xyz[tid][1], xyz[tid][2], box, lbox); }
    }
}

template<class KeyType, class T>
void computeSfcKeysClampGpu(const Vec3<T>* xyz, KeyType* keys, size_t numKeys, const Box<T>& box, const IBox& lbox)
{
    if (numKeys == 0) { return; }

    constexpr int threadsPerBlock = 256;
    computeSfcKeysClampKernel<<<iceil(numKeys, threadsPerBlock), threadsPerBlock>>>(keys, xyz, numKeys, box, lbox);
    checkGpuErrors(cudaGetLastError());
}

#define COMPUTE_SFC_KEYS_CLAMP_GPU(T, KeyType)                                                                         \
    template void computeSfcKeysClampGpu(const Vec3<T>* xyz, KeyType* keys, size_t numKeys, const Box<T>& box,         \
                                         const IBox& lbox)
COMPUTE_SFC_KEYS_CLAMP_GPU(float, MortonKey<uint32_t>);
COMPUTE_SFC_KEYS_CLAMP_GPU(double, MortonKey<uint32_t>);
COMPUTE_SFC_KEYS_CLAMP_GPU(double, MortonKey<uint64_t>);
COMPUTE_SFC_KEYS_CLAMP_GPU(float, HilbertKey<uint32_t>);
COMPUTE_SFC_KEYS_CLAMP_GPU(double, HilbertKey<uint32_t>);
COMPUTE_SFC_KEYS_CLAMP_GPU(double, HilbertKey<uint64_t>);

} // namespace cstone
