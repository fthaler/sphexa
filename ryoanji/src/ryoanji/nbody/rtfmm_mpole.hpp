/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief implements elementary gravity data structures for octree nodes
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#ifdef __CUDACC__
#include <cublas_v2.h>

#include "cstone/cuda/errorcheck.cuh"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/primitives/warpscan.cuh"
#endif

#include "kernel.hpp"
#include "cstone/sfc/common.hpp"

namespace ryoanji
{

template<class Tc, class T, unsigned S>
struct GlobalData
{
    Tc surfacePointsX[S], surfacePointsY[S], surfacePointsZ[S];
    T  UT[S * S];
    T  vSinv[S * S];
    T  vSinvUT[S * S];
    T  m2m[8][S * S];
    T  r0;
};

extern void* globalData;
#ifdef __CUDACC__
extern __device__ void* globalDataDevice;
#endif

HOST_DEVICE_FUN constexpr unsigned rtfmmSurfacePoints(unsigned p) { return 6u * (p - 1) * (p - 1) + 2u; }

template<class T, std::size_t S>
using RtfmmMultipole = util::array<T, S>;

template<int stride, class T1, class T2, class T3, unsigned S>
HOST_DEVICE_FUN void P2M_add(const T1* x, const T1* y, const T1* z, const T2* m, LocalIndex begin, LocalIndex end,
                             unsigned level, const Vec4<T1>& center, const Vec3<T1>& geoCenter,
                             RtfmmMultipole<T3, S>& gv)
{
#ifdef __CUDA_ARCH__
    const GlobalData<T1, T2, S>* data = reinterpret_cast<const GlobalData<T1, T2, S>*>(globalDataDevice);
#else
    const GlobalData<T1, T2, S>* data = reinterpret_cast<const GlobalData<T1, T2, S>*>(globalData);
#endif

    T2 scale  = T2(1) / (1 << level);
    T2 scaleR = T2(2.95) / (1 << level) * data->r0;

    for (int j = 0; j < S; ++j)
    {
        T1 xj = data->surfacePointsX[j] * scaleR + geoCenter[0];
        T1 yj = data->surfacePointsY[j] * scaleR + geoCenter[1];
        T1 zj = data->surfacePointsZ[j] * scaleR + geoCenter[2];

        T2 p = 0;
        for (LocalIndex i = begin; i < end; i += stride)
        {
            T1 xi = x[i];
            T1 yi = y[i];
            T1 zi = z[i];
            T2 qi = m[i];

            T1 dx = xj - xi;
            T1 dy = yj - yi;
            T1 dz = zj - zi;
            T1 r2 = dx * dx + dy * dy + dz * dz;
            // TODO: rsqrt?
            p += qi / std::sqrt(r2);
        }
        gv[j] += p * scale;
    }
}

template<class T, unsigned S>
HOST_DEVICE_FUN RtfmmMultipole<T, S> P2M_finalize(RtfmmMultipole<T, S> gv)
{
#ifdef __CUDA_ARCH__
    const GlobalData<T, T, S>* data = reinterpret_cast<const GlobalData<T, T, S>*>(globalDataDevice);
#else
    const GlobalData<T, T, S>* data = reinterpret_cast<const GlobalData<T, T, S>*>(globalData);
#endif

    RtfmmMultipole<T, S> tmp;
    for (unsigned i = 0; i < S; ++i)
    {
        T tmpi = 0;
        for (unsigned j = 0; j < S; ++j)
            tmpi += data->UT[i * S + j] * gv[j];
        tmp[i] = tmpi;
    }
    for (unsigned i = 0; i < S; ++i)
    {
        T gvi = 0;
        for (unsigned j = 0; j < S; ++j)
            gvi += data->vSinv[i * S + j] * tmp[j];
        gv[i] = gvi;
    }

    return gv;
}

template<int stride = 1, class T1, class T2, class T3, unsigned S>
HOST_DEVICE_FUN void P2M(const T1* x, const T1* y, const T1* z, const T2* m, LocalIndex begin, LocalIndex end,
                         unsigned level, const Vec4<T1>& center, const Vec3<T1>& geoCenter, RtfmmMultipole<T3, S>& gv)
{
    gv = T3(0);
    P2M_add<stride, T1, T2, T3, S>(x, y, z, m, begin, end, level, center, geoCenter, gv);
    gv = P2M_finalize<T3, S>(gv);
}

template<class Ta, class Tc, class Tmp, unsigned S>
HOST_DEVICE_FUN DEVICE_INLINE Vec4<Ta> M2P(Vec4<Ta> acc, const Vec3<Tc>& target, const Vec3<Tc>& center,
                                           const RtfmmMultipole<Tmp, S>& multipole)
{
    assert(false && "Currently not doing anything useful");
    return {};
}

template<class T, unsigned S, class Tc>
HOST_DEVICE_FUN void addMultipole(RtfmmMultipole<T, S>& composite, const Vec3<Tc>& dX, const Vec3<Tc>& geoDX,
                                  const RtfmmMultipole<T, S>& addend)
{
#ifdef __CUDA_ARCH__
    const GlobalData<Tc, T, S>* data = reinterpret_cast<const GlobalData<Tc, T, S>*>(globalDataDevice);
#else
    const GlobalData<Tc, T, S>* data = reinterpret_cast<const GlobalData<Tc, T, S>*>(globalData);
#endif

    const unsigned octant = unsigned(geoDX[2] < 0) | (unsigned(geoDX[1] < 0) << 1) | (unsigned(geoDX[0] < 0) << 2);
    const T*       m2m    = data->m2m[octant];

    for (unsigned i = 0; i < S; ++i)
    {
        T ci = 0;
        for (unsigned j = 0; j < S; ++j)
            ci += m2m[i * S + j] * addend[j];
        composite[i] += ci;
    }
}

template<class T, unsigned S, class Tm>
HOST_DEVICE_FUN void M2M(int begin, int end, const Vec4<T>& Xout, const Vec4<T>* Xsrc, const Vec3<T>& geoXout,
                         const Vec3<T>* geoXsrc, const RtfmmMultipole<Tm, S>* Msrc, RtfmmMultipole<Tm, S>& Mout)
{
    constexpr T               NaN          = std::numeric_limits<T>::signaling_NaN();
    constexpr cstone::Vec4<T> dummyXsrc    = {NaN, NaN, NaN, NaN};
    constexpr cstone::Vec3<T> dummyGeoXsrc = {NaN, NaN, NaN};

    Mout = 0;
    for (int i = begin; i < end; i++)
    {
        const RtfmmMultipole<Tm, S>& Mi    = Msrc[i];
        Vec4<T>                      Xi    = Xsrc ? Xsrc[i] : dummyXsrc;
        Vec3<T>                      geoXi = geoXsrc ? geoXsrc[i] : dummyGeoXsrc;
        Vec3<T>                      dX    = makeVec3(Xout - Xi);
        Vec3<T>                      geoDX = geoXout - geoXi;
        addMultipole<Tm, S, T>(Mout, dX, geoDX, Mi);
    }
}

#define INSTANTIATE_RTFMM_MULTIPOLE(S)                                                                                 \
    template<int stride, class T1, class T2, class T3>                                                                 \
    HOST_DEVICE_FUN void P2M_add(const T1* x, const T1* y, const T1* z, const T2* m, LocalIndex begin, LocalIndex end, \
                                 unsigned level, const Vec4<T1>& center, const Vec3<T1>& geoCenter,                    \
                                 RtfmmMultipole<T3, S>& gv)                                                            \
    {                                                                                                                  \
        return P2M_add<stride, T1, T2, T3, S>(x, y, z, m, begin, end, level, center, geoCenter, gv);                   \
    }                                                                                                                  \
    template<class T>                                                                                                  \
    HOST_DEVICE_FUN RtfmmMultipole<T, S> P2M_finalize(RtfmmMultipole<T, S> gv)                                         \
    {                                                                                                                  \
        return P2M_finalize<T, S>(gv);                                                                                 \
    }                                                                                                                  \
    template<int stride = 1, class T1, class T2, class T3>                                                             \
    HOST_DEVICE_FUN void P2M(const T1* x, const T1* y, const T1* z, const T2* m, LocalIndex begin, LocalIndex end,     \
                             unsigned level, const Vec4<T1>& center, const Vec3<T1>& geoCenter,                        \
                             RtfmmMultipole<T3, S>& gv)                                                                \
    {                                                                                                                  \
        return P2M<stride, T1, T2, T3, S>(x, y, z, m, begin, end, level, center, geoCenter, gv);                       \
    }                                                                                                                  \
    template<class Ta, class Tc, class Tmp>                                                                            \
    HOST_DEVICE_FUN DEVICE_INLINE Vec4<Ta> M2P(Vec4<Ta> acc, const Vec3<Tc>& target, const Vec3<Tc>& center,           \
                                               const RtfmmMultipole<Tmp, S>& multipole)                                \
    {                                                                                                                  \
        return M2P<Ta, Tc, Tmp, S>(acc, target, center, multipole);                                                    \
    }                                                                                                                  \
    template<class T, class Tc>                                                                                        \
    HOST_DEVICE_FUN void addQuadrupole(RtfmmMultipole<T, S>& composite, const Vec3<Tc>& dX, const Vec3<Tc>& geoDX,     \
                                       const RtfmmMultipole<T, S>& addend)                                             \
    {                                                                                                                  \
        return addQuadrupole<T, S, Tc>(composite, dX, geoDX, addend);                                                  \
    }                                                                                                                  \
    template<class T, class Tm>                                                                                        \
    HOST_DEVICE_FUN void M2M(int begin, int end, const Vec4<T>& Xout, const Vec4<T>* Xsrc, const Vec3<T>& geoXout,     \
                             const Vec3<T>* geoXsrc, const RtfmmMultipole<Tm, S>* Msrc, RtfmmMultipole<Tm, S>& Mout)   \
    {                                                                                                                  \
        M2M<T, S, Tm>(begin, end, Xout, Xsrc, geoXout, geoXsrc, Msrc, Mout);                                           \
    }

INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(4))
INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(5))

template<class Tc, class T, unsigned P>
void rtfmmInit(Tc r0);

void rtfmmFinalize();

#ifdef __CUDACC__
template<unsigned S, class T1, class T2, class KeyType>
__global__ void rtfmmP2mKernel(const T1* x, const T1* y, const T1* z, const T2* q, const TreeNodeIndex* leafToInternal,
                               const KeyType* leaves, TreeNodeIndex numLeaves, const LocalIndex* layout,
                               const cstone::Vec3<T1>* geoCenters, T2* qEquivAllDevice)
{
    const GlobalData<T1, T2, S>* data = reinterpret_cast<const GlobalData<T1, T2, S>*>(globalDataDevice);

    const int bidx  = blockIdx.x;
    const int tidx  = threadIdx.x;
    const int thNum = blockDim.x;

    const auto level       = cstone::treeLevel(leaves[bidx + 1] - leaves[bidx]);
    const auto internalIdx = leafToInternal[bidx];
    const T2   scale       = T2(1) / (1 << level);
    const T2   scaleR      = T2(2.95) / (1 << level) * data->r0;
    const auto center      = geoCenters[internalIdx];

    const int offset = layout[bidx];
    const int count  = layout[bidx + 1] - offset;

    for (int j = tidx; j < S; j += thNum)
    {
        T1 xj = data->surfacePointsX[j] * scaleR + center[0];
        T1 yj = data->surfacePointsY[j] * scaleR + center[1];
        T1 zj = data->surfacePointsZ[j] * scaleR + center[2];

        T2 p = 0.0f;
        for (int i = 0; i < count; i++)
        {
            T1 xi = x[offset + i];
            T1 yi = y[offset + i];
            T1 zi = z[offset + i];
            T2 qi = q[offset + i];

            T1 dx = xj - xi;
            T1 dy = yj - yi;
            T1 dz = zj - zi;
            T2 r2 = dx * dx + dy * dy + dz * dz;
            p += qi / std::sqrt(r2);
        }
        qEquivAllDevice[bidx * S + j] = p * scale;
    }
}

inline void checkCublas(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("CUBLAS error");
}

template<unsigned S, class T1, class T2, class KeyType, class Vector>
void rtfmmP2M(const T1* x, const T1* y, const T1* z, const T2* m, const TreeNodeIndex* leafToInternal,
              const KeyType* leaves, TreeNodeIndex numLeaves, const LocalIndex* layout, const Vec3<T1>* geoCenters,
              RtfmmMultipole<T2, S>* multipoles, cublasHandle_t handle, Vector& scratchBuffer)
{
    reallocateBytes(scratchBuffer, S * numLeaves * sizeof(T2), 1.1);
    T2* qEquivAllDevice = reinterpret_cast<T2*>(scratchBuffer.data());

    const int blockNum  = numLeaves;
    const int blockSize = std::min(S, 1024u);

    // TODO: CUDA stream
    rtfmmP2mKernel<S>
        <<<blockNum, blockSize>>>(x, y, z, m, leafToInternal, leaves, numLeaves, layout, geoCenters, qEquivAllDevice);

    GlobalData<T1, T2, S>* data;
    checkGpuErrors(cudaMemcpyFromSymbol(&data, globalDataDevice, sizeof(void*)));

    T2   alpha = 1;
    T2   beta  = 0;
    auto gemm  = []<class... Args>(Args&&... args)
    {
        if constexpr (std::is_same_v<T2, double>)
            return cublasDgemm(std::forward<Args>(args)...);
        else
            return cublasSgemm(std::forward<Args>(args)...);
    };
    checkCublas(gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, S, numLeaves, S, &alpha, data->UT, S, qEquivAllDevice, S, &beta,
                     qEquivAllDevice, S));
    checkCublas(gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, S, numLeaves, S, &alpha, data->vSinv, S, qEquivAllDevice, S,
                     &beta, qEquivAllDevice, S));

    cstone::scatterGpu(leafToInternal, numLeaves, reinterpret_cast<const RtfmmMultipole<T2, S>*>(qEquivAllDevice),
                       multipoles);
}

template<unsigned S, class T1, class T2>
__global__ void rtfmmSetupM2mPointers(TreeNodeIndex numNodes, const TreeNodeIndex* childOffsets,
                                      const Vec3<T1>* geoCenters, RtfmmMultipole<T2, S>* multipoles,
                                      RtfmmMultipole<T2, S>* tmp, T2** m2mPtrs, T2** multipolePtrs, T2** tmpPtrs)
{
    const GlobalData<T1, T2, S>* data = reinterpret_cast<const GlobalData<T1, T2, S>*>(globalDataDevice);

    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    multipolePtrs[i] = reinterpret_cast<T2*>(&multipoles[i]);
    tmpPtrs[i]       = reinterpret_cast<T2*>(&tmp[i]);

    TreeNodeIndex firstChild = childOffsets[i];
    if (firstChild != 0)
    {
        for (int c = 0; c < 8; ++c)
        {
            TreeNodeIndex child = firstChild + c;
            auto          geoDX = geoCenters[i] - geoCenters[child];
            unsigned octant = unsigned(geoDX[2] < 0) | (unsigned(geoDX[1] < 0) << 1) | (unsigned(geoDX[0] < 0) << 2);
            m2mPtrs[child]  = const_cast<T2*>(data->m2m[octant]);
        }
    }
}

template<unsigned S, class T2>
__global__ void rtfmmM2mReduction(TreeNodeIndex firstParent, TreeNodeIndex lastParent,
                                  const TreeNodeIndex* childOffsets, const RtfmmMultipole<T2, S>* tmp,
                                  RtfmmMultipole<T2, S>* multipoles)
{
    const int bidx  = blockIdx.x;
    const int tidx  = threadIdx.x;
    const int thNum = blockDim.x;

    const TreeNodeIndex parent     = bidx + firstParent;
    const TreeNodeIndex firstChild = childOffsets[parent];
    if (!firstChild) return;

    auto& output = multipoles[parent];

    for (int i = tidx; i < S; i += thNum)
    {
        T2 sum = 0;
        for (int j = 0; j < 8; j++)
        {
            auto y = tmp[firstChild + j][i];
            sum += y;
        }
        output[i] = sum;
    }
}

template<unsigned S, class T1, class T2, class... Vectors>
void rtfmmM2M(std::span<const TreeNodeIndex> levelRange, const TreeNodeIndex* childOffsets,
              const TreeNodeIndex numNodes, const Vec3<T1>* geoCenters, RtfmmMultipole<T2, S>* multipoles,
              cublasHandle_t handle, std::tuple<Vectors&...> scratchBuffers)
{
    constexpr double growthRate = 1.1;
    static_assert(sizeof...(Vectors) >= 4);

    reallocateBytes(std::get<0>(scratchBuffers), numNodes * sizeof(T2*), growthRate);
    reallocateBytes(std::get<1>(scratchBuffers), numNodes * sizeof(T2*), growthRate);
    reallocateBytes(std::get<2>(scratchBuffers), numNodes * sizeof(T2*), growthRate);
    T2** m2mPtrs       = reinterpret_cast<T2**>(std::get<0>(scratchBuffers).data());
    T2** multipolePtrs = reinterpret_cast<T2**>(std::get<1>(scratchBuffers).data());
    T2** tmpPtrs       = reinterpret_cast<T2**>(std::get<2>(scratchBuffers).data());

    reallocateBytes(std::get<3>(scratchBuffers), numNodes * sizeof(RtfmmMultipole<T2, S>), growthRate);
    RtfmmMultipole<T2, S>* tmp = reinterpret_cast<RtfmmMultipole<T2, S>*>(std::get<3>(scratchBuffers).data());

    {
        constexpr int blockSize = 256;
        int           numBlocks = (numNodes + blockSize - 1) / blockSize;
        rtfmmSetupM2mPointers<S><<<numBlocks, blockSize>>>(numNodes, childOffsets, geoCenters, multipoles, tmp, m2mPtrs,
                                                           multipolePtrs, tmpPtrs);
    }

    T2   alpha       = 1;
    T2   beta        = 0;
    auto gemvBatched = []<class... Args>(Args&&... args)
    {
        if constexpr (std::is_same_v<T2, double>)
            return cublasDgemvBatched(std::forward<Args>(args)...);
        else
            return cublasSgemvBatched(std::forward<Args>(args)...);
    };

    const int numLevels = levelRange.size() - 1;
    for (int level = numLevels - 1; level > 0; level--)
    {
        int batchCount = levelRange[level + 1] - levelRange[level];
        if (batchCount)
        {
            int offset = levelRange[level];

            checkCublas(gemvBatched(handle, CUBLAS_OP_T, S, S, &alpha, m2mPtrs + offset, S, multipolePtrs + offset, 1,
                                    &beta, tmpPtrs + offset, 1, batchCount));

            int firstParent = levelRange[level - 1];
            int lastParent  = levelRange[level];
            int numParents  = lastParent - firstParent;
            if (numParents)
            {
                constexpr int blockSize = std::min(S, 1024u);
                int           numBlocks = numParents;
                rtfmmM2mReduction<S><<<numBlocks, blockSize>>>(firstParent, lastParent, childOffsets, tmp, multipoles);
            }
        }
    }
}
#endif

} // namespace ryoanji
