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
#endif

#include <span>

#include "kernel.hpp"
#include "cstone/sfc/common.hpp"

namespace ryoanji
{

/*! @brief Translation matrices for equivalence charges
 *
 * @tparam Tc   the xyz coordinate type, float or double
 * @tparam T    the multipole moment type, float or double
 * @tparam S    number of elements per multipole
 */
template<class Tc, class T, unsigned S>
struct GlobalData
{
    /*! @brief We want to align buffers to this size so we can load with float4 or double4_a16
     *
     * float 4 has to be 16-byte aligned, double4 is deprecated in favor of double4_a16 or double4_a32
     * We use 16-byte alignment, as that's the current maximum supported global load transaction size
     */
    static constexpr unsigned T4_alignment = 16;

#if __CUDACC_VER_MAJOR__ >= 13
    using Double4a16 = double4_16a;
#else
    using Double4a16 = double4;
#endif

    using T4 = std::conditional_t<std::is_same_v<T, float>, float4, Double4a16>;

    static constexpr unsigned T4_a_elem = T4_alignment / sizeof(T);
    //! @brief S padded to multiple of 16-byte size
    static constexpr unsigned S_padded  = (S + T4_a_elem - 1) / T4_a_elem * T4_a_elem;

    Tc                      surfacePointsX[S], surfacePointsY[S], surfacePointsZ[S];
    T                       UT[S * S];
    T                       vSinv[S * S];
    T                       vSinvUT[S * S];
    alignas(T4_alignment) T m2m[8][S * S_padded];
    T                       r0;
};

extern void* globalData;
#ifdef __CUDACC__
extern void* globalDataDevice_h;
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
            ci += m2m[i * data->S_padded + j] * addend[j];
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

INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(2))
INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(3))
INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(4))
INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(5))
INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(6))
INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(7))
INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(8))
INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(9))
INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(10))
INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(11))
INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(12))

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

//! @brief compute multipoles for the leaves @p leaves[0:numLeaves], not @p multipoles must be zeroed before!
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

    auto data = reinterpret_cast<GlobalData<T1, T2, S>*>(globalDataDevice_h);

    T2   alpha = 1;
    T2   beta  = 0;
    auto gemm  = []<class... Args>(Args&&... args)
    {
        if constexpr (std::is_same_v<T2, double>)
            return cublasDgemm(std::forward<Args>(args)...);
        else
            return cublasSgemm(std::forward<Args>(args)...);
    };
    auto* mp_t2 = reinterpret_cast<T2*>(multipoles);
    checkCublas(gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, S, numLeaves, S, &alpha, data->UT, S, qEquivAllDevice, S, &beta,
                     mp_t2, S));
    checkCublas(gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, S, numLeaves, S, &alpha, data->vSinv, S, mp_t2, S, &beta,
                     qEquivAllDevice, S));

    checkGpuErrors(cudaMemset(multipoles, 0, S * numLeaves * sizeof(T2)));
    cstone::scatterGpu(leafToInternal, numLeaves, reinterpret_cast<const RtfmmMultipole<T2, S>*>(qEquivAllDevice),
                       multipoles);
}

template<unsigned ThreadsPerRow, unsigned RowsPerBlock, unsigned S, class T1, class T2>
__global__ void rtfmmM2mKernel(TreeNodeIndex firstParent, TreeNodeIndex lastParent, const TreeNodeIndex* childOffsets,
                               const Vec3<T1>* geoCenters, RtfmmMultipole<T2, S>* multipoles)
{
    const TreeNodeIndex parent     = blockIdx.z + firstParent;
    const TreeNodeIndex firstChild = childOffsets[parent];
    if (!firstChild) return;

    const GlobalData<T1, T2, S>* data = reinterpret_cast<const GlobalData<T1, T2, S>*>(globalDataDevice);

    T2* m2m[8];
#pragma unroll
    for (int c = 0; c < 8; ++c)
    {
        TreeNodeIndex child  = firstChild + c;
        auto          geoDX  = geoCenters[parent] - geoCenters[child];
        unsigned      octant = unsigned(geoDX[2] < 0) | (unsigned(geoDX[1] < 0) << 1) | (unsigned(geoDX[0] < 0) << 2);
        m2m[c]               = const_cast<T2*>(data->m2m[octant]);
    }

    constexpr unsigned tileSize = 4 * ThreadsPerRow;
    const unsigned     row      = blockIdx.x * RowsPerBlock + threadIdx.y;
    if (row >= S) return;

    T2 accum = 0;
    for (unsigned k0 = 0; k0 + tileSize <= S; k0 += tileSize)
    {
        unsigned k = k0 + threadIdx.x * 4;

#pragma unroll
        for (int c = 0; c < 8; ++c)
        {
            using T2_4 = GlobalData<T1, T2, S>::T4;
            T2_4 a = *reinterpret_cast<T2_4*>(&m2m[c][row * data->S_padded + k]);
            T2_4 b;
            if constexpr ((sizeof(T2) * S) % GlobalData<T1, T2, S>::T4_alignment == 0)
            {
                b = *reinterpret_cast<T2_4*>(&multipoles[firstChild + c][k]);
            }
            else
            {
                b = {multipoles[firstChild + c][k], multipoles[firstChild + c][k + 1],
                     multipoles[firstChild + c][k + 2], multipoles[firstChild + c][k + 3]};
            }

            accum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }

    for (unsigned k = (S / tileSize) * tileSize + threadIdx.x; k < S; k += ThreadsPerRow)
    {
#pragma unroll
        for (int c = 0; c < 8; ++c)
            accum += m2m[c][row * data->S_padded + k] * multipoles[firstChild + c][k];
    }

#pragma unroll
    for (unsigned offset = ThreadsPerRow / 2; offset > 0; offset >>= 1)
        accum += __shfl_down_sync(0xffffffffu, accum, offset, ThreadsPerRow);

    if (threadIdx.x == 0) multipoles[parent][row] = accum;
}

template<unsigned S, class T1, class T2>
void rtfmmM2M(std::span<const TreeNodeIndex> levelRange, const TreeNodeIndex* childOffsets,
              const TreeNodeIndex numNodes, const Vec3<T1>* geoCenters, RtfmmMultipole<T2, S>* multipoles)
{
    const int numLevels = levelRange.size() - 1;
    for (int level = numLevels - 1; level > 0; level--)
    {
        int batchCount = levelRange[level + 1] - levelRange[level];
        if (batchCount)
        {
            unsigned firstParent = levelRange[level - 1];
            unsigned lastParent  = levelRange[level];
            unsigned numParents  = lastParent - firstParent;
            if (numParents)
            {
                constexpr unsigned threadsPerRow = 8;
                constexpr unsigned rowsPerBlock  = 16;
                constexpr dim3     blockSize     = {threadsPerRow, rowsPerBlock, 1};
                const dim3         numBlocks     = {(S + rowsPerBlock - 1) / rowsPerBlock, 1, numParents};
                rtfmmM2mKernel<threadsPerRow, rowsPerBlock, S>
                    <<<numBlocks, blockSize>>>(firstParent, lastParent, childOffsets, geoCenters, multipoles);
            }
        }
    }
}
#endif

} // namespace ryoanji
