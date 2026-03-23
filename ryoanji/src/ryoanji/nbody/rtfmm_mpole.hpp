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

#include "kernel.hpp"

namespace ryoanji
{

template<class Tc, class T, unsigned S>
struct GlobalData
{
    Tc surfacePointsX[S], surfacePointsY[S], surfacePointsZ[S];
    T  uT[S * S], vSinv[S * S];
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
                             const Vec4<T1>& center, RtfmmMultipole<T3, S>& gv)
{
    T2 scale  = 1; // TODO: 1 / 2^treeLevel
    T2 scaleR = 1; // TODO: ?

#ifdef __CUDA_ARCH__
    const GlobalData<T1, T2, S>* data = reinterpret_cast<const GlobalData<T1, T2, S>*>(globalDataDevice);
#else
    const GlobalData<T1, T2, S>* data = reinterpret_cast<const GlobalData<T1, T2, S>*>(globalData);
#endif

    for (int j = 0; j < S; ++j)
    {
        T1 xj = data->surfacePointsX[j] * scaleR + center[0];
        T1 yj = data->surfacePointsY[j] * scaleR + center[1];
        T1 zj = data->surfacePointsZ[j] * scaleR + center[2];

        T2 p = 0;
        for (LocalIndex i = begin; i < end; ++i)
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
            tmpi += data->uT[i * S + j] * gv[j];
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
                         const Vec4<T1>& center, RtfmmMultipole<T3, S>& gv)
{
    gv = T3(0);
    P2M_add<stride, T1, T2, T3, S>(x, y, z, m, begin, end, center, gv);
    gv = P2M_finalize<T3, S>(gv);
}

template<class Ta, class Tc, class Tmp, unsigned S>
HOST_DEVICE_FUN DEVICE_INLINE Vec4<Ta> M2P(Vec4<Ta> acc, const Vec3<Tc>& target, const Vec3<Tc>& center,
                                           const RtfmmMultipole<Tmp, S>& multipole)
{
    return {};
}

template<class T, unsigned S, class Tc>
HOST_DEVICE_FUN void addQuadrupole(RtfmmMultipole<T, S>& composite, Vec3<Tc> dX, const RtfmmMultipole<T, S>& addend)
{
}

#define INSTANTIATE_RTFMM_MULTIPOLE(S)                                                                                 \
    template<int stride, class T1, class T2, class T3>                                                                 \
    HOST_DEVICE_FUN void P2M_add(const T1* x, const T1* y, const T1* z, const T2* m, LocalIndex begin, LocalIndex end, \
                                 const Vec4<T1>& center, RtfmmMultipole<T3, S>& gv)                                    \
    {                                                                                                                  \
        return P2M_add<stride, T1, T2, T3, S>(x, y, z, m, begin, end, center, gv);                                     \
    }                                                                                                                  \
    template<class T>                                                                                                  \
    HOST_DEVICE_FUN RtfmmMultipole<T, S> P2M_finalize(RtfmmMultipole<T, S> gv)                                         \
    {                                                                                                                  \
        return P2M_finalize<T, S>(gv);                                                                                 \
    }                                                                                                                  \
    template<int stride = 1, class T1, class T2, class T3>                                                             \
    HOST_DEVICE_FUN void P2M(const T1* x, const T1* y, const T1* z, const T2* m, LocalIndex begin, LocalIndex end,     \
                             const Vec4<T1>& center, RtfmmMultipole<T3, S>& gv)                                        \
    {                                                                                                                  \
        return P2M<stride, T1, T2, T3, S>(x, y, z, m, begin, end, center, gv);                                         \
    }                                                                                                                  \
    template<class Ta, class Tc, class Tmp>                                                                            \
    HOST_DEVICE_FUN DEVICE_INLINE Vec4<Ta> M2P(Vec4<Ta> acc, const Vec3<Tc>& target, const Vec3<Tc>& center,           \
                                               const RtfmmMultipole<Tmp, S>& multipole)                                \
    {                                                                                                                  \
        return M2P<Ta, Tc, Tmp, S>(acc, target, center, multipole);                                                    \
    }                                                                                                                  \
    template<class T, class Tc>                                                                                        \
    HOST_DEVICE_FUN void addQuadrupole(RtfmmMultipole<T, S>& composite, Vec3<Tc> dX,                                   \
                                       const RtfmmMultipole<T, S>& addend)                                             \
    {                                                                                                                  \
        return addQuadrupole<T, S, Tc>(composite, dX, addend);                                                         \
    }

INSTANTIATE_RTFMM_MULTIPOLE(rtfmmSurfacePoints(5))

template<class Tc, class T, unsigned P>
void rtfmmInit(Tc r0);

void rtfmmFinalize();

template<class T1, class T2, class T3, class T4, unsigned S>
void rtfmmP2M(const T1* x, const T1* y, const T1* z, const T2* m, const T3* uT, const T3* vSinv,
              RtfmmMultipole<T4, S>* multipoles);

} // namespace ryoanji
