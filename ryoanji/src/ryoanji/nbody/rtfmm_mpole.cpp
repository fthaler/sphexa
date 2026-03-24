#include "rtfmm_mpole.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

#include <cblas.h>
#include <lapacke.h>

#include "cstone/tree/definitions.h"
#ifdef __CUDACC__
#include "cstone/cuda/cuda_utils.cuh"
#endif

namespace ryoanji
{

void*            globalData;
#ifdef __CUDACC__
__device__ void* globalDataDevice;
#endif

template<class T>
std::vector<cstone::Vec3<T>> getSurfacePoints(unsigned p, T r = 1, cstone::Vec3<T> x = {0, 0, 0}, int dir = 0)
{
    const unsigned num = rtfmmSurfacePoints(p);

    std::vector<cstone::Vec3<T>> points;
    if (dir == 0)
    {
        for (uint32_t i = 0; i < p; i++)
        {
            for (uint32_t j = 0; j < p; j++)
            {
                for (uint32_t k = 0; k < p; k++)
                {
                    if (i == 0 || i == p - 1 || j == 0 || j == p - 1 || k == 0 || k == p - 1)
                    {
                        points.push_back(
                            {-1.0 + i * 2.0 / (p - 1), -1.0 + j * 2.0 / (p - 1), -1.0 + k * 2.0 / (p - 1)});
                    }
                }
            }
        }
    }
    else
    {
        for (uint32_t k = 0; k < p; k++)
        {
            for (uint32_t j = 0; j < p; j++)
            {
                for (uint32_t i = 0; i < p; i++)
                {
                    if (i == 0 || i == p - 1 || j == 0 || j == p - 1 || k == 0 || k == p - 1)
                    {
                        points.push_back(
                            {-1.0 + i * 2.0 / (p - 1), -1.0 + j * 2.0 / (p - 1), -1.0 + k * 2.0 / (p - 1)});
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < points.size(); i++)
    {
        auto& p = points[i];
        p       = p * r + x;
    }
    assert(points.size() == num);

    return points;
}

template<class Tc, class T, unsigned P, unsigned S = rtfmmSurfacePoints(P)>
void initSurfacePoints(GlobalData<Tc, T, S>* globalData)
{
    auto            surfacePoints = getSurfacePoints<Tc>(P);
    std::vector<Tc> x(S), y(S), z(S);
    for (unsigned s = 0; s < S; ++s)
    {
        globalData->surfacePointsX[s] = surfacePoints[s][0];
        globalData->surfacePointsY[s] = surfacePoints[s][1];
        globalData->surfacePointsZ[s] = surfacePoints[s][2];
    }
}

template<class Tc>
std::tuple<std::vector<Tc>, int, int> getP2pMatrix(const std::vector<cstone::Vec3<Tc>>& xSrc,
                                                   const std::vector<cstone::Vec3<Tc>>& xTar)
{
    int             numSrc = xSrc.size();
    int             numTar = xTar.size();
    std::vector<Tc> matrixP2p(numTar * numSrc);
    for (int j = 0; j < numTar; j++)
    {
        auto xtar = xTar[j];
        for (int i = 0; i < numSrc; i++)
        {
            auto xsrc                 = xSrc[i];
            auto dx                   = xtar - xsrc;
            auto r                    = std::sqrt(util::norm2(dx));
            Tc   invr                 = r == 0 ? 0 : 1 / r;
            matrixP2p[j * numSrc + i] = invr;
        }
    }
    return {std::move(matrixP2p), numTar, numSrc};
}

template<class Tc>
void svd(int m, int n, std::vector<Tc> A, std::vector<Tc>& U, std::vector<Tc>& S, std::vector<Tc>& VT)
{
    int k = std::min(m, n);

    U  = std::vector<Tc>(m * m);
    VT = std::vector<Tc>(n * n);
    std::vector<Tc> s(k);

    std::vector<Tc> wbuff(k);

    if constexpr (std::is_same_v<Tc, double>)
    {
        LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, A.data(), n, s.data(), U.data(), m, VT.data(), n,
                       wbuff.data());
    }
    else
    {
        LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, A.data(), n, s.data(), U.data(), m, VT.data(), n,
                       wbuff.data());
    }

    S = std::vector<Tc>(m * n);

    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < n; i++)
        {
            if (i == j && j < k)
                S[j * n + i] = s[j];
            else
                S[j * n + i] = 0;
        }
    }
}

template<class T>
std::vector<T> transpose(int m, int n, const std::vector<T>& A)
{
    assert(A.size() == m * n);
    std::vector<T> res(n * m);
    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < n; i++)
        {
            res[i * m + j] = A[j * n + i];
        }
    }
    return res;
}

template<class T>
std::vector<T> pseudoInverse(int m, int n, std::vector<T> S)
{
    assert(S.size() == m * n);
    int         k    = std::min(m, n);
    T           sMax = 0;
    constexpr T EPS  = std::is_same_v<T, double> ? 4e-16 : 4e-8;
    for (int i = 0; i < k; i++)
    {
        sMax = std::max(sMax, std::abs(S[i * n + i]));
    }
    for (int i = 0; i < k; i++)
    {
        S[i * n + i] = S[i * n + i] > sMax * EPS ? 1.0 / S[i * n + i] : 0.0;
    }
    return transpose(m, n, S);
}

template<class T>
std::vector<T> matMatMul(int m, int k, int n, const std::vector<T>& A, const std::vector<T>& B)
{
    assert(A.size() == m * k);
    assert(B.size() == k * n);
    std::vector<T> C(m * n);
    if constexpr (std::is_same_v<T, double>)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A.data(), k, B.data(), n, 0.0, C.data(),
                    n);
    }
    else
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A.data(), k, B.data(), n, 0.0, C.data(),
                    n);
    }
    return C;
}

template<class T>
cstone::Vec3<T> getChildCellX(const cstone::Vec3<T>& xPar, T rPar, int octant, bool isPeriodic)
{
    cstone::Vec3<T> x;
    if (!isPeriodic)
    {
        assert(octant >= 0 && octant <= 7 && "octant out of range");
        if (octant == 0)
            x = xPar + cstone::Vec3<T>{-rPar / 2, -rPar / 2, -rPar / 2};
        else if (octant == 1)
            x = xPar + cstone::Vec3<T>{-rPar / 2, -rPar / 2, rPar / 2};
        else if (octant == 2)
            x = xPar + cstone::Vec3<T>{-rPar / 2, rPar / 2, -rPar / 2};
        else if (octant == 3)
            x = xPar + cstone::Vec3<T>{-rPar / 2, rPar / 2, rPar / 2};
        else if (octant == 4)
            x = xPar + cstone::Vec3<T>{rPar / 2, -rPar / 2, -rPar / 2};
        else if (octant == 5)
            x = xPar + cstone::Vec3<T>{rPar / 2, -rPar / 2, rPar / 2};
        else if (octant == 6)
            x = xPar + cstone::Vec3<T>{rPar / 2, rPar / 2, -rPar / 2};
        else if (octant == 7)
            x = xPar + cstone::Vec3<T>{rPar / 2, rPar / 2, rPar / 2};
    }
    else
    {
        assert(octant >= 0 && octant <= 26 && "periodic octant out of range");
        int k = octant / 9 - 1;
        int j = (octant % 9) / 3 - 1;
        int i = (octant % 3) - 1;
        x     = xPar + cstone::Vec3<T>{T(i), T(j), T(k)} * (rPar * 2 / 3.0);
    }
    return x;
}

template<class Tc, class T, unsigned P, unsigned S = rtfmmSurfacePoints(P)>
void initMatrices(GlobalData<Tc, T, S>* globalData, Tc r0)
{
    /* P2M */
    std::vector<cstone::Vec3<Tc>> xCheckUp = getSurfacePoints(P, r0 * 2.95);
    std::vector<cstone::Vec3<Tc>> xEquivUp = getSurfacePoints(P, r0 * 1.05);
    auto [e2cUpPrecompute, m, n]           = getP2pMatrix(xEquivUp, xCheckUp);
    std::vector<Tc> u, s, vT;
    svd(m, n, e2cUpPrecompute, u, s, vT);
    auto utP2mPrecompute    = transpose(m, m, u);
    auto vP2mPrecompute     = transpose(n, n, vT);
    auto sinvP2mPrecompute  = pseudoInverse(m, n, s);
    auto vsinvP2mPrecompute = matMatMul(n, n, m, vP2mPrecompute, sinvP2mPrecompute);

    assert(S == m);
    assert(S == n);
    std::copy_n(utP2mPrecompute.data(), S * S, globalData->uT);
    std::copy_n(vsinvP2mPrecompute.data(), S * S, globalData->vSinv);

    /* M2M */
    const auto& xCheckUpParent     = xCheckUp;
    const auto& utM2mPrecompute    = utP2mPrecompute;
    const auto& vsinvM2mPrecompute = vsinvP2mPrecompute;

    for (int octant = 0; octant < 8; octant++)
    {
        cstone::Vec3<Tc>              offsetChild   = getChildCellX({0, 0, 0}, r0, octant, false);
        std::vector<cstone::Vec3<Tc>> xEquivChildUp = getSurfacePoints(P, r0 / 2 * 1.05, offsetChild);
        auto [ce2pcUpPrecompute, mm, nn]            = getP2pMatrix(xEquivChildUp, xCheckUpParent);
        assert(S == mm);
        assert(S == nn);
        auto m1  = matMatMul(m, m, nn, utM2mPrecompute, ce2pcUpPrecompute);
        auto m2m = matMatMul(n, m, nn, vsinvM2mPrecompute, m1);
        std::copy_n(m2m.data(), S * S, globalData->m2m[octant]);
    }
}

template<class Tc, class T, unsigned P>
void rtfmmInit(Tc r0)
{
    constexpr auto S                  = rtfmmSurfacePoints(P);
    globalData                        = new GlobalData<Tc, T, S>();
    GlobalData<Tc, T, S>* globalDataT = reinterpret_cast<GlobalData<Tc, T, S>*>(globalData);

    globalDataT->r0 = r0;
    initSurfacePoints<Tc, T, P>(globalDataT);
    initMatrices<Tc, T, P>(globalDataT, r0);

#ifdef __CUDACC__
    void* devicePtr;
    checkGpuErrors(cudaMalloc(&devicePtr, sizeof(GlobalData<Tc, T, S>)));
    checkGpuErrors(cudaMemcpy(devicePtr, globalData, sizeof(GlobalData<Tc, T, S>), cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpyToSymbol(globalDataDevice, &devicePtr, sizeof(void*)));
#endif
}

template void rtfmmInit<double, double, 5>(double);

} // namespace ryoanji
