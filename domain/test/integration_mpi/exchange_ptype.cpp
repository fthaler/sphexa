/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @brief 2-rank test syncing a domain and halo-exchanging a uint8_t particle field
 *
 * Exercises the domain exchangeHalos path with a uint8_t field (e.g. ptype). On the GPU build,
 * the send path requires the GATHER_RANGES(uint8_t, unsigned) instantiation to link.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <mpi.h>
#include <vector>

#ifdef SPH_EXA_PTYPE_GPU_TEST
#define USE_CUDA
#include "cstone/cuda/device_vector.h"
#include "cstone/cuda/stream_holder.cuh"
#endif

#include "cstone/domain/domain.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

//! @brief global particle set in SFC order, cf. FocusDomain.moreHalos
constexpr int numGlobal = 12;

template<class T>
void initGlobal(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, std::vector<T>& h,
                std::vector<uint8_t>& ptype)
{
    x     = {0.0, 0.11, 0.261, 0.281, 0.301, 0.321, 0.521, 0.541, 0.561, 0.761, 0.781, 1.000};
    y     = {0.0, 0.12, 0.262, 0.282, 0.302, 0.322, 0.522, 0.542, 0.562, 0.762, 0.781, 1.000};
    z     = {0.0, 0.13, 0.263, 0.283, 0.303, 0.323, 0.523, 0.543, 0.563, 0.763, 0.781, 1.000};
    h     = {0.1, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109, 0.110, 0.111};
    ptype = {1, 0, 3, 0, 0, 2, 7, 0, 5, 1, 0, 4};
}

//! @brief index of the particle with unique x coordinate @p x in the global set, or std::size_t(-1) if not found
template<class T>
std::size_t globalIndex(T x, const std::vector<T>& xGlobal)
{
    auto it = std::find(xGlobal.begin(), xGlobal.end(), x);
    return it == xGlobal.end() ? std::size_t(-1) : std::size_t(it - xGlobal.begin());
}

//! @brief check ptype per particle: buffer slots are SFC-ordered, so identify each particle by its
//!       (unique) x coordinate and compare against the true global ptype
template<class T>
void checkPtype(const std::vector<T>& x, const std::vector<uint8_t>& ptype,
                const std::vector<T>& xGlobal, const std::vector<uint8_t>& ptypeGlobal)
{
    ASSERT_EQ(x.size(), ptype.size());
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        std::size_t j = globalIndex(x[i], xGlobal);
        ASSERT_NE(j, std::size_t(-1)) << "particle x = " << x[i] << " not found in global set";
        EXPECT_EQ(ptype[i], ptypeGlobal[j]) << "buffer slot " << i << " (x = " << x[i] << ")";
    }
}

//! @brief fill the owned range [start:end) of @p ptype with the true values, matching particles by (unique) x
//!
//! Halo slots of @p ptype are left untouched.
template<class T>
void fillOwnedPtype(std::vector<uint8_t>& ptype, const std::vector<T>& x, LocalIndex start, LocalIndex end,
                    const std::vector<T>& xGlobal, const std::vector<uint8_t>& ptypeGlobal)
{
    for (LocalIndex i = start; i < end; ++i)
    {
        std::size_t j = globalIndex(x[i], xGlobal);
        ASSERT_NE(j, std::size_t(-1));
        ptype[i] = ptypeGlobal[j];
    }
}

template<class KeyType, class T>
void ptypeExchangeCpu(int rank, int numRanks)
{
    int bucketSize = 4;
    float theta    = 1.0;
    Domain<KeyType, T> domain(execution::cpu, rank, numRanks, bucketSize, bucketSize, theta, MPI_COMM_WORLD);

    std::vector<T> xGlobal, yGlobal, zGlobal, hGlobal;
    std::vector<uint8_t> ptypeGlobal;
    initGlobal(xGlobal, yGlobal, zGlobal, hGlobal, ptypeGlobal);

    // rank 0 gets particles with even index before the sync, rank 1 the odd ones
    std::vector<T> x, y, z, h;
    for (int i = rank; i < numGlobal; i += numRanks)
    {
        x.push_back(xGlobal[i]);
        y.push_back(yGlobal[i]);
        z.push_back(zGlobal[i]);
        h.push_back(hGlobal[i]);
    }

    std::vector<KeyType> keys(x.size());
    std::vector<T> s1, s2, s3;
    domain.sync(keys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    ASSERT_EQ(domain.nParticlesWithHalos(), x.size());

    // fill the owned range with the true values; halo slots hold garbage until the exchange
    std::vector<uint8_t> ptype(x.size(), uint8_t(-1));
    fillOwnedPtype(ptype, x, domain.startIndex(), domain.endIndex(), xGlobal, ptypeGlobal);

    domain.exchangeHalos(std::tie(ptype), s1, s2);

    checkPtype(x, ptype, xGlobal, ptypeGlobal);
}

#ifdef SPH_EXA_PTYPE_GPU_TEST
template<class KeyType, class T>
void ptypeExchangeGpu(int rank, int numRanks)
{
    int bucketSize = 4;
    float theta    = 1.0;

    StreamHolder stream;
    Domain<KeyType, T, execution::Gpu> domain(stream.exec(), rank, numRanks, bucketSize, bucketSize, theta,
                                              MPI_COMM_WORLD);

    std::vector<T> xGlobal, yGlobal, zGlobal, hGlobal;
    std::vector<uint8_t> ptypeGlobal;
    initGlobal(xGlobal, yGlobal, zGlobal, hGlobal, ptypeGlobal);

    std::vector<T> x, y, z, h;
    for (int i = rank; i < numGlobal; i += numRanks)
    {
        x.push_back(xGlobal[i]);
        y.push_back(yGlobal[i]);
        z.push_back(zGlobal[i]);
        h.push_back(hGlobal[i]);
    }

    DeviceVector<KeyType> d_keys = std::vector<KeyType>(x.size(), KeyType(0));
    DeviceVector<T> d_x = x, d_y = y, d_z = z, d_h = h;
    DeviceVector<T> s1, s2, s3;
    domain.sync(d_keys, d_x, d_y, d_z, d_h, std::tuple{}, std::tie(s1, s2, s3));

    ASSERT_EQ(domain.nParticlesWithHalos(), d_x.size());

    // fill the owned range with the true values; halo slots are overwritten by the exchange
    std::vector<T> xBuf = toHost(d_x);
    std::vector<uint8_t> ptypeInit(d_x.size(), uint8_t(-1));
    fillOwnedPtype(ptypeInit, xBuf, domain.startIndex(), domain.endIndex(), xGlobal, ptypeGlobal);
    DeviceVector<uint8_t> d_ptype = ptypeInit;

    DeviceVector<char> sendBuf, recvBuf;
    domain.exchangeHalos(std::tie(d_ptype), sendBuf, recvBuf);

    checkPtype(xBuf, toHost(d_ptype), xGlobal, ptypeGlobal);
}
#endif

TEST(PtypeExchange, cpu)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    ptypeExchangeCpu<uint64_t, double>(rank, numRanks);
    ptypeExchangeCpu<unsigned, float>(rank, numRanks);
}

#ifdef SPH_EXA_PTYPE_GPU_TEST
TEST(PtypeExchange, gpu)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    ptypeExchangeGpu<uint64_t, double>(rank, numRanks);
}
#endif
