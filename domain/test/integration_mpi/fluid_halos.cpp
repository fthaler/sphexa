/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @brief 2-rank tests for the ve-pt fluid halo machinery (stable partition + second Halos instance)
 *
 * Covers (M2): the partition -> fluidHalos exchange -> restore round trip with conserved quantities
 * identical, zero-fluid-count leaf handling in the fluid layout, and request emptiness for an
 * all-non-fluid layout (a rank whose whole buffer is non-fluid must issue an empty request set and
 * match nothing).
 *
 * On the GPU build, the round trip exercises computeFluidLayout(Gpu), stablePartition(Gpu) and the
 * GPU halo exchange with dedicated scratch buffers; the send path requires the
 * GATHER_RANGES(uint8_t, unsigned) instantiation to link.
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
#include "cstone/halos/halos.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

//! @brief global particle set in SFC order, cf. exchange_ptype.cpp
constexpr int numGlobal = 12;

template<class T>
void initGlobal(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, std::vector<T>& h,
                std::vector<uint8_t>& ptype, std::vector<T>& m, std::vector<T>& vx)
{
    x     = {0.0, 0.11, 0.261, 0.281, 0.301, 0.321, 0.521, 0.541, 0.561, 0.761, 0.781, 1.000};
    y     = {0.0, 0.12, 0.262, 0.282, 0.302, 0.322, 0.522, 0.542, 0.562, 0.762, 0.781, 1.000};
    z     = {0.0, 0.13, 0.263, 0.283, 0.303, 0.323, 0.523, 0.543, 0.563, 0.763, 0.781, 1.000};
    h     = {0.1, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109, 0.110, 0.111};
    ptype = {1, 0, 3, 0, 0, 2, 7, 0, 5, 1, 0, 4};
    m     = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
    vx    = {12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0};
}

//! @brief index of the particle with unique x coordinate @p x in the global set, or std::size_t(-1) if not found
template<class T>
std::size_t globalIndex(T x, const std::vector<T>& xGlobal)
{
    auto it = std::find(xGlobal.begin(), xGlobal.end(), x);
    return it == xGlobal.end() ? std::size_t(-1) : std::size_t(it - xGlobal.begin());
}

//! @brief fill the owned range [start:end) of @p ptype and @p m with the true values, and @p vx for
//! owned *fluid* particles only: vx probes the fluid-only halo exchange, so non-fluid slots (owned
//! and halo) keep the -1 sentinel, matching particles by (unique) x; halo slots are left untouched
template<class T>
void fillOwned(std::vector<uint8_t>& ptype, std::vector<T>& m, std::vector<T>& vx, const std::vector<T>& x,
               LocalIndex start, LocalIndex end, const std::vector<T>& xGlobal, const std::vector<uint8_t>& ptypeGlobal,
               const std::vector<T>& mGlobal, const std::vector<T>& vxGlobal)
{
    for (LocalIndex i = start; i < end; ++i)
    {
        std::size_t j = globalIndex(x[i], xGlobal);
        ASSERT_NE(j, std::size_t(-1));
        ptype[i] = ptypeGlobal[j];
        m[i]     = mGlobal[j];
        if (ptypeGlobal[j] == 0) { vx[i] = vxGlobal[j]; }
    }
}

//! @brief check ptype/m per particle and vx for fluid particles against the global set, identifying
//! particles by (unique) x; non-fluid vx slots legitimately keep the -1 sentinel (never exchanged)
template<class T>
void checkAgainstGlobal(const std::vector<T>& x, const std::vector<uint8_t>& ptype, const std::vector<T>& m,
                        const std::vector<T>& vx, const std::vector<T>& xGlobal,
                        const std::vector<uint8_t>& ptypeGlobal, const std::vector<T>& mGlobal,
                        const std::vector<T>& vxGlobal)
{
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        std::size_t j = globalIndex(x[i], xGlobal);
        ASSERT_NE(j, std::size_t(-1)) << "particle x = " << x[i] << " not found in global set";
        EXPECT_EQ(ptype[i], ptypeGlobal[j]) << "buffer slot " << i << " (x = " << x[i] << ")";
        EXPECT_EQ(m[i], mGlobal[j]) << "buffer slot " << i << " (x = " << x[i] << ")";
        if (ptype[i] == 0) { EXPECT_EQ(vx[i], vxGlobal[j]) << "buffer slot " << i << " (x = " << x[i] << ")"; }
    }
}

//! @brief peer-consistency: the number of fluid halos sent to each peer matches the number received
template<class HalosType>
void checkPeerConsistency(const HalosType& fluidHalos, int rank, int numRanks)
{
    std::vector<int> sendCounts(size_t(numRanks) * numRanks, 0);
    for (int peer = 0; peer < numRanks; ++peer)
    {
        sendCounts[size_t(rank) * numRanks + peer] = int(fluidHalos.outgoing()[peer].totalCount());
    }
    MPI_Allreduce(MPI_IN_PLACE, sendCounts.data(), int(sendCounts.size()), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    for (int peer = 0; peer < numRanks; ++peer)
    {
        ASSERT_EQ(sendCounts[size_t(peer) * numRanks + rank], int(fluidHalos.incoming()[peer].count()));
    }
}

/*! @brief partition -> fluidHalos exchange -> restore round trip
 *
 * Mirrors the ve-pt propagator's computeForces steps 1-4/9 on raw vectors: after the full sync and
 * the (ptype, m) exchange, the buffer is stable-partitioned into the fluid-compacted layout, the
 * fluid halo pattern is established on a second Halos instance and vx (a field only exchanged
 * through the fluid pattern) is delivered into the compacted halo fluid slots, then the cached
 * inverse permutation restores the full SFC layout with all conserved quantities identical.
 */
template<class KeyType, class T>
void fluidRoundTripCpu(int rank, int numRanks)
{
    int bucketSize = 4;
    float theta    = 1.0;
    Domain<KeyType, T> domain(execution::cpu, rank, numRanks, bucketSize, bucketSize, theta, MPI_COMM_WORLD);

    std::vector<T> xGlobal, yGlobal, zGlobal, hGlobal, mGlobal, vxGlobal;
    std::vector<uint8_t> ptypeGlobal;
    initGlobal(xGlobal, yGlobal, zGlobal, hGlobal, ptypeGlobal, mGlobal, vxGlobal);

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

    LocalIndex bufSize = domain.nParticlesWithHalos();
    ASSERT_EQ(bufSize, x.size());

    // step 1: fill the owned ptype/m/vx with the true values (halo slots hold garbage until exchanged)
    std::vector<uint8_t> ptype(bufSize, uint8_t(255));
    std::vector<T> m(bufSize, -1), vx(bufSize, -1);
    fillOwned(ptype, m, vx, x, domain.startIndex(), domain.endIndex(), xGlobal, ptypeGlobal, mGlobal, vxGlobal);

    // step 1 (contd): full-pattern halo exchange of ptype and m
    domain.exchangeHalos(std::tie(ptype, m), s1, s2);

    // step 2: fluid layout, verified against a brute-force per-leaf count (covers zero-fluid leaves)
    std::vector<LocalIndex> fluidLayoutAcc, fluidLayoutHost;
    computeFluidLayout(execution::cpu, domain.layout(), ptype.data(), fluidLayoutAcc, fluidLayoutHost, 1.05);

    std::span<const LocalIndex> layout = domain.layout();
    TreeNodeIndex     numLeaves        = TreeNodeIndex(layout.size()) - 1;
    ASSERT_EQ(fluidLayoutHost.size(), std::size_t(numLeaves) + 1);
    LocalIndex fluidCount = 0;
    for (TreeNodeIndex i = 0; i < numLeaves; ++i)
    {
        ASSERT_EQ(fluidLayoutHost[i], fluidCount) << "leaf " << i;
        for (LocalIndex j = layout[i]; j < layout[i + 1]; ++j)
        {
            if (ptype[j] == 0) { ++fluidCount; }
        }
    }
    ASSERT_EQ(fluidLayoutHost.back(), fluidCount);
    LocalIndex nFluid = fluidCount;

    // step 3: fluid halo request exchange on a second Halos instance
    Halos<KeyType> fluidHalos(rank, MPI_COMM_WORLD);
    fluidHalos.exchangeRequests(domain.focusTree().treeLeaves(), domain.focusTree().assignment(), fluidLayoutHost);
    checkPeerConsistency(fluidHalos, rank, numRanks);

    // incoming ranges must lie in the halo fluid slots [0:fluidFirst) + [fluidLast:nFluid)
    LocalIndex fluidFirst = fluidLayoutHost[domain.startCell()];
    LocalIndex fluidLast  = fluidLayoutHost[domain.endCell()];
    for (int peer = 0; peer < numRanks; ++peer)
    {
        auto in = fluidHalos.incoming()[peer];
        if (in.count() == 0) { continue; }
        EXPECT_TRUE(in.start() + in.count() <= fluidFirst || in.start() >= fluidLast) << "peer " << peer;
    }

    // step 4: fused stable partition + gather of the conserved set
    std::vector<LocalIndex> ordering(bufSize), orderingInv(bufSize);
    std::vector<uint64_t>   partitionScratch(bufSize);
    stablePartition(execution::cpu, ptype.data(), bufSize, nFluid, ordering.data(), orderingInv.data(),
                    reinterpret_cast<unsigned*>(partitionScratch.data()));
    for (LocalIndex j = 0; j < nFluid; ++j)
    {
        ASSERT_EQ(ptype[ordering[j]], 0) << "compacted slot " << j;
    }

    std::vector<T>       x0 = x, y0 = y, z0 = z, h0 = h, m0 = m;
    std::vector<uint8_t> ptype0 = ptype;
    std::vector<KeyType> keys0  = keys;

    auto gatherList = std::tie(keys, x, y, z, h, m, ptype, vx);
    gatherArrays(execution::cpu, std::span<const LocalIndex>{ordering.data(), std::size_t(bufSize)}, 0, gatherList,
                 std::tie(partitionScratch));

    // the compacted fluid slots hold the fluid particles in their original relative order
    for (LocalIndex j = 0; j < nFluid; ++j)
    {
        std::size_t g = globalIndex(x[j], xGlobal);
        ASSERT_NE(g, std::size_t(-1)) << "compacted slot " << j;
        ASSERT_EQ(ptype[j], ptypeGlobal[g]);
        ASSERT_EQ(m[j], mGlobal[g]);
    }

    // the fluid halo exchange delivers vx into the compacted halo fluid slots [0:fluidFirst) +
    // [fluidLast:nFluid) and must not touch the non-fluid slots [nFluid:bufSize)
    std::vector<uint64_t> fluidSendScratch(bufSize), fluidRecvScratch(bufSize);
    fluidHalos.exchangeHalos(execution::cpu, std::tie(vx), fluidSendScratch, fluidRecvScratch);
    checkPeerConsistency(fluidHalos, rank, numRanks);

    for (LocalIndex j = 0; j < nFluid; ++j)
    {
        std::size_t g = globalIndex(x[j], xGlobal);
        ASSERT_NE(g, std::size_t(-1));
        ASSERT_EQ(vx[j], vxGlobal[g]) << "compacted fluid slot " << j;
    }
    for (LocalIndex j = nFluid; j < bufSize; ++j)
    {
        ASSERT_EQ(vx[j], -1) << "non-fluid slot " << j << " must not be touched by the fluid exchange";
        ASSERT_NE(ptype[j], 0);
    }

    // step 9: restore the full SFC order with the cached inverse permutation; conserved quantities
    // identical, halo vx now valid wherever the particle is fluid
    gatherArrays(execution::cpu, std::span<const LocalIndex>{orderingInv.data(), std::size_t(bufSize)}, 0, gatherList,
                 std::tie(partitionScratch));

    EXPECT_EQ(keys, keys0);
    EXPECT_EQ(x, x0);
    EXPECT_EQ(y, y0);
    EXPECT_EQ(z, z0);
    EXPECT_EQ(h, h0);
    EXPECT_EQ(m, m0);
    EXPECT_EQ(ptype, ptype0);
    checkAgainstGlobal(x, ptype, m, vx, xGlobal, ptypeGlobal, mGlobal, vxGlobal);
}

/*! @brief request emptiness for an all-non-fluid layout, plus the zero-owned-fluid positive control
 *
 * Hand-built focus tree of 8 leaves {0,2,...,16}; rank 0 owns leaves [0:4), rank 1 owns [4:8); every
 * leaf holds 2 particles. Rank 0 holds halo leaf 4 (rank 1's first leaf), rank 1 holds halo leaf 3
 * (rank 0's last leaf).
 *
 * Scenario A: leaves 0-2 fluid, all others non-fluid -> rank 1's whole buffer (owned + halos) is
 * non-fluid: its fluid layout is empty, it must issue an empty request set, match nothing and the
 * exchange must leave its buffers untouched.
 *
 * Scenario B: leaf 3 fluid as well -> rank 1 still owns zero fluid, but its halo slots hold fluid:
 * it requests leaf 3 and receives its 2 fluid particles into the compacted halo slots [0:2), while
 * rank 0 receives nothing from rank 1.
 */
template<class KeyType, class T>
void zeroFluidRankRequests(int rank, int numRanks)
{
    std::vector<KeyType>      leaves{0, 2, 4, 6, 8, 10, 12, 14, 16};
    std::vector<TreeIndexPair> assignment(numRanks);
    assignment[0] = TreeIndexPair(0, 4);
    assignment[1] = TreeIndexPair(4, 8);

    // buffer layouts: rank 0 owns leaves 0-3 and holds halo leaf 4; rank 1 owns leaves 4-7 and holds
    // halo leaf 3; every present leaf holds 2 particles
    std::vector<LocalIndex> layout0{0, 2, 4, 6, 8, 10, 10, 10, 10};
    std::vector<LocalIndex> layout1{0, 0, 0, 0, 2, 4, 6, 8, 10};
    std::span<const LocalIndex> layout = (rank == 0) ? std::span<const LocalIndex>{layout0}
                                                     : std::span<const LocalIndex>{layout1};

    for (int scenario = 0; scenario < 2; ++scenario)
    {
        // fluid leaves: scenario A: {0, 1, 2}, scenario B: {0, 1, 2, 3}
        auto leafFluid = [scenario](TreeNodeIndex leaf) { return leaf < 3 + scenario; };

        // rank 0 owns leaves 0-3 and holds the halo leaf 4 (non-fluid, owned by rank 1); rank 1 owns
        // leaves 4-7 (all non-fluid) and holds the halo leaf 3, whose ptype must match the owner
        std::vector<uint8_t> ptype0(layout0.back(), 1);
        for (TreeNodeIndex leaf = 0; leaf < 4; ++leaf)
        {
            uint8_t p = leafFluid(leaf) ? 0 : 1;
            ptype0[layout0[leaf]]     = p;
            ptype0[layout0[leaf] + 1] = p;
        }
        std::vector<LocalIndex> fluidLayout0, fluidLayout0Host;
        computeFluidLayout(execution::cpu, layout0, ptype0.data(), fluidLayout0, fluidLayout0Host, 1.05);

        std::vector<uint8_t> ptype1(layout1.back(), 1);
        if (leafFluid(3))
        {
            ptype1[0] = 0;
            ptype1[1] = 0;
        }
        std::vector<LocalIndex> fluidLayout1, fluidLayout1Host;
        computeFluidLayout(execution::cpu, layout1, ptype1.data(), fluidLayout1, fluidLayout1Host, 1.05);

        auto& ptype       = (rank == 0) ? ptype0 : ptype1;
        auto& fluidLayout = (rank == 0) ? fluidLayout0Host : fluidLayout1Host;
        std::vector<T> v(layout.back());
        for (std::size_t i = 0; i < v.size(); ++i) { v[i] = (ptype[i] == 0) ? T(i) : T(-1); }
        std::vector<T> v0 = v;

        Halos<KeyType> fluidHalos(rank, MPI_COMM_WORLD);
        fluidHalos.exchangeRequests(leaves, assignment, fluidLayout);

        // every sent request arrives: sent counts match received counts
        checkPeerConsistency(fluidHalos, rank, numRanks);

        if (scenario == 0)
        {
            // rank 1 is the all-non-fluid rank: empty fluid layout, empty request set. Rank 0
            // legitimately holds the 6 fluid particles of leaves 0-2 and requests nothing either
            // (its halo leaf 4 is non-fluid), so both ranks must match nothing here.
            ASSERT_EQ(fluidLayout.back(), LocalIndex(rank == 1 ? 0 : 6));
            for (int peer = 0; peer < numRanks; ++peer)
            {
                ASSERT_EQ(fluidHalos.outgoing()[peer].totalCount(), 0);
                ASSERT_EQ(fluidHalos.incoming()[peer].count(), 0);
            }
        }
        else
        {
            // rank 1 owns zero fluid but holds 2 fluid halos of rank 0's leaf 3
            if (rank == 1)
            {
                ASSERT_EQ(fluidLayout.back(), 2);
                ASSERT_EQ(fluidHalos.outgoing()[1].totalCount(), 0);
                ASSERT_EQ(fluidHalos.incoming()[0].start(), 0);
                ASSERT_EQ(fluidHalos.incoming()[0].count(), 2);
            }
            else
            {
                ASSERT_EQ(fluidLayout.back(), 8);
                ASSERT_EQ(fluidHalos.outgoing()[1].totalCount(), 2);
                ASSERT_EQ(fluidHalos.incoming()[1].count(), 0);
            }
        }

        std::vector<uint64_t> sendScratch(v.size()), recvScratch(v.size());
        fluidHalos.exchangeHalos(execution::cpu, std::tie(v), sendScratch, recvScratch);

        if (scenario == 0)
        {
            // nothing matched: buffers untouched on both ranks
            EXPECT_EQ(v, v0);
        }
        else
        {
            if (rank == 1)
            {
                // the 2 fluid halos of rank 0's leaf 3 arrive in the compacted halo slots [0:2);
                // rank 0's leaf 3 holds particles 6 and 7 of its buffer
                EXPECT_EQ(v[0], T(6));
                EXPECT_EQ(v[1], T(7));
                EXPECT_EQ(std::vector<T>(v.begin() + 2, v.end()), std::vector<T>(v0.begin() + 2, v0.end()));
            }
            else
            {
                EXPECT_EQ(v, v0);
            }
        }

        // the two request exchanges are fenced by the Alltoall inside the next exchangeRequests
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

TEST(FluidHalos2Ranks, roundTrip)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    fluidRoundTripCpu<uint64_t, double>(rank, numRanks);
    fluidRoundTripCpu<unsigned, float>(rank, numRanks);
}

TEST(FluidHalos2Ranks, zeroFluidRankRequests)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    zeroFluidRankRequests<uint64_t, double>(rank, numRanks);
}

#ifdef SPH_EXA_PTYPE_GPU_TEST
template<class KeyType, class T>
void fluidRoundTripGpu(int rank, int numRanks)
{
    int bucketSize = 4;
    float theta    = 1.0;

    StreamHolder stream;
    Domain<KeyType, T, execution::Gpu> domain(stream.exec(), rank, numRanks, bucketSize, bucketSize, theta,
                                              MPI_COMM_WORLD);

    std::vector<T> xGlobal, yGlobal, zGlobal, hGlobal, mGlobal, vxGlobal;
    std::vector<uint8_t> ptypeGlobal;
    initGlobal(xGlobal, yGlobal, zGlobal, hGlobal, ptypeGlobal, mGlobal, vxGlobal);

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

    LocalIndex bufSize = domain.nParticlesWithHalos();
    ASSERT_EQ(bufSize, d_x.size());

    // step 1: fill owned ptype/m/vx with the true values, then the full-pattern exchange
    std::vector<T> xBuf = toHost(d_x);
    std::vector<uint8_t> ptypeInit(d_x.size(), uint8_t(255));
    std::vector<T> mInit(d_x.size(), -1), vxInit(d_x.size(), -1);
    fillOwned(ptypeInit, mInit, vxInit, xBuf, domain.startIndex(), domain.endIndex(), xGlobal, ptypeGlobal, mGlobal,
              vxGlobal);
    DeviceVector<uint8_t> d_ptype = ptypeInit;
    DeviceVector<T> d_m = mInit, d_vx = vxInit;

    DeviceVector<char> sendBuf, recvBuf;
    domain.exchangeHalos(std::tie(d_ptype, d_m), sendBuf, recvBuf);

    // step 2: fluid layout on the device, then D2H for the request exchange and the checks
    std::vector<LocalIndex> fluidLayoutHost;
    DeviceVector<LocalIndex> fluidLayoutAcc;
    computeFluidLayout(execution::Gpu(stream.exec()), domain.layout(), rawPtr(d_ptype), fluidLayoutAcc,
                       fluidLayoutHost, 1.05);

    std::vector<LocalIndex> layoutHost(domain.layout().size());
    memcpyD2HAsync(stream.exec(), domain.layout().data(), layoutHost.size(), layoutHost.data());
    syncGpu(stream.exec());
    std::vector<uint8_t>    ptypeHost = toHost(d_ptype);
    TreeNodeIndex           numLeaves = TreeNodeIndex(layoutHost.size()) - 1;
    ASSERT_EQ(fluidLayoutHost.size(), std::size_t(numLeaves) + 1);
    LocalIndex fluidCount = 0;
    for (TreeNodeIndex i = 0; i < numLeaves; ++i)
    {
        ASSERT_EQ(fluidLayoutHost[i], fluidCount) << "leaf " << i;
        for (LocalIndex j = layoutHost[i]; j < layoutHost[i + 1]; ++j)
        {
            if (ptypeHost[j] == 0) { ++fluidCount; }
        }
    }
    ASSERT_EQ(fluidLayoutHost.back(), fluidCount);
    LocalIndex nFluid = fluidCount;

    // step 3: fluid halo request exchange
    Halos<KeyType> fluidHalos(rank, MPI_COMM_WORLD);
    fluidHalos.exchangeRequests(domain.focusTree().treeLeaves(), domain.focusTree().assignment(), fluidLayoutHost);
    checkPeerConsistency(fluidHalos, rank, numRanks);

    // step 4: fused stable partition + gather of the conserved set
    DeviceVector<LocalIndex> d_ordering(bufSize), d_orderingInv(bufSize);
    DeviceVector<uint64_t>   d_partitionScratch(bufSize);
    stablePartition(execution::Gpu(stream.exec()), rawPtr(d_ptype), bufSize, nFluid, rawPtr(d_ordering),
                    rawPtr(d_orderingInv), reinterpret_cast<unsigned*>(rawPtr(d_partitionScratch)));

    DeviceVector<KeyType> d_keys0 = d_keys;
    DeviceVector<T> d_x0 = d_x, d_y0 = d_y, d_z0 = d_z, d_h0 = d_h, d_m0 = d_m;
    DeviceVector<uint8_t> d_ptype0 = d_ptype;

    auto gatherList = std::tie(d_keys, d_x, d_y, d_z, d_h, d_m, d_ptype, d_vx);
    gatherArrays(execution::Gpu(stream.exec()),
                 std::span<const LocalIndex>{rawPtr(d_ordering), std::size_t(bufSize)}, 0, gatherList,
                 std::tie(d_partitionScratch));

    // the fluid halo exchange delivers vx into the compacted halo fluid slots
    DeviceVector<uint64_t> d_fluidSendScratch(bufSize), d_fluidRecvScratch(bufSize);
    fluidHalos.exchangeHalos(execution::Gpu(stream.exec()), std::tie(d_vx), d_fluidSendScratch, d_fluidRecvScratch);
    checkPeerConsistency(fluidHalos, rank, numRanks);

    LocalIndex fluidFirst = fluidLayoutHost[domain.startCell()];
    LocalIndex fluidLast  = fluidLayoutHost[domain.endCell()];
    {
        std::vector<T> vxCompacted = toHost(d_vx);
        std::vector<T> xCompacted  = toHost(d_x);
        for (LocalIndex j = 0; j < nFluid; ++j)
        {
            std::size_t g = globalIndex(xCompacted[j], xGlobal);
            ASSERT_NE(g, std::size_t(-1));
            ASSERT_EQ(vxCompacted[j], vxGlobal[g]) << "compacted fluid slot " << j;
        }
        for (LocalIndex j = nFluid; j < bufSize; ++j)
        {
            ASSERT_EQ(vxCompacted[j], -1) << "non-fluid slot " << j;
        }
    }

    // step 9: restore the full SFC order with the cached inverse permutation
    gatherArrays(execution::Gpu(stream.exec()),
                 std::span<const LocalIndex>{rawPtr(d_orderingInv), std::size_t(bufSize)}, 0, gatherList,
                 std::tie(d_partitionScratch));

    EXPECT_EQ(toHost(d_keys), toHost(d_keys0));
    EXPECT_EQ(toHost(d_x), toHost(d_x0));
    EXPECT_EQ(toHost(d_y), toHost(d_y0));
    EXPECT_EQ(toHost(d_z), toHost(d_z0));
    EXPECT_EQ(toHost(d_h), toHost(d_h0));
    EXPECT_EQ(toHost(d_m), toHost(d_m0));
    EXPECT_EQ(toHost(d_ptype), toHost(d_ptype0));
    {
        std::vector<T>    vxFinal = toHost(d_vx);
        std::vector<uint8_t> ptypeFinal = toHost(d_ptype);
        for (std::size_t i = 0; i < vxFinal.size(); ++i)
        {
            std::size_t g = globalIndex(xBuf[i], xGlobal);
            ASSERT_NE(g, std::size_t(-1));
            EXPECT_EQ(ptypeFinal[i], ptypeGlobal[g]);
            if (ptypeFinal[i] == 0) { EXPECT_EQ(vxFinal[i], vxGlobal[g]) << "buffer slot " << i; }
        }
    }
}

TEST(FluidHalos2Ranks, roundTripGpu)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    fluidRoundTripGpu<uint64_t, double>(rank, numRanks);
}
#endif
