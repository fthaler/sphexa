/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @brief 2-rank tests for the ve-pt zero-fluid-rank path (M4 item 5a)
 *
 * One rank owns ZERO fluid particles and holds no fluid halos: its whole buffer is non-fluid. At the
 * cstone raw-vector level (hand-set ptype, cf. fluid_halos.cpp / zeroFluidRankRequests), covers the
 * parts of the ve-pt computeForces pipeline not exercised by the M2 round trip:
 *
 * - the stable partition into the fluid-compacted layout with nFluid == 0 on the zero-fluid rank
 *   (exclusive scan of the NON-fluid flag) plus the inverse-permutation restore,
 * - the fluid halo request exchange: both ranks end up with empty outgoing AND incoming sets,
 * - an exchange on the compacted layout with an empty request set (nothing moves),
 * - the fluid tree view (focus leaves with the layout pointer swapped to the fluid layout) and
 *   group construction over the empty owned fluid range: zero groups / empty ranges, crash-free,
 *   with a positive control on the rank that owns fluid.
 *
 * Needs the M2b empty-range guards (rhoTimestep empty range, minMax(Gpu) empty range, reduceDt
 * zero-group skip) to be meaningful in production; here the cstone primitives underneath are
 * exercised directly.
 *
 * On the GPU build, the round trip exercises computeFluidLayout(Gpu), stablePartition(Gpu),
 * gatherArrays(Gpu) and the cstone spatial group builder computeGroupSplits(Gpu); at the end of each
 * rank's work no pending CUDA error may survive (cudaDeviceSynchronize + cudaGetLastError, checked
 * collectively so both ranks report).
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <mpi.h>
#include <vector>

#ifdef SPH_EXA_PTYPE_GPU_TEST
#define USE_CUDA
#include "cstone/cuda/device_vector.h"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/cuda/stream_holder.cuh"
#include "cstone/traversal/groups_gpu.h"
#endif

#include "cstone/domain/domain.hpp"
#include "cstone/halos/halos.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

/*! @brief peer-consistency: the number of fluid halos sent to each peer matches the number received
 *         (cf. checkPeerConsistency in fluid_halos.cpp) */
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

/*! @brief assert that every fluid halo sent by any peer matches the local incoming range, and that
 *         no requests exist in either direction (the zero-fluid-rank shape) */
template<class HalosType>
void checkEmptyRequests(const HalosType& fluidHalos, int numRanks)
{
    for (int peer = 0; peer < numRanks; ++peer)
    {
        ASSERT_EQ(fluidHalos.outgoing()[peer].totalCount(), 0) << "peer " << peer;
        ASSERT_EQ(fluidHalos.incoming()[peer].count(), 0) << "peer " << peer;
    }
}

//! @brief the hand-built focus tree: 8 leaves {0,2,...,16} with 2 particles per present leaf; rank 0
//!        owns leaves [0:4) and holds halo leaf 4, rank 1 owns [4:8) and holds halo leaf 3
template<class KeyType>
struct ZeroFluidRankSetup
{
    std::vector<KeyType>       leaves{0, 2, 4, 6, 8, 10, 12, 14, 16};
    std::vector<TreeIndexPair> assignment;

    std::vector<LocalIndex> layout0{0, 2, 4, 6, 8, 10, 10, 10, 10};
    std::vector<LocalIndex> layout1{0, 0, 0, 0, 2, 4, 6, 8, 10};

    explicit ZeroFluidRankSetup(int numRanks) : assignment(numRanks)
    {
        assignment[0] = TreeIndexPair(0, 4);
        assignment[1] = TreeIndexPair(4, 8);
    }

    TreeNodeIndex startCell(int rank) const { return TreeNodeIndex(rank == 0 ? 0 : 4); }
    TreeNodeIndex endCell(int rank) const { return TreeNodeIndex(rank == 0 ? 4 : 8); }
};

/*! @brief partition -> empty fluid requests -> empty-range group construction -> restore, CPU */
template<class KeyType, class T>
void zeroFluidRankPartitionCpu(int rank, int numRanks)
{
    ZeroFluidRankSetup<KeyType> setup(numRanks);
    std::span<const LocalIndex> layout =
        (rank == 0) ? std::span<const LocalIndex>{setup.layout0} : std::span<const LocalIndex>{setup.layout1};

    LocalIndex bufSize = layout.back();

    // rank 0 owns 6 fluid (leaves 0-2) and 4 non-fluid (leaf 3 + halo leaf 4); rank 1 owns zero
    // fluid and its halo leaf 3 is non-fluid, so its whole buffer is non-fluid
    std::vector<uint8_t> ptype0(bufSize, 1);
    for (LocalIndex i = 0; i < 6; ++i) { ptype0[i] = 0; }
    std::vector<uint8_t> ptype1(bufSize, 1);
    std::vector<uint8_t>& ptype = (rank == 0) ? ptype0 : ptype1;

    // identify particles by unique per-slot values (SFC order is fixed in the hand-built tree)
    std::vector<KeyType> keys(bufSize);
    std::vector<T>       h(bufSize);
    for (LocalIndex i = 0; i < bufSize; ++i)
    {
        keys[i] = 100 + KeyType(i);
        h[i]    = 0.1 * i;
    }

    // step 2: fluid layout, verified against a brute-force per-leaf count
    std::vector<LocalIndex> fluidLayoutAcc, fluidLayoutHost;
    computeFluidLayout(execution::cpu, layout, ptype.data(), fluidLayoutAcc, fluidLayoutHost, 1.05);

    TreeNodeIndex numLeaves = TreeNodeIndex(layout.size()) - 1;
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
    ASSERT_EQ(nFluid, LocalIndex(rank == 0 ? 6 : 0));

    LocalIndex fluidFirst = fluidLayoutHost[setup.startCell(rank)];
    LocalIndex fluidLast  = fluidLayoutHost[setup.endCell(rank)];
    if (rank == 0) { ASSERT_EQ(fluidLast - fluidFirst, nFluid); }
    else { ASSERT_EQ(fluidFirst, fluidLast); } // empty owned fluid range on the zero-fluid rank

    // step 3: fluid halo request exchange — both directions empty on both ranks (rank 1 has no
    // fluid at all, rank 0's halo leaf is non-fluid)
    Halos<KeyType> fluidHalos(rank, MPI_COMM_WORLD);
    fluidHalos.exchangeRequests(setup.leaves, setup.assignment, fluidLayoutHost);
    checkPeerConsistency(fluidHalos, rank, numRanks);
    checkEmptyRequests(fluidHalos, numRanks);

    // step 4: fused stable partition + gather of a small conserved set; on rank 1 nFluid == 0, so
    // no compacted fluid slots exist and the gather only touches the non-fluid tail
    std::vector<LocalIndex> ordering(bufSize), orderingInv(bufSize);
    std::vector<uint64_t>   partitionScratch(bufSize);
    stablePartition(execution::cpu, ptype.data(), bufSize, nFluid, ordering.data(), orderingInv.data(),
                    reinterpret_cast<unsigned*>(partitionScratch.data()));
    for (LocalIndex j = 0; j < nFluid; ++j)
    {
        ASSERT_EQ(ptype[ordering[j]], 0) << "compacted slot " << j;
    }

    std::vector<KeyType> keys0 = keys;
    std::vector<T>       h0    = h;
    std::vector<uint8_t> ptype0Save = ptype;

    auto gatherList = std::tie(keys, h, ptype);
    gatherArrays(execution::cpu, std::span<const LocalIndex>{ordering.data(), std::size_t(bufSize)}, 0, gatherList,
                 std::tie(partitionScratch));

    // the compacted fluid slots hold the fluid particles in their original relative order
    for (LocalIndex j = 0; j < nFluid; ++j)
    {
        ASSERT_EQ(keys[j], keys0[ordering[j]]) << "compacted slot " << j;
        ASSERT_EQ(h[j], h0[ordering[j]]) << "compacted slot " << j;
        ASSERT_EQ(ptype[j], 0) << "compacted slot " << j;
    }

    // step 7 (exchange): an exchange on the compacted layout with an empty request set moves nothing
    std::vector<T> v = h0;
    {
        std::vector<uint64_t> sendScratch(bufSize), recvScratch(bufSize);
        fluidHalos.exchangeHalos(execution::cpu, std::tie(v), sendScratch, recvScratch);
        checkPeerConsistency(fluidHalos, rank, numRanks);
        EXPECT_EQ(v, h0);
    }

    // step 5: fluid tree view — the focus leaves with the layout pointer swapped to the fluid layout
    OctreeNsView<T, KeyType> fluidTreeView{};
    fluidTreeView.numLeafNodes = numLeaves;
    fluidTreeView.leaves       = setup.leaves.data();
    fluidTreeView.layout       = fluidLayoutAcc.data();
    for (TreeNodeIndex i = 0; i < numLeaves + 1; ++i)
    {
        ASSERT_EQ(fluidTreeView.layout[i], fluidLayoutHost[i]) << "leaf " << i;
    }

    // step 7 (groups): CPU group construction over the owned fluid range [fluidFirst:fluidLast) is
    // the trivial whole-range group (sph/groups.hpp); on the zero-fluid rank it degenerates to a
    // single group over an empty range
    GroupData<execution::Cpu> groups;
    groups.firstBody  = fluidFirst;
    groups.lastBody   = fluidLast;
    groups.numGroups  = 1;
    groups.groupStart = &groups.firstBody;
    groups.groupEnd   = &groups.lastBody;
    auto gv           = groups.view();
    ASSERT_EQ(gv.numGroups, 1);
    ASSERT_EQ(gv.firstBody, fluidFirst);
    ASSERT_EQ(gv.lastBody, fluidLast);
    if (rank == 0) { ASSERT_LT(gv.firstBody, gv.lastBody); }
    else { ASSERT_EQ(gv.firstBody, gv.lastBody); } // empty group range on the zero-fluid rank

    // step 9: restore the full SFC order with the cached inverse permutation
    gatherArrays(execution::cpu, std::span<const LocalIndex>{orderingInv.data(), std::size_t(bufSize)}, 0, gatherList,
                 std::tie(partitionScratch));

    EXPECT_EQ(keys, keys0);
    EXPECT_EQ(h, h0);
    EXPECT_EQ(ptype, ptype0Save);
}

TEST(ZeroFluidRank, partitionAndGroups)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    zeroFluidRankPartitionCpu<uint64_t, double>(rank, numRanks);
    zeroFluidRankPartitionCpu<unsigned, float>(rank, numRanks);
}

#ifdef SPH_EXA_PTYPE_GPU_TEST
/*! @brief partition -> empty fluid requests -> empty-range group construction -> restore, GPU
 *
 * Same scenario as the CPU variant; the group construction uses the real cstone spatial group
 * builder (computeGroupSplits) over the owned fluid range, mirroring the ve-pt GPU path.
 */
template<class KeyType, class T>
void zeroFluidRankPartitionGpu(int rank, int numRanks)
{
    StreamHolder stream;
    ZeroFluidRankSetup<KeyType> setup(numRanks);

    std::vector<LocalIndex> layoutHost = (rank == 0) ? setup.layout0 : setup.layout1;
    LocalIndex              bufSize    = layoutHost.back();

    std::vector<uint8_t> ptypeInit(bufSize, 1);
    if (rank == 0)
    {
        for (LocalIndex i = 0; i < 6; ++i) { ptypeInit[i] = 0; }
    }

    // unique per-slot identifiers plus coordinates for the group split criterion
    std::vector<KeyType> keysInit(bufSize);
    std::vector<T>       hInit(bufSize), xInit(bufSize), yInit(bufSize), zInit(bufSize);
    for (LocalIndex i = 0; i < bufSize; ++i)
    {
        keysInit[i] = 100 + KeyType(i);
        hInit[i]    = 0.05;
        xInit[i]    = 0.01 * i;
        yInit[i]    = 0.02 * i;
        zInit[i]    = 0.03 * i;
    }

    DeviceVector<uint8_t> d_ptype = ptypeInit;
    DeviceVector<KeyType> d_keys  = keysInit;
    DeviceVector<T> d_h = hInit, d_x = xInit, d_y = yInit, d_z = zInit;
    DeviceVector<LocalIndex> d_layout = layoutHost;

    // step 2: fluid layout on the device, then D2H for the request exchange and the checks
    std::vector<LocalIndex> fluidLayoutHost;
    DeviceVector<LocalIndex> fluidLayoutAcc;
    computeFluidLayout(execution::Gpu(stream.exec()), std::span<const LocalIndex>{rawPtr(d_layout), layoutHost.size()},
                       rawPtr(d_ptype), fluidLayoutAcc, fluidLayoutHost, 1.05);

    TreeNodeIndex numLeaves = TreeNodeIndex(layoutHost.size()) - 1;
    ASSERT_EQ(fluidLayoutHost.size(), std::size_t(numLeaves) + 1);
    LocalIndex fluidCount = 0;
    for (TreeNodeIndex i = 0; i < numLeaves; ++i)
    {
        ASSERT_EQ(fluidLayoutHost[i], fluidCount) << "leaf " << i;
        for (LocalIndex j = layoutHost[i]; j < layoutHost[i + 1]; ++j)
        {
            if (ptypeInit[j] == 0) { ++fluidCount; }
        }
    }
    ASSERT_EQ(fluidLayoutHost.back(), fluidCount);
    LocalIndex nFluid = fluidCount;
    ASSERT_EQ(nFluid, LocalIndex(rank == 0 ? 6 : 0));

    LocalIndex fluidFirst = fluidLayoutHost[setup.startCell(rank)];
    LocalIndex fluidLast  = fluidLayoutHost[setup.endCell(rank)];
    if (rank == 0) { ASSERT_EQ(fluidLast - fluidFirst, nFluid); }
    else { ASSERT_EQ(fluidFirst, fluidLast); }

    // step 3: fluid halo request exchange — empty in both directions on both ranks
    Halos<KeyType> fluidHalos(rank, MPI_COMM_WORLD);
    fluidHalos.exchangeRequests(setup.leaves, setup.assignment, fluidLayoutHost);
    checkPeerConsistency(fluidHalos, rank, numRanks);
    checkEmptyRequests(fluidHalos, numRanks);

    // step 4: fused stable partition + gather of a small conserved set
    DeviceVector<LocalIndex> d_ordering(bufSize), d_orderingInv(bufSize);
    DeviceVector<uint64_t>   d_partitionScratch(bufSize);
    stablePartition(execution::Gpu(stream.exec()), rawPtr(d_ptype), bufSize, nFluid, rawPtr(d_ordering),
                    rawPtr(d_orderingInv), reinterpret_cast<unsigned*>(rawPtr(d_partitionScratch)));

    DeviceVector<KeyType> d_keys0 = d_keys;
    DeviceVector<T> d_h0 = d_h;
    DeviceVector<uint8_t> d_ptype0 = d_ptype;

    auto gatherList = std::tie(d_keys, d_h, d_ptype);
    gatherArrays(execution::Gpu(stream.exec()),
                 std::span<const LocalIndex>{rawPtr(d_ordering), std::size_t(bufSize)}, 0, gatherList,
                 std::tie(d_partitionScratch));

    {
        std::vector<LocalIndex> orderingHost   = toHost(d_ordering);
        std::vector<KeyType>    keysCompacted  = toHost(d_keys);
        std::vector<T>          hCompacted     = toHost(d_h);
        std::vector<uint8_t>    ptypeCompacted = toHost(d_ptype);
        for (LocalIndex j = 0; j < nFluid; ++j)
        {
            LocalIndex src = orderingHost[j];
            ASSERT_EQ(keysCompacted[j], keysInit[src]) << "compacted slot " << j;
            ASSERT_EQ(hCompacted[j], hInit[src]) << "compacted slot " << j;
            ASSERT_EQ(ptypeCompacted[j], 0) << "compacted slot " << j;
        }
    }

    // step 7 (exchange): an exchange on the compacted layout with an empty request set moves nothing
    {
        DeviceVector<T> d_v = hInit;
        DeviceVector<uint64_t> d_sendScratch(bufSize), d_recvScratch(bufSize);
        fluidHalos.exchangeHalos(execution::Gpu(stream.exec()), std::tie(d_v), d_sendScratch, d_recvScratch);
        checkPeerConsistency(fluidHalos, rank, numRanks);
        EXPECT_EQ(toHost(d_v), hInit);
    }

    // step 5: fluid tree view — the focus leaves with the layout pointer swapped to the fluid layout
    DeviceVector<KeyType> d_leaves = setup.leaves;
    OctreeNsView<T, KeyType> fluidTreeView{};
    fluidTreeView.numLeafNodes = numLeaves;
    fluidTreeView.leaves       = rawPtr(d_leaves);
    fluidTreeView.layout       = rawPtr(fluidLayoutAcc);
    EXPECT_EQ(toHost(fluidLayoutAcc), fluidLayoutHost);

    // step 7 (groups): the cstone spatial group builder over the owned fluid range
    // [fluidFirst:fluidLast) with the fluid view; the zero-fluid rank owns an empty range and must
    // come out with zero groups and an empty range, crash-free
    GroupData<execution::Gpu> groups;
    {
        DeviceVector<LocalIndex> numSplitsPerGroup;
        float                    tolFactor = 2.0f;
        cstone::Box<T>           box{0, 1};
        computeGroupSplits(stream.exec(), fluidFirst, fluidLast, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h),
                           rawPtr(d_leaves), numLeaves, rawPtr(fluidLayoutAcc), box, 2 * GpuConfig::warpSize,
                           tolFactor, numSplitsPerGroup, groups.data);
        stream.sync();
    }
    groups.firstBody  = fluidFirst;
    groups.lastBody   = fluidLast;
    groups.numGroups  = LocalIndex(groups.data.size()) - 1;
    groups.groupStart = rawPtr(groups.data);
    groups.groupEnd   = rawPtr(groups.data) + 1;

    {
        std::vector<LocalIndex> groupRanges = toHost(groups.data);
        if (rank == 1)
        {
            // empty fluid range: zero groups, the single range entry repeats the (empty) end
            ASSERT_EQ(groups.numGroups, 0);
            ASSERT_EQ(groupRanges, std::vector<LocalIndex>{fluidLast});
        }
        else
        {
            // positive control: the owned fluid range is covered by >= 1 group in [fluidFirst:fluidLast)
            ASSERT_GE(groups.numGroups, 1);
            ASSERT_EQ(groupRanges.front(), fluidFirst);
            ASSERT_EQ(groupRanges.back(), fluidLast);
        }
    }

    // step 9: restore the full SFC order with the cached inverse permutation
    gatherArrays(execution::Gpu(stream.exec()),
                 std::span<const LocalIndex>{rawPtr(d_orderingInv), std::size_t(bufSize)}, 0, gatherList,
                 std::tie(d_partitionScratch));

    EXPECT_EQ(toHost(d_keys), toHost(d_keys0));
    EXPECT_EQ(toHost(d_h), toHost(d_h0));
    EXPECT_EQ(toHost(d_ptype), toHost(d_ptype0));

    // no pending CUDA error may survive the zero-fluid-rank work: the old hazard was a non-sticky
    // pending cudaErrorInvalidConfiguration surfacing at the next unrelated call. Checked
    // collectively so both ranks report.
    checkGpuErrors(cudaDeviceSynchronize());
    cudaError_t localError  = cudaGetLastError();
    int         localErrInt = int(localError), globalErrInt = 0;
    MPI_Allreduce(&localErrInt, &globalErrInt, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    ASSERT_EQ(cudaError_t(globalErrInt), cudaSuccess) << cudaGetErrorString(cudaError_t(globalErrInt));
}

TEST(ZeroFluidRank, partitionAndGroupsGpu)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    zeroFluidRankPartitionGpu<uint64_t, double>(rank, numRanks);
}
#endif
