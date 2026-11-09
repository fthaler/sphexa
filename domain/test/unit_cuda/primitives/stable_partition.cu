/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file @brief GPU tests for the fused stable partition, the fluid leaf layout helper and the masked fill
 */

#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "cstone/cuda/stream_holder.cuh"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/primitives/primitives_gpu.h"

#include "../../unit/primitives/stable_partition_checks.hpp"

using namespace cstone;

namespace
{

//! @brief gathering with the ordering (resident on device) must place the fluid elements at [0:nFluid)
//!        in original order
void checkGpuGather(execution::Gpu exec, const std::vector<uint8_t>& ptype, const std::vector<LocalIndex>& ordering,
                    LocalIndex nFluid)
{
    LocalIndex n = LocalIndex(ptype.size());
    std::vector<double> src(n);
    for (LocalIndex i = 0; i < n; ++i) { src[i] = 0.5 * i; }

    DeviceVector<double> d_src           = src;
    DeviceVector<double> d_dst           = std::vector<double>(n, 0.0);
    DeviceVector<LocalIndex> d_ordering  = ordering;
    gather(exec, std::span<const LocalIndex>(rawPtr(d_ordering), n), rawPtr(d_src), rawPtr(d_dst));
    syncGpu(exec);

    checkGatherResult(ptype, src, toHost(d_dst), ordering, nFluid);
}

void checkGpuPartition(const std::vector<uint8_t>& ptype, LocalIndex nFluid)
{
    StreamHolder stream;
    auto exec = stream.exec();

    LocalIndex n = LocalIndex(ptype.size());
    DeviceVector<uint8_t> d_ptype       = ptype;
    DeviceVector<LocalIndex> d_ordering = std::vector<LocalIndex>(n, LocalIndex(-1));
    DeviceVector<LocalIndex> d_orderInv = std::vector<LocalIndex>(n, LocalIndex(-1));
    DeviceVector<unsigned> d_workspace  = std::vector<unsigned>(n, 0u);

    stablePartition(exec, rawPtr(d_ptype), n, nFluid, rawPtr(d_ordering), rawPtr(d_orderInv), rawPtr(d_workspace));
    syncGpu(exec);

    std::vector<LocalIndex> ordering   = toHost(d_ordering);
    std::vector<LocalIndex> orderingInv = toHost(d_orderInv);

    checkPermutations(ptype, ordering, orderingInv, nFluid);
    checkStability(ptype, ordering, nFluid);
    checkGpuGather(exec, ptype, ordering, nFluid);
}

} // namespace

TEST(StablePartitionGpu, edgeCases)
{
    checkGpuPartition({0, 0, 0}, 3);          // all fluid
    checkGpuPartition({1, 2, 3}, 0);          // all non-fluid
    checkGpuPartition({1, 1, 0, 1}, 1);       // single fluid
    checkGpuPartition({0, 1, 0, 1, 0, 1}, 3); // alternating
}

TEST(StablePartitionGpu, randomized)
{
    std::mt19937 gen(42);

    for (int rep = 0; rep < 20; ++rep)
    {
        LocalIndex n = LocalIndex(1 + rep * 11);
        std::vector<uint8_t> ptype;
        LocalIndex nFluid = randomFlags(gen, n, ptype);
        checkGpuPartition(ptype, nFluid);
    }
}

TEST(FluidLayoutGpu, bruteForce)
{
    StreamHolder stream;
    auto exec = stream.exec();

    std::vector<LocalIndex> layout{0, 4, 4, 7, 12, 15};
    std::vector<uint8_t> ptype{0, 2, 0, 0, 1, 0, 0, 3, 0, 0, 0, 1, 0, 0, 5};

    DeviceVector<LocalIndex> d_layout = layout;
    DeviceVector<uint8_t> d_ptype     = ptype;

    DeviceVector<LocalIndex> fluidLayoutAcc;
    std::vector<LocalIndex> fluidLayoutHost;
    computeFluidLayout(exec, std::span<const LocalIndex>(rawPtr(d_layout), d_layout.size()), rawPtr(d_ptype),
                       fluidLayoutAcc, fluidLayoutHost, 1.0);

    std::vector<LocalIndex> ref{0, 3, 3, 5, 8, 10};
    EXPECT_EQ(fluidLayoutHost, ref);
    EXPECT_EQ(toHost(fluidLayoutAcc), ref);
    EXPECT_EQ(fluidLayoutHost.back(), 10); // nFluid < bufSize padding case
    EXPECT_EQ(fluidLayoutHost[1], fluidLayoutHost[2]);
}

TEST(FluidLayoutGpu, allFluidEqualsLayout)
{
    StreamHolder stream;
    auto exec = stream.exec();

    std::vector<LocalIndex> layout{0, 2, 5, 5, 9};
    std::vector<uint8_t> ptype(9, 0);

    DeviceVector<LocalIndex> d_layout = layout;
    DeviceVector<uint8_t> d_ptype     = ptype;

    DeviceVector<LocalIndex> fluidLayoutAcc;
    std::vector<LocalIndex> fluidLayoutHost;
    computeFluidLayout(exec, std::span<const LocalIndex>(rawPtr(d_layout), d_layout.size()), rawPtr(d_ptype),
                       fluidLayoutAcc, fluidLayoutHost, 1.0);

    EXPECT_EQ(fluidLayoutHost, layout);
    EXPECT_EQ(toHost(fluidLayoutAcc), layout);
}

TEST(FillMaskedGpu, basic)
{
    StreamHolder stream;
    auto exec = stream.exec();

    std::vector<double> v{1., 2., 3., 4., 5., 6.};
    std::vector<uint8_t> mask{0, 1, 0, 0, 2, 1};

    DeviceVector<double> d_v    = v;
    DeviceVector<uint8_t> d_mask = mask;

    fillMasked(exec, rawPtr(d_v), rawPtr(d_v) + d_v.size(), rawPtr(d_mask), -7.);
    syncGpu(exec);

    std::vector<double> ref{1., -7., 3., 4., -7., -7.};
    EXPECT_EQ(toHost(d_v), ref);
}

TEST(FillMaskedGpu, emptyRange)
{
    StreamHolder stream;
    auto exec = stream.exec();

    std::vector<double> v{1., 2., 3.};
    std::vector<uint8_t> mask{1, 1, 1};

    DeviceVector<double> d_v     = v;
    DeviceVector<uint8_t> d_mask = mask;

    fillMasked(exec, rawPtr(d_v), rawPtr(d_v), rawPtr(d_mask), -7.);
    syncGpu(exec);
    EXPECT_EQ(toHost(d_v), std::vector<double>({1., 2., 3.}));
}
