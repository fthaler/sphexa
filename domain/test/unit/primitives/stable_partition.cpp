/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file @brief Tests for the fused stable partition, the fluid leaf layout helper and the masked fill
 */

#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "cstone/primitives/primitives_acc.hpp"
#include "stable_partition_checks.hpp"

using namespace cstone;

namespace
{

//! @brief run the gather on the CPU and verify it (fluid elements at [0:nFluid) in original order)
void checkGather(const std::vector<uint8_t>& ptype, const std::vector<LocalIndex>& ordering, LocalIndex nFluid)
{
    execution::Cpu exec;
    LocalIndex n = LocalIndex(ptype.size());

    std::vector<double> src(n), dst(n);
    for (LocalIndex i = 0; i < n; ++i) { src[i] = 0.5 * i; }
    gather(exec, std::span<const LocalIndex>(ordering), src.data(), dst.data());

    checkGatherResult(ptype, src, dst, ordering, nFluid);
}

} // namespace

TEST(StablePartition, edgeCases)
{
    execution::Cpu exec;
    unsigned workspace[16];

    {
        std::vector<uint8_t> ptype{0, 0, 0}; // all fluid
        std::vector<LocalIndex> ordering(3, LocalIndex(-1)), orderingInv(3, LocalIndex(-1));
        stablePartition(exec, ptype.data(), 3, 3, ordering.data(), orderingInv.data(), workspace);
        checkPermutations(ptype, ordering, orderingInv, 3);
        checkStability(ptype, ordering, 3);
        EXPECT_EQ(ordering, (std::vector<LocalIndex>{0, 1, 2}));
        EXPECT_EQ(orderingInv, (std::vector<LocalIndex>{0, 1, 2}));
    }
    {
        std::vector<uint8_t> ptype{1, 2, 3}; // all non-fluid
        std::vector<LocalIndex> ordering(3, LocalIndex(-1)), orderingInv(3, LocalIndex(-1));
        stablePartition(exec, ptype.data(), 3, 0, ordering.data(), orderingInv.data(), workspace);
        checkPermutations(ptype, ordering, orderingInv, 0);
        checkStability(ptype, ordering, 0);
        EXPECT_EQ(ordering, (std::vector<LocalIndex>{0, 1, 2}));
        EXPECT_EQ(orderingInv, (std::vector<LocalIndex>{0, 1, 2}));
    }
    {
        std::vector<uint8_t> ptype{1, 1, 0, 1}; // single fluid particle
        std::vector<LocalIndex> ordering(4, LocalIndex(-1)), orderingInv(4, LocalIndex(-1));
        stablePartition(exec, ptype.data(), 4, 1, ordering.data(), orderingInv.data(), workspace);
        checkPermutations(ptype, ordering, orderingInv, 1);
        checkStability(ptype, ordering, 1);
        EXPECT_EQ(ordering, (std::vector<LocalIndex>{2, 0, 1, 3}));
        EXPECT_EQ(orderingInv, (std::vector<LocalIndex>{1, 2, 0, 3}));
    }
    {
        std::vector<uint8_t> ptype{0, 1, 0, 1, 0, 1}; // alternating
        std::vector<LocalIndex> ordering(6, LocalIndex(-1)), orderingInv(6, LocalIndex(-1));
        stablePartition(exec, ptype.data(), 6, 3, ordering.data(), orderingInv.data(), workspace);
        checkPermutations(ptype, ordering, orderingInv, 3);
        checkStability(ptype, ordering, 3);
        EXPECT_EQ(ordering, (std::vector<LocalIndex>{0, 2, 4, 1, 3, 5}));
        EXPECT_EQ(orderingInv, (std::vector<LocalIndex>{0, 3, 1, 4, 2, 5}));
        checkGather(ptype, ordering, 3);
    }
}

TEST(StablePartition, randomized)
{
    execution::Cpu exec;
    std::mt19937 gen(42);

    for (int rep = 0; rep < 50; ++rep)
    {
        LocalIndex n = LocalIndex(1 + rep * 7);
        std::vector<uint8_t> ptype;
        LocalIndex nFluid = randomFlags(gen, n, ptype);

        std::vector<LocalIndex> ordering(n), orderingInv(n);
        std::vector<unsigned> workspace(n);
        stablePartition(exec, ptype.data(), n, nFluid, ordering.data(), orderingInv.data(), workspace.data());

        checkPermutations(ptype, ordering, orderingInv, nFluid);
        checkStability(ptype, ordering, nFluid);
        checkGather(ptype, ordering, nFluid);
    }
}

namespace
{

//! @brief brute-force reference for the per-leaf fluid offsets
std::vector<LocalIndex> referenceFluidLayout(const std::vector<LocalIndex>& layout, const std::vector<uint8_t>& ptype)
{
    std::vector<LocalIndex> fluidLayout(layout.size());
    LocalIndex fluidCount = 0;
    for (std::size_t i = 0; i + 1 < layout.size(); ++i)
    {
        fluidLayout[i] = fluidCount;
        for (LocalIndex j = layout[i]; j < layout[i + 1]; ++j)
        {
            if (ptype[j] == 0) { ++fluidCount; }
        }
    }
    fluidLayout.back() = fluidCount;
    return fluidLayout;
}

} // namespace

TEST(FluidLayout, bruteForce)
{
    execution::Cpu exec;
    // layout over a buffer of 15 particles with 5 leaves, one of which is empty (zero-count leaf)
    std::vector<LocalIndex> layout{0, 4, 4, 7, 12, 15};
    std::vector<uint8_t> ptype{0, 2, 0, 0, 1, 0, 0, 3, 0, 0, 0, 1, 0, 0, 5};
    // fluid counts per leaf: 3, -, 2, 3, 2 -> exclusive scan {0, 3, 3, 5, 8, 10}
    LocalIndex nFluid = 10;
    ASSERT_LT(nFluid, ptype.size()); // padding case: nFluid < bufSize

    std::vector<LocalIndex> fluidLayoutHost, fluidLayoutAcc;
    computeFluidLayout(exec, layout, ptype.data(), fluidLayoutAcc, fluidLayoutHost, 1.0);

    EXPECT_EQ(fluidLayoutHost, referenceFluidLayout(layout, ptype));
    EXPECT_EQ(fluidLayoutAcc, fluidLayoutHost);
    EXPECT_EQ(fluidLayoutHost.back(), nFluid);
    EXPECT_EQ(fluidLayoutHost[1], fluidLayoutHost[2]); // empty leaf -> empty fluid range
}

TEST(FluidLayout, allFluidEqualsLayout)
{
    execution::Cpu exec;
    std::vector<LocalIndex> layout{0, 2, 5, 5, 9};
    std::vector<uint8_t> ptype(9, 0);

    std::vector<LocalIndex> fluidLayoutHost, fluidLayoutAcc;
    computeFluidLayout(exec, layout, ptype.data(), fluidLayoutAcc, fluidLayoutHost, 1.0);

    EXPECT_EQ(fluidLayoutHost, layout);
    EXPECT_EQ(fluidLayoutAcc, layout);
}

TEST(FillMasked, basic)
{
    execution::Cpu exec;
    std::vector<double> v{1., 2., 3., 4., 5., 6.};
    std::vector<uint8_t> mask{0, 1, 0, 0, 2, 1};

    fillMasked(exec, v.begin(), v.end(), mask.begin(), -7.);
    std::vector<double> ref{1., -7., 3., 4., -7., -7.};
    EXPECT_EQ(v, ref);
}

TEST(FillMasked, emptyRange)
{
    execution::Cpu exec;
    std::vector<double> v{1., 2., 3.};
    std::vector<uint8_t> mask{1, 1, 1};

    fillMasked(exec, v.begin(), v.begin(), mask.begin(), -7.);
    fillMasked(exec, v.end(), v.end(), mask.end(), -7.);
    EXPECT_EQ(v, std::vector<double>({1., 2., 3.}));
}

//! @brief mask offset must be independent of the filled range
TEST(FillMasked, subrange)
{
    execution::Cpu exec;
    std::vector<double> v{1., 2., 3., 4., 5., 6.};
    std::vector<uint8_t> mask{0, 0, 1, 0, 1, 0};

    LocalIndex first = 1, last = 5;
    fillMasked(exec, v.begin() + first, v.begin() + last, mask.data() + first, 0.);
    std::vector<double> ref{1., 2., 0., 4., 0., 6.};
    EXPECT_EQ(v, ref);
}
