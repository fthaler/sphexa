/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file @brief Host-side verification helpers shared by the CPU and GPU stablePartition tests
 */
#pragma once

#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "cstone/primitives/primitives_acc.hpp"

namespace cstone
{

//! @brief check both permutations for consistency: written everywhere, inverse of each other
inline void checkPermutations(const std::vector<uint8_t>& ptype, const std::vector<LocalIndex>& ordering,
                              const std::vector<LocalIndex>& orderingInv, LocalIndex nFluid)
{
    LocalIndex n = LocalIndex(ptype.size());

    ASSERT_EQ(ordering.size(), n);
    ASSERT_EQ(orderingInv.size(), n);

    std::vector<bool> seenSrc(n, false);
    for (LocalIndex j = 0; j < n; ++j)
    {
        ASSERT_LT(ordering[j], n);
        EXPECT_FALSE(seenSrc[ordering[j]]) << "ordering not a permutation at " << j;
        seenSrc[ordering[j]] = true;
        EXPECT_EQ(orderingInv[ordering[j]], j);
        EXPECT_EQ(ptype[ordering[j]] == 0, j < nFluid); // fluid compacted into [0:nFluid)
    }
}

//! @brief check stability: fluid and non-fluid elements keep their relative order
inline void checkStability(const std::vector<uint8_t>& ptype, const std::vector<LocalIndex>& ordering,
                           LocalIndex nFluid)
{
    LocalIndex n = LocalIndex(ptype.size());

    std::vector<LocalIndex> fluid, nonFluid;
    for (LocalIndex i = 0; i < n; ++i)
    {
        if (ptype[i] == 0) { fluid.push_back(i); }
        else { nonFluid.push_back(i); }
    }

    for (LocalIndex j = 0; j < nFluid; ++j) { EXPECT_EQ(ordering[j], fluid[j]); }
    for (LocalIndex j = nFluid; j < n; ++j) { EXPECT_EQ(ordering[j], nonFluid[j - nFluid]); }
}

//! @brief host-side gather verification: dst[i] == src[ordering[i]] and fluid elements land
//!        at [0:nFluid) in original order
inline void checkGatherResult(const std::vector<uint8_t>& ptype, const std::vector<double>& src,
                              const std::vector<double>& dst, const std::vector<LocalIndex>& ordering,
                              LocalIndex nFluid)
{
    ASSERT_EQ(src.size(), ptype.size());
    ASSERT_EQ(dst.size(), ptype.size());

    for (std::size_t i = 0; i < dst.size(); ++i) { EXPECT_EQ(dst[i], src[ordering[i]]); }

    std::vector<double> fluidRef;
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        if (ptype[i] == 0) { fluidRef.push_back(src[i]); }
    }
    EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + nFluid), fluidRef);
}

//! @brief random ptype flags with 25% fluid probability, returns the number of fluid elements
inline LocalIndex randomFlags(std::mt19937& gen, LocalIndex n, std::vector<uint8_t>& ptype)
{
    std::uniform_int_distribution<int> flags(0, 3);
    ptype.resize(n);

    LocalIndex nFluid = 0;
    for (LocalIndex i = 0; i < n; ++i)
    {
        ptype[i] = uint8_t(flags(gen));
        nFluid += LocalIndex(ptype[i] == 0);
    }
    return nFluid;
}

} // namespace cstone
