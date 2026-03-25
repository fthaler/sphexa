/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Focused octree rebalance on GPUs
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <span>

#include "cstone/tree/definitions.h"
#include "cstone/domain/index_ranges.hpp"

namespace cstone
{

template<class KeyType>
extern void rebalanceDecisionEssentialGpu(const KeyType* prefixes,
                                          const TreeNodeIndex* childOffsets,
                                          const TreeNodeIndex* parents,
                                          const unsigned* counts,
                                          const uint8_t* macs,
                                          KeyType focusStart,
                                          KeyType focusEnd,
                                          unsigned bucketSize,
                                          TreeNodeIndex* nodeOps,
                                          TreeNodeIndex numNodes);

/*! @brief compute synthetic counts that will produce a uniform tree with depth @p maxLevel
 *
 * @tparam KeyType
 * @param nodeKeys    warren-salmon SFC keys for each tree cell, including internal
 * @param counts      particle counts of each tree node, length = nodeKeys.size()
 * @param bucketSize  refinement count criterion (Ncrit)
 * @param maxLevel    desired level to refine to
 */
template<class KeyType>
extern void
synthCountsMaxLevelGpu(std::span<const KeyType> nodeKeys, unsigned* counts, unsigned bucketSize, int maxLevel);

template<class KeyType>
extern bool protectAncestorsGpu(const KeyType*, const TreeNodeIndex*, TreeNodeIndex*, TreeNodeIndex);

template<class KeyType>
extern ResolutionStatus enforceKeysGpu(const KeyType* forcedKeys,
                                       TreeNodeIndex numForcedKeys,
                                       const KeyType* nodeKeys,
                                       const TreeNodeIndex* childOffsets,
                                       const TreeNodeIndex* parents,
                                       TreeNodeIndex* nodeOps);

//! @brief see CPU version
template<class KeyType>
extern void rangeCountGpu(std::span<const KeyType> leaves,
                          std::span<const unsigned> counts,
                          std::span<const KeyType> leavesFocus,
                          std::span<const TreeNodeIndex> leavesFocusIdx,
                          std::span<unsigned> countsFocus);

} // namespace cstone