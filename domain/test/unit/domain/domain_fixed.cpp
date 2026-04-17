/*
 * Cornerstone octree
 *
 * Copyright (c) 2026 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Testing for fixed domain utilities
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/domain/domain_fixed_util.hpp"

using namespace cstone;

TEST(DomainFixed, keysFromCellAssignment)
{
    using KeyType = uint32_t;
    using T = float;

    Box<T> box(0, 8);

    Vec3<int> numCells_xyz = {2, 2, 2};
    int numCells = numCells_xyz[0] * numCells_xyz[1] * numCells_xyz[2];
    std::vector<int> numAtomsInCell(numCells, 2);

    Vec3<T> gridLower = {4, 4, 4};
    Vec3<T> gridUpper = {8, 8, 8};

    std::vector<Vec3<T>> cellCenters{{5, 5, 5}, {5, 5, 7}, {5, 7, 5}, {5, 7, 7},
                                     {7, 5, 5}, {7, 5, 7}, {7, 7, 5}, {7, 7, 7}};

    std::vector<Vec3<T>> cellKeys(numCells);
    std::vector<KeyType> keysRef(std::accumulate(numAtomsInCell.begin(), numAtomsInCell.end(), 0));
    int offset = 0;
    for (int i = 0; i < numCells; ++i)
    {
        auto cellKey = sfc3D<SfcKind<KeyType>>(cellCenters[i][0], cellCenters[i][1], cellCenters[i][2], box);
        std::fill(keysRef.begin() + offset, keysRef.begin() + offset + numAtomsInCell[i], cellKey) ;
        offset += numAtomsInCell[i];
    }

    std::vector<KeyType> testKeys;
    computeKeysFromCellAssignment(box, numAtomsInCell.data(), numCells_xyz, gridLower, gridUpper, testKeys);
    EXPECT_EQ(testKeys, keysRef);
}
