/*
 * Cornerstone octree
 *
 * Copyright (c) 2026 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Utility functions for fixed domain
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <numeric>
#include <span>
#include <vector>

#include "cstone/sfc/sfc.hpp"
#include "cstone/util/reallocate.hpp"

namespace cstone
{
/*! @brief Compute particles keys according to provided cell membership
 *
 * @param[in]  globBox            global coordinate bounding box
 * @param[in]  numAtomsInCells    number of atoms in local cell, sorted by x < y < z center coordinates, on host,
 * @param[in]  numCells_xyz       number of cells in the local grid in each dimension
 * @param[in]  gridLowerCorner    local grid lower corner
 * @param[in]  gridUpperCorner    local grid upper corner
 * @param[out] keysOut            output SFC particles keys
 *
 * Particles are simply assigned the cell key in which they reside (instead of the SFC key computed from coordinates)
 * This could be done on the GPU, but not clear it would be much faster
 */
template<class T, class KeyVec>
void computeKeysFromCellAssignment(const Box<T>& globBox,
                                   const int* numAtomsInCells,
                                   Vec3<int> numCells_xyz,
                                   Vec3<T> gridLowerCorner,
                                   Vec3<T> gridUpperCorner,
                                   KeyVec& keysOut)
{
    using KeyType = KeyVec::value_type;

    int numCells = numCells_xyz[0] * numCells_xyz[1] * numCells_xyz[2];

    std::vector<int> cellOffsets(numCells + 1, 0);
    std::inclusive_scan(numAtomsInCells, numAtomsInCells + numCells, cellOffsets.begin() + 1);

    auto cellDx = gridUpperCorner - gridLowerCorner;
    cellDx[0] /= T(numCells_xyz[0]);
    cellDx[1] /= T(numCells_xyz[1]);
    cellDx[2] /= T(numCells_xyz[2]);

    reallocate(cellOffsets.back(), 1.5, keysOut);
    std::vector<KeyType> keys(keysOut.size());

#pragma omp parallel schedule(static)
    for (int i = 0; i < numCells; ++i) // all GROMACS grid (leaf) cells in local domain
    {
        int czi = i % numCells_xyz[2];
        int cyi = (i / numCells_xyz[2]) % numCells_xyz[1];
        int cxi = i / (numCells_xyz[2] * numCells_xyz[1]);

        Vec3<T> cellCenter = gridLowerCorner + Vec3<T>{cellDx[0] * cxi, cellDx[1] * cyi, cellDx[2] * czi} + cellDx *
                             T(0.5);
        KeyType cellKey = cstone::sfc3D<SfcKind<KeyType>>(cellCenter[0], cellCenter[1], cellCenter[2], globBox);
        std::fill(keys.begin() + cellOffsets[i], keys.begin() + cellOffsets[i + 1], cellKey);
    }
    keysOut = keys; // upload to GPU, if active
}

} // namespace cstone
