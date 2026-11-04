/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  CPU/GPU wrappers for basic algorithms
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <span>

#include "cstone/execution.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "gather.hpp"

namespace cstone
{

template<class It, class T>
void fill(execution::Cpu, It first, It last, T value)
{
    if (last <= first) { return; }
    std::fill(first, last, value);
}

//! @brief fill @p value into [first:last) where the mask is non-zero, no-op for empty ranges
template<class It, class MaskIt, class T>
void fillMasked(execution::Cpu, It first, It last, MaskIt maskFirst, T value)
{
    if (last <= first) { return; }
    for (; first != last; ++first, ++maskFirst)
    {
        if (*maskFirst != 0) { *first = value; }
    }
}

template<class T>
void copy_n(execution::Cpu, const T* src, std::size_t n, T* dest)
{
    omp_copy(src, src + n, dest);
}

template<class T1, class T2, class T3>
void scale(execution::Cpu, const T1* in1, const T1* in2, T2* out, T3 value)
{
    std::transform(in1, in2, out, [value](auto v_) { return v_ * value; });
}

template<class IndexType, class ValueType>
void gather(execution::Cpu, std::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
    gather(ordering, source, destination);
}

template<class IndexType, class ValueType>
void scatter(execution::Cpu, std::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
    scatter(ordering, source, destination);
}

template<class T>
std::tuple<T, T> minMax(execution::Cpu, const T* first, const T* last)
{
    assert(last >= first);

    T minimum = INFINITY;
    T maximum = -INFINITY;

#pragma omp parallel for reduction(min : minimum) reduction(max : maximum)
    for (size_t pi = 0; pi < std::size_t(last - first); pi++)
    {
        T value = first[pi];
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }

    return std::make_tuple(minimum, maximum);
}

//! @brief sortByKey with temp buffer management
template<class KeyType, class ValueType, class KeyBuf, class ValueBuf>
void sortByKey(execution::Cpu,
               std::span<KeyType> keys,
               std::span<ValueType> values,
               KeyBuf& /*keyBuf*/,
               ValueBuf& /*valueBuf*/,
               float /*growthRate*/)
{
    assert(keys.size() == values.size());
    sort_by_key(keys.begin(), keys.end(), values.begin());
}

template<class T1, class T2>
void sequence(execution::Cpu, T1* first, T1* last, T2 value)
{
    std::iota(first, last, value);
}

template<class BufferType>
void sequence(execution::Cpu exec, LocalIndex first, LocalIndex n, BufferType& buffer, double growthRate)
{
    reallocateBytes(buffer, sizeof(LocalIndex) * (first + n), growthRate);
    auto* seq = reinterpret_cast<LocalIndex*>(buffer.data());
    sequence(exec, seq + first, seq + first + n, first);
}

/*! @brief stable partition of [0:n) into fluid (ptype == 0) and non-fluid (ptype != 0) elements
 *
 * Writes both permutations of the partition in a single pass:
 *   - @p ordering[j] is the source index of the j-th element in the partitioned order
 *     (fluid elements compacted to [0:nFluid) in their original order, non-fluid elements
 *     appended at [nFluid:n) in their original order)
 *   - @p orderingInv[i] is the destination of element i, i.e. orderingInv[ordering[j]] == j
 *
 * @param[in]  ptype        element type flags, 0 = fluid, non-zero = non-fluid
 * @param[in]  n            number of elements, including halos
 * @param[in]  nFluid       number of fluid elements, i.e. the number of ptype == 0 elements in [0:n)
 * @param[out] ordering     destination ordering, length n
 * @param[out] orderingInv  inverse of @p ordering, length n
 * @param[-]   workspace    scratch of length >= n, only used by the GPU path
 */
inline void stablePartition(execution::Cpu,
                            const uint8_t* ptype,
                            LocalIndex n,
                            LocalIndex nFluid,
                            LocalIndex* ordering,
                            LocalIndex* orderingInv,
                            unsigned* /*workspace*/)
{
    LocalIndex fluidCount = 0, nonFluidCount = 0;
    for (LocalIndex i = 0; i < n; ++i)
    {
        if (ptype[i] == 0) { ordering[fluidCount++] = i; }
        else { ordering[nFluid + nonFluidCount++] = i; }
    }
    assert(fluidCount == nFluid);

    for (LocalIndex j = 0; j < n; ++j)
    {
        orderingInv[ordering[j]] = j;
    }
}

/*! @brief compute the fluid particle offset of each focus-tree leaf
 *
 * Per-leaf count of elements with ptype == 0 over the leaf ranges [layout[i]:layout[i+1]) followed by
 * an exclusive scan, such that fluidLayout[i] is the number of fluid particles in all leaves before leaf i
 * and fluidLayout.back() is the total number of fluid particles in the buffer.
 *
 * @param[in]    exec             execution policy
 * @param[in]    layout           particle offsets per leaf cell, length numLeafNodes + 1
 * @param[in]    ptype            element type flags over the full particle buffer, 0 = fluid
 * @param[out]   fluidLayoutAcc   leaf fluid offsets in the memory space of @p exec, reallocated to
 *                                numLeafNodes + 1 on every call
 * @param[out]   fluidLayoutHost  host copy of @p fluidLayoutAcc, reallocated to numLeafNodes + 1 on every call
 * @param[in]    growthRate       buffer growth rate used for (re)allocation
 */
template<class AccVector, class HostVector>
void computeFluidLayout(execution::Cpu,
                        std::span<const LocalIndex> layout,
                        const uint8_t* ptype,
                        AccVector& fluidLayoutAcc,
                        HostVector& fluidLayoutHost,
                        double growthRate)
{
    TreeNodeIndex numLeafNodes = TreeNodeIndex(layout.size()) - 1;
    assert(numLeafNodes >= 0);

    reallocate(fluidLayoutHost, numLeafNodes + 1, growthRate);
    reallocate(fluidLayoutAcc, numLeafNodes + 1, growthRate);

    LocalIndex fluidCount = 0;
    for (TreeNodeIndex i = 0; i < numLeafNodes; ++i)
    {
        fluidLayoutHost[i] = fluidCount;
        for (LocalIndex j = layout[i]; j < layout[i + 1]; ++j)
        {
            if (ptype[j] == 0) { ++fluidCount; }
        }
    }
    fluidLayoutHost[numLeafNodes] = fluidCount;

    std::copy(fluidLayoutHost.begin(), fluidLayoutHost.end(), fluidLayoutAcc.begin());
}

} // namespace cstone
