/*
 * Cornerstone octree
 *
 * Copyright (c) 2026 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief A domain class with caller-provided SFC boundaries. SFC sorts local particles and builds an LET.
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/domain/assignment.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/focus/octree_focus_mpi.hpp"
#include "cstone/halos/halos.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/box_mpi.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/sfc/sfc_gpu.h"
#include "cstone/util/reallocate.hpp"
#include "cstone/util/type_list.hpp"

namespace cstone
{

template<class KeyType, class T, class Accelerator = CpuTag>
class DomainFixed
{
    static_assert(std::is_unsigned<KeyType>{}, "SFC key type needs to be an unsigned integer\n");

    //! @brief A vector template that resides on the hardware specified as Accelerator
    template<class ValueType>
    using AccVector = std::conditional_t<HaveGpu<Accelerator>{}, DeviceVector<ValueType>, std::vector<ValueType>>;

    constexpr static bool useGpu = HaveGpu<Accelerator>{};

public:
    //! @brief floating point type used for the coordinate bounding box
    using RealType = T;

    //! @brief construct empty Domain
    DomainFixed()
        : focusTree_(0, 0, 0, MPI_COMM_NULL)
        , halos_(0, MPI_COMM_NULL)
    {
    }

    /*! @brief set the SFC domain decomposition boundaries and update the LET structure
     *
     * @param[in] boundaries     SFC boundary start of each rank
     * @param[in] box            global coordinate bounding
     * @param[in] maxLevel       tree level to refine down to in the local part of the LET
     * @param[in] theta          refinement angle for the non-local LET parts
     * @param[in] comm           MPI communicator where rank i starts at boundaries[i]
     * @param[-]  scratch        buffer temporary scratch usage
     *
     * We assume this is called only at the start of a simulation
     */
    template<class DevVec>
    void setBoundaries(std::span<const KeyType> boundaries,
                       const Box<T>& box,
                       int maxLevel,
                       float theta,
                       MPI_Comm comm,
                       DevVec& scratch)
    {
        MPI_Comm_size(comm, &numRanks_);

        std::vector<KeyType> domainBoundaries(numRanks_);
        std::copy(boundaries.begin(), boundaries.end(), domainBoundaries.begin());
        std::vector<int> sfcSegmentToRank(numRanks_);
        std::iota(sfcSegmentToRank.begin(), sfcSegmentToRank.end(), 0);
        sort_by_key(domainBoundaries.begin(), domainBoundaries.end(), sfcSegmentToRank.begin());
        domainBoundaries.push_back(nodeRange<KeyType>(0));

        {
            // invert sfcSegmentToRank
            std::vector<int> rankToSfcSegment(numRanks_);
            std::iota(rankToSfcSegment.begin(), rankToSfcSegment.end(), 0);
            sort_by_key(sfcSegmentToRank.begin(), sfcSegmentToRank.end(), rankToSfcSegment.begin());

            int callerRank;
            MPI_Comm_rank(comm, &callerRank);
            MPI_Comm_split(comm, 0, rankToSfcSegment[callerRank], &comm_);
            myRank_ = rankToSfcSegment[callerRank];
        }

        box_ = box;

        /*******************************/
        // Global tree, a low-res tree that resolves the domain boundaries, can be only 1 cell per rank

        globalLeaves_    = computeSpanningTree<KeyType>(domainBoundaries);
        globalLeavesAcc_ = globalLeaves_; // Upload to GPU if active
        reallocate(nNodes(globalLeaves_), allocGrowthRate_, globalLeafCountsAcc_);

        // CPU/GPU linked tree for globalLeaves_
        globalOctreeAcc_.resize(nNodes(globalLeavesAcc_));
        if constexpr (useGpu) { buildOctreeGpu(globalLeavesAcc_.data(), globalOctreeAcc_.data()); }
        else { updateInternalTree<KeyType>(globalLeaves_, globalOctreeAcc_.data()); }

        std::vector<unsigned> dummyCounts(nNodes(globalLeaves_), 1);
        assignment_ = makeSfcAssignment(numRanks_, dummyCounts, globalLeaves_.data());

        /*******************************/
        // LET structure build

        unsigned bucketSizeFocus = 1; // dummy value
        FocusedOctree<KeyType, T, Accelerator> focusTree(myRank_, numRanks_, bucketSizeFocus, comm_);

        auto invThetaEff = invThetaMinMac(theta);
        focusTree.convergeToLevel(box, assignment_, globalLeavesAcc_, invThetaEff, maxLevel, scratch);
        focusTree_ = std::move(focusTree);

        // Mark all cells 2-layers out as halos
        focusTree_.discoverAdjacent(3.01, scratch, false);
        reallocate(focusTree_.octreeViewAcc().numLeafNodes + 1, allocGrowthRate_, layoutAcc_, layout_);

        halos_ = Halos<KeyType, Accelerator>(myRank_, comm_);
        downloadTreeToHost();
    }

    /*! @brief Call on DD steps.
     *
     *        Preconditions: - keys,x,y,z,q must have equal size. sfcOrder will be resized
     *                       - setBoundaries() has been called
     *        Does: - Particle SFC sort
     *              - LET local particle indexing, i.e. tree cell offsets and particle counts
     *        Does not do: - communication
     */
    template<class KeyVec, class VectorX, class VectorI, class... Vectors>
    void sync(KeyVec& keys,
              VectorX& x,
              VectorX& y,
              VectorX& z,
              VectorX& q,
              VectorI& sfcOrder,
              std::tuple<Vectors&...> scratchBuffers)
    {
        staticChecks<KeyVec, VectorX, Vectors...>(scratchBuffers);
        checkSizesEqual(x.size(), keys, x, y, z, q);
        LocalIndex numParticles = x.size();
        bufDesc_ = {0, numParticles, numParticles};
        lowMemReallocate(numParticles, allocGrowthRate_, {}, scratchBuffers);

        // Compute and sort SFC keys
        computeSfcKeys<useGpu>(x.data(), y.data(), z.data(), sfcKindPointer(keys.data()), x.size(), box_);
        sequence<useGpu>(startIndex(), nParticles(), sfcOrder, allocGrowthRate_);
        std::span<KeyType> keyView(keys.data() + startIndex(), nParticles());
        sortByKey<useGpu>(keyView, std::span{sfcOrder.data() + startIndex(), nParticles()}, get<0>(scratchBuffers),
                          get<1>(scratchBuffers), allocGrowthRate_);

        // assert particles are in local subdomain
        {
            KeyType minKey, maxKey;
            if constexpr (useGpu)
            {
                memcpyD2H(keyView.data(), 1, &minKey);
                memcpyD2H(keyView.data() + keyView.size() - 1, 1, &maxKey);
            }
            else
            {
                minKey = keyView.front();
                maxKey = keyView.back();
            }
            if (minKey < assignment_[myRank_] || maxKey >= assignment_[myRank_ + 1])
            {
                throw std::runtime_error("keys not in local subdomain\n");
            }
        }

        // reorder particles to SFC order
        gatherArrays({sfcOrder.data(), nParticles()}, 0, std::tie(x, y, z, q), scratchBuffers);

        // compute local node particle counts and node particle layout
        focusTree_.computeLocalCounts(keyView);

        fill<useGpu>(layoutAcc_.begin(), layoutAcc_.end(), 0);
        auto leafCounts    = focusTree_.leafCountsAcc();
        auto letLocalRange = focusTree_.assignment()[myRank_];

        if constexpr (useGpu)
        {
            exclusiveScanGpu(leafCounts.data() + letLocalRange.start(), leafCounts.data() + letLocalRange.end(),
                             layoutAcc_.data() + letLocalRange.start(), LocalIndex(0));
        }
        else
        {
            std::exclusive_scan(leafCounts.begin() + letLocalRange.start(), leafCounts.begin() + letLocalRange.end(),
                                layoutAcc_.begin() + letLocalRange.start(), LocalIndex(0));
        }
        fill<useGpu>(layoutAcc_.begin() + letLocalRange.end(), layoutAcc_.end(), numParticles);

        if constexpr (useGpu) { memcpyD2H(layoutAcc_.data(). layoutAcc_.size(), layout_.data()); }
    }

    /*! @brief Call on DD steps.
     *
     *        Preconditions: - keys,x,y,z,q must have equal size. sfcOrder will be resized
     *                       - setBoundaries() has been called
     *                       - x,y,z coordinates are within local domain, starting at index 0
     *        Does: - Particle SFC sort
     *              - LET local particle indexing, i.e. tree cell offsets and particle counts
     */
    template<class KeyVec, class VectorX, class VectorI, class... Vectors>
    void syncWithHalos(KeyVec& keys,
                       VectorX& x,
                       VectorX& y,
                       VectorX& z,
                       VectorX& q,
                       VectorI& sfcOrder,
                       std::tuple<Vectors&...> scratch)
    {
        staticChecks<KeyVec, VectorX, Vectors...>(scratch);
        checkSizesEqual(x.size(), keys, x, y, z, q);
        LocalIndex numParticles = x.size();
        bufDesc_                = {0, numParticles, numParticles};
        lowMemReallocate(numParticles, allocGrowthRate_, {}, scratch);

        // Compute and sort SFC keys
        computeSfcKeys<useGpu>(x.data(), y.data(), z.data(), sfcKindPointer(keys.data()), x.size(), box_);
        sequence<useGpu>(startIndex(), nParticles(), sfcOrder, allocGrowthRate_);
        std::span<KeyType> keyView(keys.data() + startIndex(), nParticles());
        sortByKey<useGpu>(keyView, std::span{sfcOrder.data() + startIndex(), nParticles()}, get<0>(scratch),
                          get<1>(scratch), allocGrowthRate_);

        // assert particles are in local subdomain
        {
            KeyType minKey, maxKey;
            if constexpr (useGpu)
            {
                memcpyD2H(keyView.data(), 1, &minKey);
                memcpyD2H(keyView.data() + keyView.size() - 1, 1, &maxKey);
            }
            else
            {
                minKey = keyView.front();
                maxKey = keyView.back();
            }
            if (minKey < assignment_[myRank_] || maxKey >= assignment_[myRank_ + 1])
            {
                throw std::runtime_error("keys not in local subdomain\n");
            }
        }

        // compute node counts of the global tree
        if constexpr (useGpu)
        {
            computeNodeCountsGpu(globalLeavesAcc_.data(), globalLeafCountsAcc_.data(), nNodes(globalLeavesAcc_),
                                 {keyView.data(), keyView.size()}, std::numeric_limits<unsigned>::max(), false);
        }
        else
        {
            computeNodeCounts(globalLeavesAcc_.data(), globalLeafCountsAcc_.data(), nNodes(globalLeavesAcc_),
                              {keyView.data(), keyView.size()}, std::numeric_limits<unsigned>::max(), false);
        }
        // assumes all ranks have the same number of global nodes (otherwise would have to use allgatherv)
        mpiAllgatherGpuDirect<useGpu>((unsigned*)MPI_IN_PLACE, globalLeafCountsAcc_.data(),
                                      assignment_.numNodesPerRank()[myRank_], comm_);

        // compute LET counts
        focusTree_.updateCounts(keyView, {globalLeavesAcc_.data(), globalLeavesAcc_.size()},
                                {globalLeafCountsAcc_.data(), globalLeafCountsAcc_.size()}, std::get<0>(scratch));

        focusTree_.computeLayout({rawPtr(layoutAcc_), layoutAcc_.size()}, layout_);
        if constexpr (not useGpu) { layoutAcc_ = layout_; }

        auto myRange = focusTree_.assignment()[myRank_];
        bufDesc_ = {layout_[myRange.start()], layout_[myRange.end()], layout_.back()};

        // We now know which cells have how many halos. Make space for them
        lowMemReallocate(bufDesc_.size, allocGrowthRate_, std::tie(x, y, z, q), scratch);

        // reorder particles to SFC order and place them behind halos with lower SFC keys
        gatherArrays({sfcOrder.data(), nParticles()}, bufDesc_.start, std::tie(x, y, z, q), scratch);

        halos_.exchangeRequests(focusTree_.treeLeaves(), focusTree_.assignment(), layout_);
    }

    //! @brief repeat the halo exchange pattern from the previous sync operation for a different set of arrays
    template<class... Vectors, class SendBuffer, class ReceiveBuffer>
    void exchangeHalos(std::tuple<Vectors&...> arrays, SendBuffer& sendBuffer, ReceiveBuffer& receiveBuffer) const
    {
        std::apply([this](auto&... arrays) { this->checkSizesEqual(this->bufDesc_.size, arrays...); }, arrays);
        this->halos_.exchangeHalos(arrays, sendBuffer, receiveBuffer);
    }

    //! @brief return the index of the first particle that's part of the local assignment
    [[nodiscard]] LocalIndex startIndex() const { return bufDesc_.start; }
    //! @brief return one past the index of the last particle that's part of the local assignment
    [[nodiscard]] LocalIndex endIndex() const { return bufDesc_.end; }
    //! @brief return number of locally assigned particles
    [[nodiscard]] LocalIndex nParticles() const { return endIndex() - startIndex(); }
    //! @brief return number of locally assigned particles plus number of halos
    [[nodiscard]] LocalIndex nParticlesWithHalos() const { return bufDesc_.size; }
    //! @brief read only visibility of the global octree in traversible layout
    OctreeView<const KeyType> globalTree() const
    {
        auto ret   = globalOctreeAcc_.cdata();
        ret.leaves = globalLeavesAcc_.data();
        return ret;
    }
    //! @brief read only visibility of the focused octree
    const FocusedOctree<KeyType, T, Accelerator>& focusTree() const { return focusTree_; }
    //! @brief the index of the first locally assigned cell in focusTree()
    TreeNodeIndex startCell() const { return focusTree_.assignment()[myRank_].start(); }
    //! @brief the index of the last locally assigned cell in focusTree()
    TreeNodeIndex endCell() const { return focusTree_.assignment()[myRank_].end(); }
    //! @brief particle offsets of each focus tree leaf cell
    std::span<const LocalIndex> layout() const { return {rawPtr(layoutAcc_), layoutAcc_.size()}; }
    //! @brief return the coordinate bounding box from the previous sync call
    const Box<T>& box() const { return box_; }

    KeyType assignmentStart() const { return assignment_[myRank_]; }

    void setGrowthAllocRate(float factor) { allocGrowthRate_ = factor; }

    //! @brief Pointers to GPU data if active, CPU otherwise
    OctreeNsView<T, KeyType> octreeProperties() const
    {
        auto ft = focusTree_.octreeViewAcc();
        return {ft.numLeafNodes,
                ft.prefixes,
                ft.childOffsets,
                ft.parents,
                ft.internalToLeaf,
                ft.levelRange,
                focusTree_.treeLeavesAcc().data(),
                rawPtr(layoutAcc_),
                focusTree_.geoCentersAcc().data(),
                focusTree_.geoSizesAcc().data()};
    }

    //! @brief Pointers to CPU data
    OctreeNsView<T, KeyType> octreePropertiesHost() const
    {
        if constexpr (useGpu)
        {
            auto ft = octreeHost_.octreeViewAcc();
            return {ft.numLeafNodes,
                    ft.prefixes,
                    ft.childOffsets,
                    ft.parents,
                    ft.internalToLeaf,
                    ft.levelRange,
                    focusTree_.treeLeaves().data(),
                    layout_.data(),
                    geoCentersHost_.data(),
                    geoSizesHost_.data()};
        }
        else
        {
            return octreeProperties();
        }
    }

private:

    void downloadTreeToHost()
    {
        if constexpr (useGpu)
        {
            downloadFromGpu(octreeHost_, focusTree_.octreeViewAcc());
            auto numNodes = octreeHost_.numNodes;

            reallocate(numNodes, allocGrowthRate_, geoCentersHost_, geoSizesHost_);
            memcpyD2H(focusTree_.geoCentersAcc().data(), numNodes, geoCentersHost_.data());
            memcpyD2H(focusTree_.geoSizesAcc().data(), numNodes, geoSizesHost_.data());
        }
    }

    //! @brief make sure all array sizes are equal to @p value
    template<class... Arrays>
    static void checkSizesEqual(std::size_t value, const Arrays&... arrays)
    {
        std::array<std::size_t, sizeof...(Arrays)> sizes{arrays.size()...};
        bool allEqual = size_t(std::count(begin(sizes), end(sizes), value)) == sizes.size();
        if (!allEqual) { throw std::runtime_error("Domain sync: input array sizes are inconsistent\n"); }
    }

    /*! @brief check type requirements on scratch buffers
     *
     * @tparam KeyVec           type of vector used to store SFC keys
     * @tparam ConservedVectors types of conserved particle field vectors (x,y,z,...)
     * @param  scratchBuffers   a tuple of references to vectors for scratch usage
     *
     * At least 2 scratch buffers are needed. For each value type appearing in the list of conserved
     * vectors, either a scratch buffer with a matching value_type (more efficient due to swaps) or with a value_type
     * of equal or bigger size (less efficient due an additional copy) is needed.
     */
    template<class KeyVec, class... ConservedVectors, class ScratchBuffers>
    void staticChecks(ScratchBuffers& scratchBuffers)
    {
        static_assert(std::is_same_v<typename KeyVec::value_type, KeyType>);
        static_assert(std::tuple_size_v<ScratchBuffers> >= 1);

        auto tup               = scratchBuffers;
        constexpr auto matches = std::make_tuple(util::FindIndex<ConservedVectors&, std::decay_t<decltype(tup)>>{}...);
        constexpr auto smaller =
            std::make_tuple(util::FindIndex<ConservedVectors&, std::decay_t<decltype(tup)>, SmallerElementSize>{}...);

        auto valueTypeCheck = [](auto m, auto s)
        {
            constexpr int numScratchBuffers = std::tuple_size_v<std::decay_t<decltype(tup)>>;
            static_assert(m < numScratchBuffers || s < numScratchBuffers,
                          "one of the conserved fields has a value_type bigger than the value_types of available "
                          "scratch buffers");
        };
        util::for_each_tuple(valueTypeCheck, matches, smaller);
    }

    void diagnostics(size_t assignedSize)
    {
        auto focusAssignment = focusTree_.assignment();
        auto focusTree       = focusTree_.treeLeaves();
        auto flags           = focusTree_.flags();
        auto globalTree      = globalLeaves_;

        TreeNodeIndex numFocusPeers    = 0;
        TreeNodeIndex numFocusTruePeer = 0;
        for (int i = 0; i < numRanks_; ++i)
        {
            if (i != myRank_)
            {
                numFocusPeers += focusAssignment[i].count();
                for (TreeNodeIndex fi = focusAssignment[i].start(); fi < focusAssignment[i].end(); ++fi)
                {
                    KeyType fnstart  = focusTree[fi];
                    KeyType fnend    = focusTree[fi + 1];
                    TreeNodeIndex gi = findNodeAbove(globalTree.data(), globalTree.size(), fnstart);
                    if (!(gi < nNodes(globalTree) && globalTree[gi] == fnstart && globalTree[gi + 1] <= fnend))
                    {
                        numFocusTruePeer++;
                    }
                }
            }
        }

        int numFlags = std::count_if(flags.begin(), flags.end(), [](auto x) { return x > 0; });
        auto fPeerFlags =
            focusPeers<KeyType>({assignment_.data(), size_t(numRanks_ + 1)}, myRank_, globalTree, focusTree);
        std::vector<int> fPeers;
        peerFlagsToList(fPeerFlags, fPeers, PeerMask::focus);

        auto hPeerFlags = haloPeers(myRank_, layout_, focusTree_.assignment());
        std::vector<int> hPeers;
        peerFlagsToList(hPeerFlags, hPeers, PeerMask::halo);

        for (int i = 0; i < numRanks_; ++i)
        {
            if (i == myRank_)
            {
                std::cout << "rank " << i << " " << assignedSize << " " << layout_.back()
                          << " focus h/true/peers/loc/tot: " << numFlags << "/" << numFocusTruePeer << "/"
                          << numFocusPeers << "/" << focusAssignment[myRank_].count() << "/" << flags.size()
                          << " peers: [" << std::max(hPeers.size(), fPeers.size()) << "] ";
                if (numRanks_ <= 64)
                {
                    for (auto r : fPeers)
                    {
                        bool isHalo = std::count(hPeers.begin(), hPeers.end(), r) == 1;
                        if (isHalo) { std::cout << r << " "; }
                        else { std::cout << "*" << r << " "; }
                    }
                    for (auto r : hPeers)
                    {
                        bool isFocus = std::count(fPeers.begin(), fPeers.end(), r) == 1;
                        if (not isFocus) { std::cout << "^" << r << " "; }
                    }
                }
                std::cout << std::endl;
            }
            MPI_Barrier(comm_);
        }
    }

    int myRank_;
    int numRanks_;

    //! @brief MPI communicator for all collective and point-to-point operations
    MPI_Comm comm_;

    //! @brief buffer growth rate when reallocating
    float allocGrowthRate_{1.05};

    /*! @brief description of particle buffers, storing start and end indices of assigned particles and total size
     *
     *  First element: array index of first local particle belonging to the assignment
     *  i.e. the index of the first particle that belongs to this rank and is not a halo
     *  Second element: index (upper bound) of last particle that belongs to the assignment
     */
    BufferDescription bufDesc_{0, 0, 0};

    //! @brief global coordinate bounding box
    Box<T> box_{0, 1};

    //! @brief SFC decomposition data
    std::vector<KeyType> globalLeaves_;
    AccVector<KeyType> globalLeavesAcc_;
    AccVector<unsigned> globalLeafCountsAcc_;
    OctreeData<KeyType, Accelerator> globalOctreeAcc_;
    SfcAssignment<KeyType> assignment_;

    /*! @brief locally focused, fully traversable octree, used for halo discovery and exchange
     *
     * -Uses bucketSizeFocus_ as the maximum particle count per leaf within the focused SFC area.
     * -Outside the focus area, each leaf node with a particle count larger than bucketSizeFocus_
     *  fulfills a MAC with theta as the opening parameter
     * -Also contains particle counts.
     */
    FocusedOctree<KeyType, T, Accelerator> focusTree_;

    //! @brief CPU copy of structural data in focusTree
    OctreeData<KeyType, CpuTag> octreeHost_;
    std::vector<Vec3<T>> geoCentersHost_, geoSizesHost_;

    //! @brief particle offsets of each leaf node in focusedTree_, length = focusedTree_.treeLeaves().size()
    AccVector<LocalIndex> layoutAcc_;
    std::vector<LocalIndex> layout_;

    //! @brief stores particle offsets to perform halo exchanges
    Halos<KeyType, Accelerator> halos_;
};

} // namespace cstone
