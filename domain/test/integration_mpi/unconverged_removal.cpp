/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @brief 2-rank tests for ve-pt unconverged-particle removal (first full-propagator MPI test)
 *
 * Drives the real HydroVePtProp through the sphexa.cpp loop order (activateFields -> sync ->
 * computeForces -> integrate -> sync) on a hand-placed mixed IC in an open box:
 * an 8-particle mutually-close fluid clump (each member has 7 fluid neighbors, i.e. nc = 8 ->
 * survives), 3 isolated fluid singletons farther than the capped h-growth reach from any other
 * fluid particle (nc <= 1 -> marked in integrate step 3) and 20 gravity particles (ptype = 1,
 * never marked).
 *
 * Positive case: removeUnconvergedParticles = 1 -> the singletons are removed at the final sync.
 * Checked: the global owned count drops by exactly the doomed count, the surviving id set is
 * exactly clump + gravity, and no removeKey marks remain. A second iteration (computeForces +
 * integrate) proves post-removal stability: no new marks, no convergence throw.
 *
 * Negative case: same IC with the flag unset -> "Neighbor search did not converge" throws in
 * integrate step 3 on exactly the ranks that own a doomed particle. The throw is caught per rank and
 * verified collectively (Allreduces over per-rank flags): integrate throws after applyPartition_
 * (a local gather), so no collective is left dangling when the flags are gathered.
 *
 * Gravity case: the positive IC with d.g != 0, so that sync() takes the syncGrav branch and
 * computeForces runs the multipole traversal — removal must not depend on the halo layout branch.
 *
 * Emptied-rank case: an IC in which every fluid particle is unconverged (16 isolated singletons, no
 * clump, no gravity) -> the removal empties the whole domain and every rank ends with zero owned
 * particles, i.e. the zero-fluid-rank path, reached through the propagator instead of a hand-built
 * layout as in zero_fluid_rank.cpp.
 *
 * On the GPU build the same four scenarios run on the GPU propagator instantiation; the test
 * exercises computeFluidLayout/stablePartition/count(Gpu) and the fluid halo exchange with
 * dedicated scratch buffers.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <mpi.h>
#include <vector>

#ifdef SPH_EXA_PTYPE_GPU_TEST
#define USE_CUDA
#include "cstone/cuda/device_vector.h"
#endif

#include "cstone/domain/domain.hpp"
#include "cstone/primitives/primitives_acc.hpp"

#include "sph/types.hpp"
#include "propagator/ve_hydro_pt.hpp"

using namespace cstone;

namespace
{

//! @brief minimal simulation-data holder for the propagator: SimulationData without the cooling fields
template<class Acc>
struct RemovalSimData
{
    using Exec      = Acc;
    using KeyType   = sph::SphTypes::KeyType;
    using RealType  = sph::SphTypes::CoordinateType;
    using HydroData = sphexa::ParticlesData<Acc>;

    HydroData hydro;
    MPI_Comm  comm{MPI_COMM_WORLD};
};

//! @brief host copy of a field buffer (GPU: D2H, CPU: plain copy)
template<class Acc, class T>
std::vector<T> hostCopy(const typename sphexa::ParticlesData<Acc>::template FieldVector<T>& v)
{
    if constexpr (cstone::execution::HaveGpu<Acc>{}) { return toHost(v); }
    else { return v; }
}

//! @brief IC population layout: [0:8) fluid clump, [8:11) doomed fluid singletons, [11:31) gravity
constexpr int numClump      = 8;
constexpr int numSingletons = 3;
constexpr int numGrav       = 20;
constexpr int numGlobal     = numClump + numSingletons + numGrav;
constexpr int singletonFirst = numClump;

//! @brief hand-placed mixed IC. The clump (2x2x2 grid, spacing 0.04, h = 0.08; search radius
//! 2h = 0.16 covers the 0.069 body diagonal) gives every member 7 fluid neighbors, i.e. nc = 8 > 1.
//! The singletons sit >= 4 away from any other fluid particle with h = 0.05: even the capped h
//! growth (10 iterations, ~8x at ng0 = 8) cannot reach the clump, so they end with nc = 1 and get
//! marked. Gravity particles never take part in the fluid neighbor search (fluid tree view), so
//! their placement does not affect the marking.
template<class T>
void initGlobalIc(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, std::vector<float>& h,
                  std::vector<float>& m, std::vector<uint8_t>& ptype)
{
    x.reserve(numGlobal), y.reserve(numGlobal), z.reserve(numGlobal);
    h.reserve(numGlobal), m.reserve(numGlobal), ptype.reserve(numGlobal);

    // fluid clump: 2x2x2 grid around (5, 5, 5), spacing 0.04
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
            {
                x.push_back(T(5.0) + (i ? T(0.02) : T(-0.02)));
                y.push_back(T(5.0) + (j ? T(0.02) : T(-0.02)));
                z.push_back(T(5.0) + (k ? T(0.02) : T(-0.02)));
                h.push_back(0.08f);
                m.push_back(1.0f);
                ptype.push_back(0);
            }

    // isolated fluid singletons: doomed
    constexpr T singletonPos[numSingletons][3] = {{1, 1, 1}, {9, 1, 1}, {5, 1, 9}};
    for (int i = 0; i < numSingletons; ++i)
    {
        x.push_back(singletonPos[i][0]);
        y.push_back(singletonPos[i][1]);
        z.push_back(singletonPos[i][2]);
        h.push_back(0.05f);
        m.push_back(1.0f);
        ptype.push_back(0);
    }

    // gravity particles: 5x2x2 spread, reasonable mass and softening
    for (int i = 0; i < numGrav; ++i)
    {
        x.push_back(T(0.5) + T(2.0) * T(i % 5));
        y.push_back(T(0.5) + T(4.0) * T((i / 5) % 2));
        z.push_back(T(0.5) + T(4.0) * T(i / 10));
        h.push_back(0.5f);
        m.push_back(1.0f);
        ptype.push_back(1);
    }
}

//! @brief fill the locally-held IC subset (interleaved by rank, as in fluid_halos.cpp) and resize.
//! activateFields must run first so that resize allocates every conserved + dependent field.
template<class Acc, class T>
void initLocalFrom(RemovalSimData<Acc>& simData, int rank, int numRanks, const std::vector<T>& xG, const std::vector<T>& yG,
                   const std::vector<T>& zG, const std::vector<float>& hG, const std::vector<float>& mG,
                   const std::vector<uint8_t>& ptypeG)
{
    using RealType = T;

    std::vector<RealType> x, y, z, temp;
    std::vector<float>    h, m, x_m1, y_m1, z_m1;
    std::vector<uint8_t>  ptype;
    std::vector<uint64_t> id;
    for (int i = rank; i < int(xG.size()); i += numRanks)
    {
        x.push_back(xG[i]);
        y.push_back(yG[i]);
        z.push_back(zG[i]);
        h.push_back(hG[i]);
        m.push_back(mG[i]);
        temp.push_back(RealType(1.0));
        x_m1.push_back(xG[i]); // no prior motion: the KDK drift reads dXn = x - x_m1
        y_m1.push_back(yG[i]);
        z_m1.push_back(zG[i]);
        ptype.push_back(ptypeG[i]);
        id.push_back(uint64_t(i));
    }

    // resize AFTER the subset is built: every active field must end up with the same size for the
    // sync's size check (the subset size differs per rank)
    auto& d = simData.hydro;
    d.resize(x.size());

    d.x     = x;
    d.y     = y;
    d.z     = z;
    d.h     = h;
    d.m     = m;
    d.temp  = temp;
    d.x_m1  = x_m1;
    d.y_m1  = y_m1;
    d.z_m1  = z_m1;
    d.du_m1 = std::vector<float>(x.size(), 0.0f);
    d.alpha = std::vector<float>(x.size(), float(d.alphamin));
    d.ptype = ptype;
    d.id    = id;

    d.numParticlesGlobal = xG.size();
    d.ng0                = 8; // ngmin = 2: the clump never enters the h-growth loop (nc = 8 >= 2)
    d.removeUnconvergedParticles = 1;
}

template<class Acc>
void initLocalIc(RemovalSimData<Acc>& simData, int rank, int numRanks)
{
    using RealType = typename RemovalSimData<Acc>::RealType;

    std::vector<RealType> xG, yG, zG;
    std::vector<float>    hG, mG;
    std::vector<uint8_t>  ptypeG;
    initGlobalIc(xG, yG, zG, hG, mG, ptypeG);
    initLocalFrom(simData, rank, numRanks, xG, yG, zG, hG, mG, ptypeG);
}

//! @brief IC for the emptied-rank scenario: 16 isolated fluid singletons, no clump, no gravity.
//! Mutual spacing >= 2.5, far beyond the capped h-growth reach (h grows ~8x from 0.05 at ng0 = 8, so
//! the search radius stays below 1): every particle ends with nc = 1 and gets marked. 16 > bucketSize
//! (8) forces at least two leaves, so both ranks own particles before the removal.
constexpr int numAllDoomed = 16;

template<class T>
void initGlobalAllDoomedIc(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, std::vector<float>& h,
                           std::vector<float>& m, std::vector<uint8_t>& ptype)
{
    for (int i = 0; i < numAllDoomed; ++i)
    {
        x.push_back(T(1.5) + T(2.5) * T(i % 2));
        y.push_back(T(1.5) + T(5.0) * T((i / 2) % 2));
        z.push_back(T(1.5) + T(2.5) * T(i / 4));
        h.push_back(0.05f);
        m.push_back(1.0f);
        ptype.push_back(0);
    }
}

template<class Acc>
void initAllDoomedIc(RemovalSimData<Acc>& simData, int rank, int numRanks)
{
    using RealType = typename RemovalSimData<Acc>::RealType;

    std::vector<RealType> xG, yG, zG;
    std::vector<float>    hG, mG;
    std::vector<uint8_t>  ptypeG;
    initGlobalAllDoomedIc(xG, yG, zG, hG, mG, ptypeG);
    initLocalFrom(simData, rank, numRanks, xG, yG, zG, hG, mG, ptypeG);
}

//! @brief collect the sorted owned ids of all ranks (Allgatherv over the owned ranges)
template<class Acc>
std::vector<uint64_t> globalOwnedIds(const RemovalSimData<Acc>& simData,
                                     const Domain<sph::SphTypes::KeyType, sph::SphTypes::CoordinateType, Acc>& domain,
                                     int numRanks)
{
    auto idLocal = hostCopy<Acc, uint64_t>(simData.hydro.id);
    std::vector<uint64_t> owned(idLocal.begin() + domain.startIndex(), idLocal.begin() + domain.endIndex());

    std::vector<int> counts(numRanks), displs(numRanks);
    int localCount = int(owned.size());
    MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int r = 1; r < numRanks; ++r) { displs[r] = displs[r - 1] + counts[r - 1]; }

    std::vector<uint64_t> all(size_t(displs[numRanks - 1] + counts[numRanks - 1]));
    MPI_Allgatherv(owned.data(), localCount, MpiType<uint64_t>{}, all.data(), counts.data(), displs.data(),
                   MpiType<uint64_t>{}, MPI_COMM_WORLD);
    std::sort(all.begin(), all.end());
    return all;
}

//! @brief sum of the owned particle counts over all ranks (halo slots excluded)
template<class Acc>
uint64_t globalOwnedCount(const RemovalSimData<Acc>&,
                          const Domain<sph::SphTypes::KeyType, sph::SphTypes::CoordinateType, Acc>& domain)
{
    uint64_t local = domain.nParticles(), global = 0;
    MPI_Allreduce(&local, &global, 1, MpiType<uint64_t>{}, MPI_SUM, MPI_COMM_WORLD);
    return global;
}

//! @brief removeKey marks in the local key buffer, split by range
/*!
 * The removal sync does not resize the fields (the propagator resizes them at the next
 * computeForces), so the slots of the removed particles survive as stale marks above
 * nParticlesWithHalos. Only [0, nParticles) (owned) and [nParticles, nParticlesWithHalos)
 * (halo, delivered by the halo exchange and re-keyed in setupHalos) are live slots; a mark
 * in either range is a defect, a mark in the tail is expected leftover content.
 */
struct LocalMarks
{
    std::size_t owned = 0;
    std::size_t halo  = 0;
    std::size_t tail  = 0;
    std::size_t live() const { return owned + halo; }
};

template<class Acc>
LocalMarks localRemoveMarks(const RemovalSimData<Acc>& simData, std::size_t startIndex, std::size_t endIndex,
                            std::size_t nParticlesWithHalos)
{
    using KeyType = sph::SphTypes::KeyType;
    auto      keys = hostCopy<Acc, KeyType>(simData.hydro.keys);
    const auto mark = KeyType(cstone::removeKey<KeyType>{});

    LocalMarks marks;
    for (std::size_t i = 0; i < keys.size(); ++i)
    {
        if (keys[i] != mark) { continue; }
        if (i >= startIndex && i < endIndex) { ++marks.owned; }
        else if (i < nParticlesWithHalos) { ++marks.halo; }
        else { ++marks.tail; }
    }
    return marks;
}

//! @brief Allreduce of the local live mark count, so that a one-rank defect fails every rank
/*!
 * The peers must agree on whether to stop: a rank that returns early while its peer enters the
 * next computeForces deadlocks the whole job (observed as a run-until-SLURM-timeout hang), so the
 * decision is reduced and taken identically everywhere.
 */
inline unsigned long long globalLiveMarks(unsigned long long localMarks)
{
    unsigned long long globalMarks = 0;
    MPI_Allreduce(&localMarks, &globalMarks, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return globalMarks;
}

//! @brief report the per-range mark counts of this rank (diagnostics on failure)
template<class Acc>
void reportLocalMarks(const RemovalSimData<Acc>& simData, std::size_t startIndex, std::size_t endIndex,
                      std::size_t nParticlesWithHalos, std::size_t nIterations)
{
    int  rank  = 0;
    auto marks = localRemoveMarks(simData, startIndex, endIndex, nParticlesWithHalos);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (marks.live() + marks.tail == 0) { return; }

    std::cerr << "rank " << rank << " (after iteration " << nIterations << "): removeKey marks owned=" << marks.owned
              << " halo=" << marks.halo << " tail=" << marks.tail << " (owned=[" << startIndex << ":" << endIndex
              << "), nParticlesWithHalos=" << nParticlesWithHalos
              << ", buffer=" << hostCopy<Acc, sph::SphTypes::KeyType>(simData.hydro.keys).size() << ")" << std::endl;
}

//! @brief owned-range-aware call, the halo range starts at endIndex (buffer index space)
template<class Acc>
LocalMarks localRemoveMarks(const RemovalSimData<Acc>& simData,
                            const Domain<sph::SphTypes::KeyType, sph::SphTypes::CoordinateType, Acc>& domain)
{
    return localRemoveMarks<Acc>(simData, domain.startIndex(), domain.endIndex(), domain.nParticlesWithHalos());
}

//! @brief the positive scenario: flag set -> doomed singletons removed at the sync after integrate
//! @param withGravity run the identical IC with d.g != 0, i.e. through the syncGrav branch and the
//!                    multipole traversal, to show that the removal path is layout-branch agnostic
template<class Acc>
void removalTest(int rank, int numRanks, bool withGravity)
{
    using KeyType = sph::SphTypes::KeyType;
    using T       = sph::SphTypes::CoordinateType;
    using DomainT = Domain<KeyType, T, Acc>;

    std::ofstream nullOutput("/dev/null");
    int    bucketSize = 8, bucketSizeFocus = 8;
    float  theta = 1.0;
    Box<T> box(0, 10);

    DomainT domain(sphexa::ParticlesData<Acc>::exec, rank, numRanks, bucketSize, bucketSizeFocus, theta,
                   MPI_COMM_WORLD, box);

    RemovalSimData<Acc> simData;
    sphexa::HydroVePtProp<false, DomainT, RemovalSimData<Acc>> propagator(nullOutput, rank);
    propagator.activateFields(simData);
    initLocalIc(simData, rank, numRanks);
    if (withGravity) { simData.hydro.g = 1.0; }

    //! sphexa.cpp loop order: initial sync, then computeForces -> integrate
    propagator.sync(domain, simData);
    ASSERT_EQ(globalOwnedCount(simData, domain), uint64_t(numGlobal));

    propagator.computeForces(domain, simData);
    propagator.integrate(domain, simData); // marks the 3 singletons (keys = removeKey)

    propagator.sync(domain, simData); // removal happens here

    ASSERT_EQ(globalOwnedCount(simData, domain), uint64_t(numGlobal - numSingletons));

    // survivor id set: exactly clump + gravity
    auto ids = globalOwnedIds(simData, domain, numRanks);
    ASSERT_EQ(ids.size(), std::size_t(numGlobal - numSingletons));
    for (int i = 0; i < numGlobal; ++i)
    {
        bool isDoomed = (i >= singletonFirst && i < singletonFirst + numSingletons);
        auto it       = std::find(ids.begin(), ids.end(), uint64_t(i));
        if (isDoomed) { ASSERT_EQ(it, ids.end()) << "doomed id " << i << " survived"; }
        else { ASSERT_NE(it, ids.end()) << "id " << i << " lost"; }
    }

    //! no removeKey marks left in the live key range (owned + halo slots)
    //!
    //! Collective-safe: the pass/fail decision is Allreduced, so a single rank with a stray mark
    //! fails the test on both ranks instead of leaving its peer in the next collective.
    auto marks     = localRemoveMarks(simData, domain);
    auto liveMarks = globalLiveMarks(marks.live());
    reportLocalMarks(simData, domain.startIndex(), domain.endIndex(), domain.nParticlesWithHalos(), 1);
    EXPECT_EQ(liveMarks, 0ULL) << "removeKey marks left in the live key range after the removal sync";
    if (liveMarks != 0) { return; }

    // one more iteration: post-removal stability (no convergence throw, no new marks)
    propagator.computeForces(domain, simData);
    propagator.integrate(domain, simData);
    marks     = localRemoveMarks(simData, domain);
    liveMarks = globalLiveMarks(marks.live());
    reportLocalMarks(simData, domain.startIndex(), domain.endIndex(), domain.nParticlesWithHalos(), 2);
    EXPECT_EQ(liveMarks, 0ULL) << "removeKey marks appeared in the live key range after a further iteration";
}

//! @brief the negative scenario: flag unset -> "Neighbor search did not converge" in integrate
template<class Acc>
void noFlagThrowTest(int rank, int numRanks)
{
    using KeyType = sph::SphTypes::KeyType;
    using T       = sph::SphTypes::CoordinateType;
    using DomainT = Domain<KeyType, T, Acc>;

    std::ofstream nullOutput("/dev/null");
    int    bucketSize = 8, bucketSizeFocus = 8;
    float  theta = 1.0;
    Box<T> box(0, 10);

    DomainT domain(sphexa::ParticlesData<Acc>::exec, rank, numRanks, bucketSize, bucketSizeFocus, theta,
                   MPI_COMM_WORLD, box);

    RemovalSimData<Acc> simData;
    sphexa::HydroVePtProp<false, DomainT, RemovalSimData<Acc>> propagator(nullOutput, rank);
    propagator.activateFields(simData);
    initLocalIc(simData, rank, numRanks);
    simData.hydro.removeUnconvergedParticles = 0;

    propagator.sync(domain, simData);
    propagator.computeForces(domain, simData);

    // expected throw: exactly the ranks that own a doomed singleton (the catch below already
    // restricts `threw` to the expected message). Counted before integrate, while the buffer is
    // still in the full SFC layout: the throw leaves it partitioned.
    const auto ids = hostCopy<Acc, uint64_t>(simData.hydro.id);
    int        ownedDoomed = 0;
    for (std::size_t i = domain.startIndex(); i < domain.endIndex(); ++i)
    {
        ownedDoomed += (ids[i] >= uint64_t(singletonFirst) && ids[i] < uint64_t(singletonFirst + numSingletons));
    }

    // the throw happens after applyPartition_ (a local gather): no collective is left dangling,
    // so the per-rank outcome can be reduced right after the catch
    bool threw = false;
    try
    {
        propagator.integrate(domain, simData);
    }
    catch (const std::runtime_error& e)
    {
        threw = (std::strstr(e.what(), "Neighbor search did not converge") != nullptr);
    }

    // collective-safe: both counts are reduced before any assertion, so no rank returns early
    int threwLocal = threw ? 1 : 0, threwGlobal = 0;
    int expectLocal = (ownedDoomed > 0) ? 1 : 0, expectGlobal = 0;
    MPI_Allreduce(&threwLocal, &threwGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&expectLocal, &expectGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    ASSERT_GT(expectGlobal, 0) << "IC defect: no rank owns a doomed particle";
    ASSERT_EQ(threwGlobal, expectGlobal) << "exactly the ranks owning a doomed particle must throw";

    // nothing was removed: the throw precedes the next sync
    ASSERT_EQ(globalOwnedCount(simData, domain), uint64_t(numGlobal));
}

//! @brief the emptied-rank scenario: every fluid particle is unconverged, so the removal leaves
//! every rank with zero owned particles (the zero-fluid-rank path, driven through the propagator)
template<class Acc>
void allDoomedTest(int rank, int numRanks)
{
    using KeyType = sph::SphTypes::KeyType;
    using T       = sph::SphTypes::CoordinateType;
    using DomainT = Domain<KeyType, T, Acc>;

    std::ofstream nullOutput("/dev/null");
    int    bucketSize = 8, bucketSizeFocus = 8;
    float  theta = 1.0;
    Box<T> box(0, 10);

    DomainT domain(sphexa::ParticlesData<Acc>::exec, rank, numRanks, bucketSize, bucketSizeFocus, theta,
                   MPI_COMM_WORLD, box);

    RemovalSimData<Acc> simData;
    sphexa::HydroVePtProp<false, DomainT, RemovalSimData<Acc>> propagator(nullOutput, rank);
    propagator.activateFields(simData);
    initAllDoomedIc(simData, rank, numRanks);

    propagator.sync(domain, simData);
    propagator.computeForces(domain, simData);
    propagator.integrate(domain, simData); // marks all 16 singletons
    propagator.sync(domain, simData);      // removes all of them

    //! collectives first, assertions after: a rank that failed here would otherwise leave its peer
    //! in a collective (the failure mode this test class had before)
    auto ownedGlobal = globalOwnedCount(simData, domain);
    auto marks       = localRemoveMarks(simData, domain);
    auto liveMarks   = globalLiveMarks(marks.live());
    reportLocalMarks(simData, domain.startIndex(), domain.endIndex(), domain.nParticlesWithHalos(), 1);

    ASSERT_EQ(ownedGlobal, 0ULL) << "every particle was doomed, the removal must empty the domain";
    EXPECT_EQ(liveMarks, 0ULL) << "removeKey marks left in the live key range after emptying the domain";
}

} // namespace

TEST(UnconvergedRemoval2Ranks, removal)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    removalTest<execution::Cpu>(rank, numRanks, false);
}

TEST(UnconvergedRemoval2Ranks, removalGravity)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    removalTest<execution::Cpu>(rank, numRanks, true);
}

TEST(UnconvergedRemoval2Ranks, noFlagThrows)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    noFlagThrowTest<execution::Cpu>(rank, numRanks);
}

TEST(UnconvergedRemoval2Ranks, allDoomed)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    allDoomedTest<execution::Cpu>(rank, numRanks);
}

#ifdef SPH_EXA_PTYPE_GPU_TEST
TEST(UnconvergedRemoval2Ranks, removalGpu)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    removalTest<execution::Gpu>(rank, numRanks, false);
}

TEST(UnconvergedRemoval2Ranks, removalGravityGpu)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    removalTest<execution::Gpu>(rank, numRanks, true);
}

TEST(UnconvergedRemoval2Ranks, noFlagThrowsGpu)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    noFlagThrowTest<execution::Gpu>(rank, numRanks);
}

TEST(UnconvergedRemoval2Ranks, allDoomedGpu)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    allDoomedTest<execution::Gpu>(rank, numRanks);
}
#endif
