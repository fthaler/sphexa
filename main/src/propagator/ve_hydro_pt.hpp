/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief A Propagator class for modern SPH with generalized volume elements, supporting
 *        multiple particle classes
 *
 * Particle class of a particle is given by its uint8 @p ptype field: 0 = fluid (SPH + gravity),
 * non-zero = gravity-only. Non-fluid particles keep their initial condition smoothing length
 * (gravity softening) and are integrated from gravity accelerations only.
 *
 * The SPH phase runs on the fluid-compacted index space [fluidFirst:fluidLast) realized by the
 * stable partition of the full SFC-ordered buffer; the full SFC layout is restored with the cached
 * inverse permutation before gravity and output. The two index spaces are the largest bug source:
 * exactly four layout transitions per step (computeForces steps 4/9, integrate steps 3/4), all via
 * the partition_/unpartition_ helper pair.
 *
 * Unconverged fluid particles (nc <= 1 after the h update, where the self-inclusive nc makes this
 * "no fluid neighbor") are removed at the next sync when removeUnconvergedParticles is set — the
 * flag arrives via the IC file attribute (ve parity, no CLI flag). Marking happens only in
 * updateSmoothingLength over the owned-fluid compacted range: gravity particles (ptype != 0) are
 * never marked and never removed (debug-checked by assertRemoveMarksFluidOnly_). Without the flag,
 * unconverged particles throw "Neighbor search did not converge" as in ve. Dumps never contain
 * doomed particles: marking happens in integrate step 3, removal in the sync of the next
 * computeForces, and the driver's only dump site lies between a post-removal computeForces and
 * integrate (main/src/sphexa/sphexa.cpp), i.e. outside that window. A driver that dumped inside it
 * would write the doomed particles, and since keys are not dumped, a restart of such a dump would
 * recompute keys from the positions and resurrect them (identical to ve). Conserved totals
 * (mass/energy) intentionally drop while removals occur.
 *
 * SPH quantities of non-fluid particles (prho, divv, xm, nc, dtCourant, fluid-only -f
 * fields) are stale/undefined in dumps: they are computed on the fluid range only; the default output
 * set holds only finite values and shared observables are not ptype-filtered.
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <mpi.h>
#include <optional>
#include <span>

#include "cstone/fields/field_get.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/util/tuple_util.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "ipropagator.hpp"
#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;
using util::FieldList;

template<bool avClean, class DomainType, class DataType>
class HydroVePtProp : public Propagator<DomainType, DataType>
{
protected:
    using Base = Propagator<DomainType, DataType>;
    using Base::pmReader;
    using Base::timer;

    using T             = typename DataType::RealType;
    using KeyType       = typename DataType::KeyType;
    using Tmass         = typename DataType::HydroData::Tmass;
    using HydroType     = typename DataType::HydroData::HydroType;
    using MultipoleType = ryoanji::CartesianQuadrupole<Tmass>;

    using Acc       = typename DataType::Exec;
    using MHolder_t = std::conditional_t<cstone::execution::HaveGpu<Acc>{},
                                         MultipoleHolderGpu<MultipoleType, DomainType, typename DataType::HydroData>,
                                         MultipoleHolderCpu<MultipoleType, DomainType, typename DataType::HydroData>>;

    template<class ValueType>
    using AccVector =
        std::conditional_t<cstone::execution::HaveGpu<Acc>{}, cstone::DeviceVector<ValueType>, std::vector<ValueType>>;

    MHolder_t      mHolder_;
    GroupData<Acc> groups_;

    //! @brief groups over the full owned range for the integrate-step-2 KDK drift
    //!
    //! The phase-7 groups_ are only valid in the fluid-compacted layout under the cached
    //! ordering_, while the drift is type-agnostic over the full SFC layout. On GPU the group
    //! kernels process one warp per group, so a single whole-range group would integrate only
    //! the first warp — hence warp-sized groups built from the full d.treeView, keeping the
    //! phase-7 groups_ intact for the integrate-step-3 h update.
    GroupData<Acc> driftGroups_;

    //! @brief cached stable-partition permutations of the particle buffer, see computeForces
    //!
    //! ordering_ stays valid between computeForces step 9 and integrate step 3 (no sync in between);
    //! it is a per-step cache, rebuilt in the next computeForces step 4, never reused across a sync.
    AccVector<cstone::LocalIndex> ordering_, orderingInv_;
    //! @brief keys-sized scratch reused by the partition and by the gatherArrays applications of both permutations
    AccVector<uint64_t> partitionScratch_;

    /*! @brief second halo exchange helper, operating on the fluid-compacted index space
     *
     * std::optional because Halos has no default constructor and the propagator is created before
     * simData (and its communicator) exists; lazily constructed in the first sync().
     */
    std::optional<cstone::Halos<KeyType>> fluidHalos_;

    //! @brief dedicated fluid-halo exchange scratch buffers, never aliased with keys/ax (invariant (d))
    AccVector<uint64_t> fluidHaloSendScratch_, fluidHaloRecvScratch_;

    //! @brief fluid-only leaf particle offsets, host-readable for exchangeRequests + accelerator
    //! copy for the fluid tree view and the kernels (step 2/5)
    std::vector<cstone::LocalIndex> fluidLayoutHost_;
    AccVector<cstone::LocalIndex>   fluidLayoutAcc_;

    //! @brief step-8 trigger flag (run-constant): divv/curlv/gradh are requested outputs
    bool iadTrioRequested_{false};

    //! @brief integrate step 3 marked particles for removal; the next sync drops them and then has to
    //! sanitize the halo keys (see clearHaloRemovalMarks_)
    bool removalPending_{false};

    /*! @brief the list of conserved particles fields with values preserved between iterations
     *
     * x, y, z, h and m are automatically considered conserved and must not be specified in this list.
     * ptype (particle class: 0 = fluid, non-zero = gravity-only) is conserved: it is never
     * released/acquired, survives domain repartition through the sync conserved tuple and
     * is restored from restart files.
     */
    using ConservedFields =
        FieldList<"temp", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "alpha", "id", "ptype">;

    //! @brief list of dependent fields, these may be used as scratch space during domain sync
    using DependentFields_ = FieldList<"ax", "ay", "az", "prho", "c", "du", "c11", "c12", "c13", "c22", "c23", "c33",
                                       "xm", "kx", "nc", "dtCourant">;

    //! @brief velocity gradient fields will only be allocated when avClean is true
    using GradVFields = FieldList<"dV11", "dV12", "dV13", "dV22", "dV23", "dV33">;

    //! @brief what will be allocated based AV cleaning choice
    using DependentFields =
        std::conditional_t<avClean, decltype(DependentFields_{} + GradVFields{}), decltype(DependentFields_{})>;

public:
    HydroVePtProp(std::ostream& output, size_t rank)
        : Base(output, rank)
    {
        if (avClean && rank == 0) { std::cout << "AV cleaning is activated" << std::endl; }
    }

    std::vector<std::string> conservedFields() const override
    {
        std::vector<std::string> ret{"x", "y", "z", "h", "m"};
        for_each_tuple(
            [&ret](auto f)
            {
                // ptype is an internal classification field, not part of the default output selection
                if (std::string_view(f.value) != "ptype") { ret.push_back(f.value); }
            },
            make_tuple(ConservedFields{}));
        return ret;
    }

    void activateFields(DataType& simData) override
    {
        auto& d = simData.hydro;
        //! @brief Fields accessed in domain sync (x,y,z,h,m,keys) are not part of extensible lists.
        d.setConserved("x", "y", "z", "h", "m");
        d.setDependent("keys");
        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFields{}));
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        auto& d = simData.hydro;
        //! @brief removal of unconverged fluid particles (opt-in via the IC file attribute)
        //!
        //! With the flag set, particles marked for removal in the previous integrate step 3
        //! (keys == removeKey) are dropped here and the conserved arrays shrink. Without the flag,
        //! unconverged particles throw in integrate step 3.
        if (d.g != 0.0)
        {
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            get<ConservedFields>(d), get<DependentFields>(d));
        }
        else
        {
            domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                        std::tuple_cat(std::tie(get<"m">(d)), get<ConservedFields>(d)), get<DependentFields>(d));
        }

        //! @brief the removal sync compacts the marks out of the owned range but can leave one in a
        //! halo slot: the halo keys are re-derived from positions by domain::setupHalos through
        //! computeSfcKeys, which preserves removeKey by design (that preservation is what carries the
        //! marks from integrate into this sync for owned particles)
        if (removalPending_)
        {
            clearHaloRemovalMarks_(domain, simData);
            removalPending_ = false;
        }

        d.treeView = domain.octreeProperties();

        //! @brief lazily construct the fluid halo helper on the first sync
        //!
        //! Halos builds everything inside exchangeRequests, so late construction is safe. Documented
        //! assumption: simData.comm == the Domain's comm (both MPI_COMM_WORLD on every current path;
        //! no accessor exists to assert it — a sub-comm Domain would deadlock here).
        if (!fluidHalos_) { fluidHalos_.emplace(Base::rank_, simData.comm); }
    }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        timer.start();
        pmReader.start();
        sync(domain, simData);
        timer.step("domain::sync");
        Base::logDomainStats(domain, simData);

        auto& d = simData.hydro;

        //! @brief step 8: output trio trigger, all-three-or-nothing, derived from the output selection
        //!
        //! Run-constant (setOutputFields runs once before the time-step loop; test the flag boolean,
        //! not membership lookups). gradh is additionally consumed by the saveFields EOS recovery
        //! pass, so the marking and the permanent allocation are trio-wide; requesting any one of
        //! the three allocates all three. The trio stays out of DependentFields: its one halo
        //! exchange (divv) uses the fluid pattern.
        using HydroData = std::decay_t<decltype(d)>;
        auto isOutput   = [&d](const char* name)
        {
            int fidx = int(cstone::getFieldIndex(name, HydroData::fieldNames));
            return std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) !=
                   d.outputFieldIndices.end();
        };
        iadTrioRequested_ = isOutput("divv") || isOutput("curlv") || isOutput("gradh");

        //! @brief step 1: single marking site (step 8) — mark the IAD trio dependent iff requested
        //!
        //! Step-1 ordering invariant (load-bearing, not defensive): sync -> mark -> d.resize ->
        //! (ptype, m) halo exchange -> fluid counts (step 2). Marking must precede d.resize, else
        //! the trio stays zero-length through step 4's gatherArrays (OOB on GPU). The halo ptype of
        //! the exchange must be fresh when step 2 counts it: a stale halo ptype yields a
        //! peer-inconsistent fluidLayout, surfacing as an MPI size-mismatch hang in the fluid
        //! exchanges, not a clean error.
        //! Stale-content invariant: d.resize leaves grown non-conserved fields holding stale
        //! contents — safe because nothing consumes the trio between the resize and the phase-7
        //! IAD: step 4's flag-gated gather reads it value-agnostically, and the first readers
        //! (rhoTimestep, computeEOS, the divv halo exchange) all run after the IAD.
        if (iadTrioRequested_) { d.setDependent("divv", "curlv", "gradh"); }

        d.resize(domain.nParticlesWithHalos());

        //! @brief step 1 (contd): halo ptype + halo m on the focus-tree pattern set by the sync
        //!
        //! Halo m: sync delivers halos only for x,y,z,h, but computeXMass and gravity's upsweep
        //! consume m over the whole buffer incl. halos. Replaces ve's homogeneous fillMassHalos
        //! (dropped in step 6). Keys-as-receive-scratch safety: the GPU packed receive (scratch
        //! keys) reaches into the owned range, but nothing reads keys before the next sync
        //! recomputes them; the CPU path ignores the scratches.
        domain.exchangeHalos(std::tie(d.ptype, d.m), get<"ax">(d), get<"keys">(d));

        //! @brief step 2: fluid-only leaf counts + exclusive scan
        //!
        //! No communication: the whole-leaf halo delivery premise makes per-leaf fluid counts
        //! peer-consistent. fluidLayout is indexed by focus-tree leaves in the fluid-compacted
        //! index space: fluidLayout[i] is the number of fluid particles in all leaves before leaf i.
        cstone::computeFluidLayout(domain.exec(), domain.layout(), cstone::rawPtr(d.ptype), fluidLayoutAcc_,
                                   fluidLayoutHost_, d.getAllocGrowthRate());
        size_t nFluid = fluidLayoutHost_.back();

        size_t fluidFirst = fluidLayoutHost_[domain.startCell()];
        size_t fluidLast  = fluidLayoutHost_[domain.endCell()];
        timer.step("fluidLayout");

        //! @brief step 3: fluid halo request exchange, in the fluid-compacted index space realized
        //! by step 4's partition
        //!
        //! Invariants (load-bearing, violations hang or corrupt — the fencing is emergent from the
        //! two Alltoalls plus the complete-on-return exchanges):
        //! (a) no main-instance exchangeHalos/exchangeRequests from the step-3 Alltoall until the
        //!     next sync's exchangeRequests Alltoall (covers the fluid exchanges and the dead zone
        //!     after them);
        //! (b) no fluid exchangeHalos after the step-7 exchanges until the next sync's
        //!     exchangeRequests;
        //! (c) fluidHalos_.exchangeRequests before its first exchangeHalos, as for the main
        //!     instance;
        //! (d) fluidHalos_ gets its own GPU scratch buffers (members above); ax/keys never aliased.
        assert(fluidHalos_);
        fluidHalos_->exchangeRequests(domain.focusTree().treeLeaves(), domain.focusTree().assignment(),
                                      fluidLayoutHost_);
        assertFluidHalosConsistent(simData, fluidFirst, fluidLast, nFluid);
        //! the request exchange is part of the fluidHaloExchange region (steps 3/7 share the name)
        timer.step("fluidHaloExchange");

        //! @brief step 4: partition into the fluid-compacted layout
        partition_(domain, simData, nFluid);
        timer.step("partition");

        //! @brief step 5: fluid tree view — the focus tree with the layout pointer swapped to the
        //! fluid layout
        //!
        //! The layout-swap alone is self-consistent: centers/sizes are key-derived node-cell bboxes,
        //! identical in both views (conservative over-approximation for overlap bounds). Under the
        //! fluid layout, updateSmoothingLengthIterative counts fluid-only neighbors, so the view
        //! must be in place before step 7.
        auto fluidTreeView   = domain.octreeProperties();
        fluidTreeView.layout = cstone::rawPtr(fluidLayoutAcc_);
        d.treeView           = fluidTreeView;
        timer.step("fluidView");

        //! @brief step 7: SPH pipeline on the owned fluid range [fluidFirst:fluidLast) with the fluid view
        //!
        //! The pipeline runs unconditionally, on ranks without owned fluid too (a rank owning zero
        //! fluid can still have nFluid > 0: nFluid counts halo fluid). Every callee accepts the
        //! empty range / zero group count: computeGroups over [fluidFirst:fluidLast) with
        //! fluidFirst == fluidLast realizes the well-defined empty state itself (CPU: a single empty
        //! group; GPU: numGroups = 0 with firstBody == lastBody), which also keeps integrate's
        //! unconditional updateSmoothingLength(groups_.view()) well-defined; the empty groups make
        //! findNeighborsSfc build an empty neighborhood (the clustered GPU builder early-returns on
        //! groups.firstBody == groups.lastBody regardless of totalBodies), on which the subsequent
        //! Ij loops and the EOS kernels are no-ops. computeMomentumEnergy on the empty path resets
        //! minDtCourant to +inf, the identity of integrate's global min-reduce.
        //!
        //! The exchanges, timer steps and PM reads between the kernels share the same skeleton on
        //! every rank: haloexchange tags each message with a per-rank epoch counter incremented on
        //! every exchangeHalos call (halos.hpp:94/108), so a rank skipping an exchange would desync
        //! its tags against peers and hang the next nonempty exchange; and Timer::writeTimings /
        //! PmReader::writeTimings dump per-rank sample vectors through H5PartWriterSeq, whose
        //! MPI_Gather uses the root's count and whose file format assumes one global step grid, so
        //! a rank emitting fewer steps truncates the final metrics gather with "MPI_Gather:
        //! message truncated" (hit by the config-b 2-rank run, 2026-11-11). A zero-owned-fluid
        //! rank receives real halo fluid (its incoming ranges cover the interface) but sends
        //! nothing (no peer requests fluid from a rank that owns none) and never reads the
        //! incoming data (no owned fluid consumes halo inputs).
        computeGroups(fluidFirst, fluidLast, d, domain.box(), groups_);
        timer.step("computeGroups");

        updateSmoothingLengthIterative(groups_.view(), d, domain.box());
        timer.step("updateSmoothingLengthIterative");

        findNeighborsSfc(groups_.view(), d, domain.box(), false, nFluid);
        timer.step("FindNeighbors");
        pmReader.step();

        computeXMass(d, domain.box());
        timer.step("XMass");

        exchangeFluidHalos_(domain, std::tie(get<"xm">(d)));
        timer.step("fluidHaloExchange");

        computeVe(d, domain.box());
        timer.step("Generalized Volume Elements");
        exchangeFluidHalos_(domain, get<"vx", "vy", "vz", "kx">(d));
        timer.step("fluidHaloExchange");

        //! @brief state-aware field rotation, both legs coupled on the trigger flag
        //!
        //! When the trio is dependent (requested outputs), both pairs are skipped: releasing ay/az
        //! without consuming them would strand them released (acquireOne throws on non-unused,
        //! releaseOne throws on non-dependent). Nothing reads ay/az between the rotation legs (pure
        //! buffer donation; the IAD input tuple has no ay/az). Otherwise run both pairs as ve does.
        //! Unconditional: the donation is what gives divv a full-length
        //! buffer for the third exchange below — while unused, divv is zero-length and would fail
        //! exchangeFluidHalos_'s buffer-size check.
        if (!iadTrioRequested_)
        {
            release(d, "ay", "az");
            acquire(d, "divv", "gradh");
        }

        //! @brief the IAD kernel selects the curlv branch whenever the curlv buffer is full-length:
        //! assert the gating survives the compaction (curlv is permanently allocated when requested)
        assert(!iadTrioRequested_ || get<"curlv">(d).size() == get<"x">(d).size());
        computeIadDivvCurlvGradh(d, domain.box());
        //! empty fluid range -> +inf before any divv access (M2b guard, ts_global.hpp): divv would
        //! only be read in [fluidFirst:fluidLast), so the donated/stale contents are never touched
        d.minDtRho = rhoTimestep(fluidFirst, fluidLast, d);
        timer.step("IadVelocityDivCurlGradh");

        computeEOS(fluidFirst, fluidLast, d);
        timer.step("EquationOfState");

        exchangeFluidHalos_(domain, get<"c11", "c12", "c13", "c22", "c23", "c33", "divv", "c">(d));
        timer.step("fluidHaloExchange");

        computeAVswitches(d, domain.box());
        timer.step("AVswitches");

        if (avClean)
        {
            exchangeFluidHalos_(domain, get<"dV11", "dV12", "dV13", "dV22", "dV23", "dV33", "prho", "alpha">(d));
        }
        else { exchangeFluidHalos_(domain, get<"prho", "alpha">(d)); }
        timer.step("fluidHaloExchange");

        if (!iadTrioRequested_)
        {
            release(d, "divv", "gradh");
            acquire(d, "ay", "az");
        }

        computeMomentumEnergy<avClean>(groups_.view(), nullptr, d, domain.box());
        timer.step("MomentumAndEnergy");
        pmReader.step();

        //! @brief step 9: restore the full SFC layout before gravity (a full-layout routine)
        unpartition_(domain, simData);
        timer.step("partitionRestore");

        //! @brief step 10: zero the non-fluid owned accelerations before gravity
        //!
        //! Gravity accumulates into ax/ay/az, and phase 7 wrote fluid accelerations only, so the
        //! non-fluid owned slots hold sync/halo-exchange scratch garbage (ax is also the exchange
        //! send buffer). Unconditional, so non-fluid accelerations are well-defined even with
        //! g == 0 (the drift consumes them for all particles). Non-fluid halo accelerations stay
        //! garbage — never read. Gravity itself keeps the full-buffer view (correct for mixed
        //! populations, layout.back() == bufSize); no nFluid gating here.
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();
        cstone::fillMasked(domain.exec(), cstone::rawPtr(d.ax) + first, cstone::rawPtr(d.ax) + last,
                           cstone::rawPtr(d.ptype) + first, HydroType(0));
        cstone::fillMasked(domain.exec(), cstone::rawPtr(d.ay) + first, cstone::rawPtr(d.ay) + last,
                           cstone::rawPtr(d.ptype) + first, HydroType(0));
        cstone::fillMasked(domain.exec(), cstone::rawPtr(d.az) + first, cstone::rawPtr(d.az) + last,
                           cstone::rawPtr(d.ptype) + first, HydroType(0));

        if (d.g != 0.0)
        {
            auto groups = mHolder_.computeSpatialGroups(d, domain);
            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            pmReader.step();
            mHolder_.traverse(groups, d, domain);
            timer.step("Gravity");
            pmReader.step();

            auto stats = mHolder_.readStats();
            timer.logStatistics("sumP2P", stats[0] / timer.getLastStepTime());
            timer.logStatistics("sumM2P", stats[2] / timer.getLastStepTime());
        }
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        //! @brief step 1: global timestep over the owned range in the full SFC layout
        //!
        //! Halos are excluded: their ax/ay/az are stale (phase 7 wrote owned fluid accelerations
        //! only; non-fluid halos are never written) and would poison the min-reduce, while
        //! minDtCourant/minDtRho were computed over fluid only in computeForces.
        computeTimestep(first, last, d);
        timer.step("Timestep");

        //! @brief step 2: KDK position update of all owned particles in the full SFC layout
        //!
        //! Type-agnostic: non-fluid particles drift on their gravity accelerations. The phase-7
        //! groups_ are only valid in the compacted layout, so the drift runs on driftGroups_
        //! (built above from the full d.treeView; trivial owned-range group on CPU). The energy
        //! update is masked per particle to the fluid class. No data dependency with the h update
        //! in step 3 (the drift uses h only for the FBC correction) — but the layouts differ, so
        //! the h update must not be reordered before the drift.
        computeGroups(first, last, d, domain.box(), driftGroups_);
        computePositions(driftGroups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)}, nullptr,
                         cstone::rawPtr(d.ptype));
        timer.step("UpdateQuantities");

        //! @brief step 3: fluid-only h update in the compacted layout
        //!
        //! Re-applies the cached ordering_ (per-step cache from this step's computeForces step 4;
        //! no sync in between): fluid owned particles are only contiguous in the compacted layout.
        //! Runs on the phase-7 fluid groups, still under the same ordering_, with nothing rebuilt
        //! in between. removeKey markings land in the permuted keys — they survive the step-4
        //! gather (a gather, never a key sort) and are removed at the next sync when
        //! removeUnconvergedParticles is set; without the flag, unconverged particles throw.
        applyPartition_(domain, simData);
        bool haveUnconvergedParticles = updateSmoothingLength(groups_.view(), d);
        if (haveUnconvergedParticles)
        {
            assertRemoveMarksFluidOnly_(domain, simData);
            if (not d.removeUnconvergedParticles) { throw std::runtime_error("Neighbor search did not converge\n"); }
            removalPending_ = true;
        }
        timer.step("partition");

        //! @brief step 4: restore the full SFC layout for the next sync
        //!
        //! Same fixed gather list + flag-gated trio call as step 3, now carrying the h/keys
        //! updated there. Precondition: no resize between syncs; the next sync resizes to the new
        //! count, so ordering_/orderingInv_ are never reused across a sync.
        unpartition_(domain, simData);
        timer.step("partitionRestore");
    }

    void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                    const cstone::Box<T>& /* box */) override
    {
        auto& d             = simData.hydro;
        auto  fieldPointers = d.data();
        auto  indicesDone   = d.outputFieldIndices;
        auto  namesDone     = d.outputFieldNames;

        // every ve-pt dump carries ptype: append it to the local output selection iff not already requested
        {
            using HydroData = std::decay_t<decltype(d)>;
            int ptypeIdx    = int(cstone::getFieldIndex("ptype", HydroData::fieldNames));
            if (std::find(indicesDone.begin(), indicesDone.end(), ptypeIdx) == indicesDone.end())
            {
                indicesDone.push_back(ptypeIdx);
                namesDone.push_back("ptype");
            }
        }

        auto output = [&]()
        {
            for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
            {
                int fidx = indicesDone[i];
                if (d.isAllocated(fidx))
                {
                    int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
                                 d.outputFieldIndices.begin();
                    std::visit(
                        [writer, c = column, key = namesDone[i]](auto field)
                        {
                            auto&& tmp = cstone::toHost(*field);
                            writeField(writer, key, tmp.data(), c);
                        },
                        fieldPointers[fidx]);
                    indicesDone.erase(indicesDone.begin() + i);
                    namesDone.erase(namesDone.begin() + i);
                }
            }
        };

        // first output pass: write everything allocated at the end of computeForces(); the requested
        // divv/curlv/gradh are permanently allocated and phase-7-written, so they are attached by
        // the phase-9 restore and written directly here (no recompute pass)
        output();

        // second output pass: write temporary quantities produced by the EOS; computeEOS is
        // element-wise, so it is safe in the full SFC layout after the restore
        if (iadTrioRequested_)
        {
            // gradh is already dependent (requested output, must not be re-acquired: acquireOne
            // throws on non-unused): release only c11,c12 donors, acquire only rho,p. Releasing
            // three donors for two acquires would strand c13 released and starve the closing
            // acquire; in the other branch a literal release(rho,p) would do the same.
            release(d, "c11", "c12");
            acquire(d, "rho", "p");
            computeEOS(first, last, d);
            output();
            release(d, "rho", "p");
            acquire(d, "c11", "c12");
        }
        else
        {
            release(d, "c11", "c12", "c13");
            acquire(d, "rho", "p", "gradh");
            computeEOS(first, last, d);
            output();
            release(d, "rho", "p", "gradh");
            acquire(d, "c11", "c12", "c13");
            // c11/c12 keep rho/p values in [first:last] (ve's partial cij recovery dropped):
            // harmless, nothing consumes cij before the next IAD
        }

        /* The following data is now lost and no longer available in the integration step
         *  c11, c12, c12: halos invalidated
         *  prho, c: destroyed
         */

        if (!indicesDone.empty() && Base::rank_ == 0)
        {
            std::cout << "WARNING: the following fields are not in use and therefore not output: ";
            for (std::size_t fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
            {
                std::cout << d.fieldNames[indicesDone[fidx]] << ",";
            }
            std::cout << d.fieldNames[indicesDone.back()] << std::endl;
        }
        timer.step("FileOutput");
    }

private:
    //! @brief the fixed partition gather set, keys first, in sync gather order; ptype rides in ConservedFields
    auto fixedGatherList(DataType& simData)
    {
        auto& d = simData.hydro;
        return std::tuple_cat(std::tie(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d)),
                              std::tie(get<"m">(d)), get<ConservedFields>(d), get<DependentFields>(d));
    }

    //! @brief release-safe application of one partition permutation to a gather set
    //!
    //! Per-field size == ordering-size throw (gatherArrays' own asserts compile out in release, and
    //! M0's release-active throw on array.size() < ordering.size() catches the null-source gather of
    //! a size-0 field). The scratch is addressed fresh on every call: gatherArrays may swap it with
    //! a same-type array (keys/id), so raw pointers are never cached across an apply.
    template<class... Arrays>
    void applyPermutation_(const Acc& exec, std::span<const cstone::LocalIndex> ordering, std::tuple<Arrays&...> arrays)
    {
        size_t bufSize = ordering.size();
        util::for_each_tuple(
            [bufSize](auto& field)
            {
                if (field.size() != bufSize)
                {
                    throw std::runtime_error("ve-pt partition: field size does not match buffer size\n");
                }
            },
            arrays);
        cstone::gatherArrays(exec, ordering, 0, arrays, std::tie(partitionScratch_));
    }

    //! @brief step 4: fused stable partition + application of the ordering to the full gather set
    //!
    //! One of the exactly four per-step layout transitions (computeForces steps 4/9, integrate
    //! steps 3/4); all of them go through partition_/unpartition_/applyPartition_. Never hand a
    //! fluid index to a full-layout routine (esp. rhoTimestep, gravity, output) or vice versa.
    void partition_(DomainType& domain, DataType& simData, size_t nFluid)
    {
        auto&  d       = simData.hydro;
        size_t bufSize = domain.nParticlesWithHalos();

        ordering_.resize(bufSize);
        orderingInv_.resize(bufSize);
        partitionScratch_.resize(bufSize);

        //! the head of partitionScratch_ serves as the GPU flags buffer (exclusive scan of the
        //! non-fluid flags), carved before the gathers
        cstone::stablePartition(domain.exec(), cstone::rawPtr(d.ptype), bufSize, nFluid, cstone::rawPtr(ordering_),
                                cstone::rawPtr(orderingInv_),
                                reinterpret_cast<unsigned*>(cstone::rawPtr(partitionScratch_)));

        applyPartition_(domain, simData);
    }

    //! @brief apply the cached ordering_ to the fixed gather set (+ flag-gated trio call)
    //!
    //! Used by computeForces step 4 (after stablePartition fills the caches) and by integrate
    //! step 3, which re-applies the per-step cache — valid because no sync happened since
    //! computeForces step 4. The ordering must match the buffer size: a resize between syncs
    //! invalidates it (the next sync resizes to the new count).
    void applyPartition_(DomainType& domain, DataType& simData)
    {
        auto&  d       = simData.hydro;
        size_t bufSize = domain.nParticlesWithHalos();

        if (ordering_.size() != bufSize) { throw std::runtime_error("ve-pt: partition ordering size mismatch\n"); }

        applyPermutation_(domain.exec(), {cstone::rawPtr(ordering_), bufSize}, fixedGatherList(simData));
        if (iadTrioRequested_)
        {
            applyPermutation_(domain.exec(), {cstone::rawPtr(ordering_), bufSize},
                              std::tie(get<"divv">(d), get<"curlv">(d), get<"gradh">(d)));
        }
    }

    //! @brief step 9: restore the full SFC order with the cached inverse permutation
    void unpartition_(DomainType& domain, DataType& simData)
    {
        auto&  d       = simData.hydro;
        size_t bufSize = domain.nParticlesWithHalos();

        //! release-safe: the fused kernel writes one orderingInv_ entry per i (the debug asserts
        //! compile out in release)
        if (orderingInv_.size() != bufSize)
        {
            throw std::runtime_error("ve-pt: inverse partition ordering size mismatch\n");
        }

        //! gather with orderingInv_ computes dst[j] = partitioned[orderingInv_[j]] = original[j];
        //! never restore by re-sorting on keys (unconverged particles carry keys[i] = removeKey)
        applyPermutation_(domain.exec(), {cstone::rawPtr(orderingInv_), bufSize}, fixedGatherList(simData));
        if (iadTrioRequested_)
        {
            applyPermutation_(domain.exec(), {cstone::rawPtr(orderingInv_), bufSize},
                              std::tie(get<"divv">(d), get<"curlv">(d), get<"gradh">(d)));
        }

        d.treeView = domain.octreeProperties();
    }

    //! @brief fluid halo exchange with the dedicated scratch buffers (invariant (d))
    //!
    //! The fluid exchanges bypass the domain's checkSizesEqual entirely (direct Halos calls), so the
    //! pre-size check happens here, at the call sites.
    template<class... Vectors>
    void exchangeFluidHalos_(DomainType& domain, std::tuple<Vectors&...> arrays)
    {
        size_t bufSize = domain.nParticlesWithHalos();
        util::for_each_tuple(
            [bufSize](auto& field)
            {
                if (field.size() != bufSize)
                {
                    throw std::runtime_error("ve-pt fluid halo exchange: field size does not match buffer size\n");
                }
            },
            arrays);
        fluidHalos_->exchangeHalos(domain.exec(), arrays, fluidHaloSendScratch_, fluidHaloRecvScratch_);
    }

    /*! @brief debug-only peer-consistency check of the fluid halo pattern
     *
     * Turns a stale halo ptype (which yields a peer-inconsistent fluidLayout) into a clean assert
     * instead of an MPI size-mismatch hang. Verifies that every fluid halo sent by any peer matches
     * the local incoming range, and that the incoming ranges tile the halo fluid slots
     * [0:fluidFirst) + [fluidLast:nFluid) of the compacted buffer.
     *
     * The checks assert (abort the rank, tearing down the mpirun job) rather than throw: a thrown,
     * caught violation would hang the peer at the next collective — the exact failure mode this
     * check exists to expose.
     */
    void assertFluidHalosConsistent([[maybe_unused]] const DataType& simData, [[maybe_unused]] size_t fluidFirst,
                                    [[maybe_unused]] size_t fluidLast, [[maybe_unused]] size_t nFluid) const
    {
#ifndef NDEBUG
        assert(fluidLayoutHost_.back() == nFluid);

        int numRanks = 0, myRank = 0;
        MPI_Comm_size(simData.comm, &numRanks);
        MPI_Comm_rank(simData.comm, &myRank);

        // entry [i * numRanks + j] = number of fluid halos rank i sends to rank j
        std::vector<int> sendCounts(size_t(numRanks) * numRanks, 0);
        for (int peer = 0; peer < numRanks; ++peer)
        {
            sendCounts[size_t(myRank) * numRanks + peer] = int(fluidHalos_->outgoing()[peer].totalCount());
        }
        MPI_Allreduce(MPI_IN_PLACE, sendCounts.data(), int(sendCounts.size()), MPI_INT, MPI_SUM, simData.comm);

        for (int peer = 0; peer < numRanks; ++peer)
        {
            assert(sendCounts[size_t(peer) * numRanks + myRank] == int(fluidHalos_->incoming()[peer].count()) &&
                   "ve-pt: fluid halo request mismatch with a peer rank");
        }

        // the incoming ranges must tile [0:fluidFirst) + [fluidLast:nFluid); the only allowed gap is
        // the owned fluid range [fluidFirst:fluidLast)
        std::vector<std::pair<size_t, size_t>> ranges;
        for (int peer = 0; peer < numRanks; ++peer)
        {
            auto in = fluidHalos_->incoming()[peer];
            if (in.count() > 0) { ranges.emplace_back(in.start(), in.start() + in.count()); }
        }
        std::sort(ranges.begin(), ranges.end());

        size_t covered = 0;
        for (auto [rangeStart, rangeEnd] : ranges)
        {
            assert((rangeStart == covered || (covered == fluidFirst && rangeStart == fluidLast)) &&
                   "ve-pt: fluid halo incoming ranges do not tile the halo fluid slots");
            covered = rangeEnd;
        }
        assert((covered == nFluid || (covered == fluidFirst && fluidLast == nFluid)) &&
               "ve-pt: fluid halo incoming ranges do not tile the halo fluid slots");
#endif
    }

    /*! @brief drop removeKey marks that landed in the halo slots of the removal sync
     *
     * domain::sync drops the particles carrying removeKey and compacts the owned range, but the halo
     * keys are re-derived from the delivered coordinates by setupHalos through computeSfcKeys, which
     * deliberately skips removal keys (that skip is what carries the marks of owned particles from
     * integrate into the sync). A dropped particle's slot can therefore end up in the new halo range
     * with its mark intact, i.e. as a live mark in a non-owned slot: it trips the debug guard
     * assertRemoveMarksFluidOnly_ (which asserts zero marks outside the owned-fluid range) and leaves
     * the halo keys inconsistent with their positions.
     *
     * Recompute the halo keys from the coordinates after zeroing the range: computeSfcKeys skips slots
     * that already hold a removal key, so the mark has to be gone before the recompute. Cost is one
     * pass over the halo range, and this only runs on removal steps (removalPending_).
     */
    void clearHaloRemovalMarks_(const DomainType& domain, DataType& simData)
    {
        auto& d = simData.hydro;

        //! buffer index space: the halo slots are [endIndex, nParticlesWithHalos)
        size_t first = domain.endIndex();
        size_t last  = domain.nParticlesWithHalos();
        if (first >= last) { return; }

        cstone::fill(domain.exec(), cstone::rawPtr(d.keys) + first, cstone::rawPtr(d.keys) + last, KeyType(0));
        cstone::computeSfcKeys(domain.exec(), cstone::rawPtr(d.x) + first, cstone::rawPtr(d.y) + first,
                               cstone::rawPtr(d.z) + first, cstone::sfcKindPointer(cstone::rawPtr(d.keys)) + first,
                               last - first, domain.box());
    }

    /*! @brief debug-only stray-mark guard for the unconverged-particle removal path
     *
     * Runs in integrate step 3 when updateSmoothingLength reported unconverged particles (the only
     * marking site): removeKey may appear only in the owned-fluid compacted range
     * [fluidFirst:fluidLast) of the permuted keys. A mark anywhere else would mean a gravity
     * particle (ptype != 0) or a halo slot got marked — a regression that would silently drop the
     * wrong particle class at the next sync. Counting is restricted to the two non-owned ranges
     * [0:fluidFirst) + [fluidLast:bufSize); marks in the owned-fluid range are the expected state.
     *
     * Asserts (aborts the rank) rather than throws, like assertFluidHalosConsistent: a thrown,
     * caught violation would hang the peers at the next collective.
     */
    void assertRemoveMarksFluidOnly_([[maybe_unused]] const DomainType& domain,
                                     [[maybe_unused]] const DataType&   simData) const
    {
#ifndef NDEBUG
        const auto& d       = simData.hydro;
        size_t      bufSize = domain.nParticlesWithHalos();

        size_t fluidFirst = fluidLayoutHost_[domain.startCell()];
        size_t fluidLast  = fluidLayoutHost_[domain.endCell()];

        const KeyType* keys        = cstone::rawPtr(d.keys);
        KeyType        markedValue = cstone::removeKey<KeyType>{};
        size_t         strayMarks  = 0;
        if constexpr (cstone::execution::HaveGpu<Acc>{})
        {
            strayMarks += cstone::count(domain.exec(), keys, keys + fluidFirst, markedValue);
            strayMarks += cstone::count(domain.exec(), keys + fluidLast, keys + bufSize, markedValue);
        }
        else
        {
            strayMarks += std::count(keys, keys + fluidFirst, markedValue);
            strayMarks += std::count(keys + fluidLast, keys + bufSize, markedValue);
        }
        assert(strayMarks == 0 && "ve-pt: non-fluid or halo particle marked for removal");
#endif
    }
};

} // namespace sphexa
