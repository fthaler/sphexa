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
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
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

    //! @brief cached stable-partition permutations of the particle buffer, see computeForces
    AccVector<cstone::LocalIndex> ordering_, orderingInv_;
    //! @brief keys-sized scratch reused by the partition and by the gatherArrays applications of both permutations
    AccVector<uint64_t> partitionScratch_;

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
        d.treeView = domain.octreeProperties();
    }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        timer.start();
        pmReader.start();
        sync(domain, simData);
        timer.step("domain::sync");
        Base::logDomainStats(domain, simData);

        auto& d = simData.hydro;
        d.resize(domain.nParticlesWithHalos());
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        //! @brief step 4: stable partition of the full buffer [0:bufSize) into fluid and non-fluid particles
        //!
        //! M1 placeholder: nFluid = d.size(), i.e. all particles are fluid, so the partition and its
        //! inverse are the identity permutation and the round trip below must not perturb any value.
        size_t bufSize = domain.nParticlesWithHalos();
        size_t nFluid  = d.size();
        ordering_.resize(bufSize);
        orderingInv_.resize(bufSize);
        partitionScratch_.resize(bufSize);
        cstone::stablePartition(domain.exec(), cstone::rawPtr(d.ptype), bufSize, nFluid, cstone::rawPtr(ordering_), cstone::rawPtr(orderingInv_),
                                reinterpret_cast<unsigned*>(cstone::rawPtr(partitionScratch_)));
        timer.step("partition");

        fillMassHalos(domain.exec(), get<"m">(d), first, last);

        //! @brief step 4 (contd): apply the partition ordering to the fixed field set, keys first, in sync order
        auto gatherList = std::tuple_cat(std::tie(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d)),
                                                 std::tie(get<"m">(d)), get<ConservedFields>(d),
                                                 get<DependentFields>(d));
        util::for_each_tuple(
            [bufSize](auto& field)
            {
                if (field.size() != bufSize)
                {
                    throw std::runtime_error("ve-pt partition: field size does not match buffer size\n");
                }
            },
            gatherList);
        cstone::gatherArrays(domain.exec(), std::span<const cstone::LocalIndex>{cstone::rawPtr(ordering_), bufSize}, 0,
                             gatherList, std::tie(partitionScratch_));

        computeGroups(first, last, d, domain.box(), groups_);
        timer.step("computeGroups");
        updateSmoothingLengthIterative(groups_.view(), d, domain.box());
        timer.step("updateSmoothingLengthIterative");
        findNeighborsSfc(groups_.view(), d, domain.box());
        timer.step("FindNeighbors");
        pmReader.step();

        computeXMass(d, domain.box());
        timer.step("XMass");
        domain.exchangeHalos(std::tie(get<"xm">(d)), get<"ax">(d), get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        computeVe(d, domain.box());
        timer.step("Generalized Volume Elements");
        domain.exchangeHalos(get<"vx", "vy", "vz", "kx">(d), get<"ax">(d), get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        release(d, "ay", "az");
        acquire(d, "divv", "gradh");
        computeIadDivvCurlvGradh(d, domain.box());
        d.minDtRho = rhoTimestep(first, last, d);
        timer.step("IadVelocityDivCurlGradh");

        computeEOS(first, last, d);
        timer.step("EquationOfState");

        domain.exchangeHalos(get<"c11", "c12", "c13", "c22", "c23", "c33", "divv", "c">(d), get<"ax">(d),
                             get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        computeAVswitches(d, domain.box());
        timer.step("AVswitches");

        if (avClean)
        {
            domain.exchangeHalos(get<"dV11", "dV12", "dV13", "dV22", "dV23", "dV33", "prho", "alpha">(d), get<"ax">(d),
                                 get<"keys">(d));
        }
        else { domain.exchangeHalos(get<"prho", "alpha">(d), get<"ax">(d), get<"keys">(d)); }
        timer.step("mpi::synchronizeHalos");

        release(d, "divv", "gradh");
        acquire(d, "ay", "az");
        computeMomentumEnergy<avClean>(groups_.view(), nullptr, d, domain.box());
        timer.step("MomentumAndEnergy");
        pmReader.step();

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

        //! @brief step 9: restore the full SFC order by applying the cached inverse permutation
        //!
        //! gather with orderingInv_ computes dst[j] = partitioned[orderingInv_[j]] = original[j];
        //! never restore by re-sorting on keys (unconverged particles carry keys[i] = removeKey).
        if (orderingInv_.size() != bufSize)
        {
            throw std::runtime_error("ve-pt: inverse partition ordering size mismatch\n");
        }
        cstone::gatherArrays(domain.exec(), std::span<const cstone::LocalIndex>{cstone::rawPtr(orderingInv_), bufSize}, 0,
                             gatherList, std::tie(partitionScratch_));
        d.treeView = domain.octreeProperties();
        timer.step("partitionRestore");
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        computeTimestep(first, last, d);
        timer.step("Timestep");
        computePositions(groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});
        bool haveUnconvergedParticles = updateSmoothingLength(groups_.view(), d);
        if (haveUnconvergedParticles && not d.removeUnconvergedParticles)
        {
            throw std::runtime_error("Neighbor search did not converge\n");
        }
        timer.step("UpdateQuantities");
    }

    void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                    const cstone::Box<T>& box) override
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

        // first output pass: write everything allocated at the end of computeForces()
        output();

        // second output pass: write temporary quantities produced by the EOS
        release(d, "c11", "c12", "c13");
        acquire(d, "rho", "p", "gradh");
        computeEOS(first, last, d);
        output();
        release(d, "rho", "p", "gradh");
        acquire(d, "c11", "c12", "c13");

        // third output pass: recover temporary curlv and divv quantities
        release(d, "prho", "c");
        acquire(d, "divv", "curlv");
        // partial recovery of cij in range [first:last] without halos, which are not needed for divv and curlv
        if (!indicesDone.empty()) { computeIadDivvCurlvGradh(d, box); }
        output();
        release(d, "divv", "curlv");
        acquire(d, "prho", "c");

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
};

} // namespace sphexa
