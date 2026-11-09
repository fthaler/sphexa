/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
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

/*!@file
 * @brief utilities for initial condition generation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/sfc/sfc.hpp"
#include "io/ifile_io.hpp"
#include "init/settings.hpp"

namespace sphexa
{

//! @brief sort x,y,z coordinates in the unit cube by SFC keys
template<class KeyType, class T>
void sortBySfcKey(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z)
{
    assert(x.size() == y.size() && y.size() == z.size());
    size_t blockSize = x.size();

    cstone::Box<T> box(0, 1);

    std::vector<KeyType> keys(blockSize);
    computeSfcKeys(x.data(), y.data(), z.data(), cstone::sfcKindPointer(keys.data()), blockSize, box);

    std::vector<cstone::LocalIndex> sfcOrder(blockSize);
    std::iota(begin(sfcOrder), end(sfcOrder), cstone::LocalIndex(0));
    cstone::sort_by_key(begin(keys), end(keys), begin(sfcOrder));

    std::vector<T> buffer(blockSize);
    cstone::gather<cstone::LocalIndex>(sfcOrder, x.data(), buffer.data());
    std::swap(x, buffer);
    cstone::gather<cstone::LocalIndex>(sfcOrder, y.data(), buffer.data());
    std::swap(y, buffer);
    cstone::gather<cstone::LocalIndex>(sfcOrder, z.data(), buffer.data());
    std::swap(z, buffer);
}

//! @brief read x,y,z coordinates from an H5Part file (at step 0)
template<class Vector>
void readTemplateBlock(const std::string& block, IFileReader* reader, Vector& x, Vector& y, Vector& z)
{
    reader->setStep(block, -1, FileMode::independent);
    size_t blockSize = reader->numParticles();
    x.resize(blockSize);
    y.resize(blockSize);
    z.resize(blockSize);

    reader->readField("x", x.data());
    reader->readField("y", y.data());
    reader->readField("z", z.data());

    reader->closeStep();
}

//! @brief read file attributes into an associative container
inline void readFileAttributes(InitSettings& settings, const std::string& settingsFile, IFileReader* reader,
                               bool verbose)
{
    if (not settingsFile.empty())
    {
        reader->setStep(settingsFile, -1, FileMode::independent);

        auto fileAttributes = reader->fileAttributes();
        for (const auto& attr : fileAttributes)
        {
            if (std::find(idTagFsStrings.begin(), idTagFsStrings.end(), attr) != idTagFsStrings.end()) { continue; }
            int64_t sz = reader->fileAttributeSize(attr);
            if (sz == 1)
            {
                bool settingRecognized = settings.count(attr);
                settings[attr]         = {};
                reader->fileAttribute(attr, &settings[attr], sz);
                if (reader->rank() == 0 && verbose)
                {
                    if (settingRecognized)
                    {
                        std::cout << "Override setting from " << settingsFile << ": " << attr << " = " << settings[attr]
                                  << std::endl;
                    }
                    else
                    {
                        std::cout << "Setting from " << settingsFile << ": " << attr << " = " << settings[attr]
                                  << " not recognized " << std::endl;
                    }
                }
            }
        }
        reader->closeStep();
    }
}

//! @brief generate particle IDs at the beginning of the simulation initialization
template<cstone::execution::Policy Exec>
void generateParticleIDs(Exec exec, std::span<uint64_t> id)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    std::vector<uint64_t> ranksLocalParticles(numRanks);
    size_t                localNumRanks = id.size();

    // fill ranksLocalParticles with the number of particles per rank
    MPI_Allgather(&localNumRanks, 1, MpiType<uint64_t>{}, ranksLocalParticles.data(), 1, MpiType<uint64_t>{},
                  MPI_COMM_WORLD);

    std::exclusive_scan(ranksLocalParticles.begin(), ranksLocalParticles.end(), ranksLocalParticles.begin(),
                        uint64_t(0));
    cstone::sequence(exec, id.data(), id.data() + id.size(), ranksLocalParticles[rank]);
}

template<class Dataset>
void initFieldsAtRest(Dataset& d, double m_part)
{
    cstone::fill(d.exec, d.m.begin(), d.m.end(), m_part);
    cstone::fill(d.exec, d.du_m1.begin(), d.du_m1.end(), 0.0);
    cstone::fill(d.exec, d.mui.begin(), d.mui.end(), d.muiConst);
    cstone::fill(d.exec, d.alpha.begin(), d.alpha.end(), d.alphamin);

    cstone::fill(d.exec, d.vx.begin(), d.vx.end(), 0.0);
    cstone::fill(d.exec, d.vy.begin(), d.vy.end(), 0.0);
    cstone::fill(d.exec, d.vz.begin(), d.vz.end(), 0.0);
    cstone::fill(d.exec, d.x_m1.begin(), d.x_m1.end(), 0.0);
    cstone::fill(d.exec, d.y_m1.begin(), d.y_m1.end(), 0.0);
    cstone::fill(d.exec, d.z_m1.begin(), d.z_m1.end(), 0.0);

    cstone::fill(d.exec, d.u.begin(), d.u.end(), 0.0);
    cstone::fill(d.exec, d.temp.begin(), d.temp.end(), 0.0);

    generateParticleIDs(d.exec, d.id);
}

/*! @brief interim mixed-IC mechanism: mark a fraction of the particles non-fluid, fresh ICs only
 *
 * The selection is a function of the particle id only (rank-count independent): every stride-th
 * global id is marked non-fluid, with stride = round(1/frac). The id % stride rule spreads the
 * marked particles over the whole domain — unlike id < f*N, which for glass ICs (where id follows
 * the global SFC order) would select one spatially contiguous corner chunk. The rule only realizes
 * fractions of the form 1/stride: frac > 0.5 is rejected (stride would round to 1, marking every
 * particle), and very small fractions may undershoot to zero marked particles.
 *
 * Fresh ICs only: ptype must be all-zero before marking (a ptype-bearing restart stores the mark
 * already). GPU mechanics: generateParticleIDs is host-MPI code, so ids are copied D2H, the
 * selection runs on the host and ptype is uploaded H2D — no kernels, no new .cu (host TU).
 */
template<class Dataset>
void markNonFluidFraction(Dataset& d, double frac)
{
    if (frac < 0.0 || frac > 1.0) { throw std::runtime_error("nonfluid fraction must be in [0:1]\n"); }
    if (frac == 0.0) { return; }
    if (frac > 0.5)
    {
        throw std::runtime_error("nonfluid fraction > 0.5: the id % stride rule cannot realize it "
                                 "(stride rounds to 1, i.e. every particle marked)\n");
    }

    size_t                localCount = d.id.size();
    std::vector<uint64_t> idHost(localCount);
    std::vector<uint8_t>  ptypeHost(localCount);
    if constexpr (Dataset::useGpu)
    {
        cstone::memcpyD2HAsync(d.exec, cstone::rawPtr(d.id), localCount, idHost.data());
        cstone::memcpyD2HAsync(d.exec, cstone::rawPtr(d.ptype), localCount, ptypeHost.data());
        cstone::syncGpu(d.exec);
    }
    else
    {
        std::copy(d.id.begin(), d.id.end(), idHost.begin());
        std::copy(d.ptype.begin(), d.ptype.end(), ptypeHost.begin());
    }

    if (std::any_of(ptypeHost.begin(), ptypeHost.end(), [](uint8_t p) { return p != 0; }))
    {
        throw std::runtime_error("markNonFluidFraction: ptype is not all-zero (fresh ICs only; "
                                 "a ptype-bearing restart already stores the mark)\n");
    }

    uint64_t stride = 1.0 / frac + 0.5;
    for (size_t i = 0; i < localCount; ++i)
    {
        if (idHost[i] % stride == 0) { ptypeHost[i] = 1; }
    }

    //! @brief the marked non-fluid h keeps its IC value (gravity softening) and feeds egrav and
    //! accelerationTimestep — assert it stays finite/sane
    if (localCount > 0)
    {
        auto [hMin, hMax] = cstone::minMax(d.exec, cstone::rawPtr(d.h), cstone::rawPtr(d.h) + localCount);
        if (not std::isfinite(hMin) || not std::isfinite(hMax) || hMin <= 0)
        {
            throw std::runtime_error("markNonFluidFraction: IC smoothing lengths are not finite/positive\n");
        }
    }

    if constexpr (Dataset::useGpu)
    {
        cstone::memcpyH2DAsync(d.exec, ptypeHost.data(), localCount, cstone::rawPtr(d.ptype));
        cstone::syncGpu(d.exec);
    }
    else { std::copy(ptypeHost.begin(), ptypeHost.end(), d.ptype.begin()); }
}

//! @brief Used to read the default values of dataset attributes
class BuiltinReader
{
public:
    using FieldType = util::Reduce<std::variant, util::Map<std::add_pointer_t, IO::Types>>;

    explicit BuiltinReader(InitSettings& attrs)
        : attributes_(attrs)
    {
    }

    [[nodiscard]] static int rank() { return -1; }

    void stepAttribute(const std::string& key, FieldType val, int64_t /*size*/)
    {
        std::visit([this, &key](auto arg) { attributes_[key] = *arg; }, val);
    };

private:
    //! @brief reference to attributes
    InitSettings& attributes_;
};

//! @brief build up an associative container with test case settings
template<class Dataset>
[[nodiscard]] InitSettings buildSettings(Dataset&& d, const InitSettings& testCaseSettings,
                                         const std::string& settingsFile, IFileReader* reader)
{
    InitSettings settings;
    // first layer: class member defaults in code
    BuiltinReader extractor(settings);
    d.hydro.loadOrStoreAttributes(&extractor);

    // second layer: test-case specific settings
    for (const auto& kv : testCaseSettings)
    {
        settings[kv.first] = kv.second;
    }

    // third layer: settings override by file given on commandline (highest precedence)
    readFileAttributes(settings, settingsFile, reader, true);

    return settings;
}

} // namespace sphexa
