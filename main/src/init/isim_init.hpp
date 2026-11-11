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

/*! @file
 * @brief Test-case simulation data initialization
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <filesystem>
#include <map>

#include "cstone/sfc/box.hpp"
#include "io/ifile_io.hpp"
#include "sphexa/simulation_data.hpp"

#include "settings.hpp"
#include "utils.hpp"
#include "io/id_tag_utils.hpp"

namespace sphexa
{

template<class Dataset>
class ISimInitializer
{
public:
    ISimInitializer(std::string settingsFile)
        : settingsFile_(settingsFile)
    {
    }

    //! @brief set the per-init non-fluid fraction option, applied in init() after initImpl and runTagging;
    //! a frac <= 0 disables the option (see markNonFluidFraction)
    void setNonFluidFraction(double frac, bool contiguous)
    {
        nonFluidFrac_       = frac;
        nonFluidContiguous_ = contiguous;
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& d,
                                                 IFileReader* reader) const
    {
        auto box = initImpl(rank, numRanks, cbrtNumPart, d, reader);
        runTagging(reader, d);
        applyNonFluidFraction(d);
        return box;
    }

    virtual const InitSettings& constants() const = 0;

    virtual ~ISimInitializer() = default;

protected:
    virtual cstone::Box<typename Dataset::RealType> initImpl(int rank, int numRanks, size_t, Dataset& d,
                                                             IFileReader*) const = 0;

private:
    std::string settingsFile_;

    double nonFluidFrac_       = 0.0;
    bool   nonFluidContiguous_ = false;

    //! @brief apply the per-init non-fluid fraction option (no-op for frac <= 0)
    void applyNonFluidFraction(Dataset& simData) const
    {
        // the CLI value overrides the settings; on restarts, FileInit reads the input file's size-1 file
        // attributes into constants() (readFileAttributes), so a fraction persisted by the fresh run
        // (writeSettings with the nonFluidFracKey extra setting) is re-applied automatically. Marking is
        // idempotent (see markNonFluidFraction).
        double frac = nonFluidFrac_;
        if (frac <= 0.0)
        {
            auto it = constants().find(nonFluidFracKey);
            if (it != constants().end()) { frac = it->second; }
        }
        if (frac > 0.0) { markNonFluidFraction(simData.hydro, frac, nonFluidContiguous_); }
    }

    void runTagging(IFileReader* reader, Dataset& simData) const
    {
        if (settingsFile_.empty()) { return; }
        auto ids = cstone::toHost(simData.hydro.id);
        auto x   = cstone::toHost(simData.hydro.x);
        auto y   = cstone::toHost(simData.hydro.y);
        auto z   = cstone::toHost(simData.hydro.z);

        reader->setStep(settingsFile_, -1, FileMode::independent);

        try
        {
            std::vector<IdSelectionSphere> selSpheres;
            std::vector<unsigned>          sphereGroupIds;
            auto                           numSpheres4 = reader->fileAttributeSize(idTagFsStrings[0]);
            selSpheres.resize(numSpheres4 / IdSelectionSphere{}.size());
            reader->fileAttribute(idTagFsStrings[0], selSpheres.data()->data(), numSpheres4);

            if (reader->fileAttributeSize(idTagFsStrings[1]) != static_cast<int64_t>(selSpheres.size()))
                throw std::runtime_error("number of selection spheres != number of sphere ids");
            sphereGroupIds.resize(selSpheres.size());
            reader->fileAttribute(std::string(idTagFsStrings[1]), sphereGroupIds.data(), selSpheres.size());

            tagIdsInSphere(ids, x, y, z, 0, ids.size(), selSpheres, sphereGroupIds);
            if (reader->rank() == 0)
                std::cout << "Tagged particles in " << std::to_string(selSpheres.size()) << " spheres\n";
        }
        catch (std::exception& e)
        {
            if (reader->rank() == 0) std::cout << "No selection spheres applied because: " << e.what();
        }

        try
        {
            std::vector<uint64_t> selList;
            std::vector<unsigned> selListGroupIds;
            auto                  attr_size = reader->fileAttributeSize(idTagFsStrings[2]);
            selList.resize(attr_size);
            reader->fileAttribute(std::string(idTagFsStrings[2]), selList.data(), attr_size);
            if (reader->fileAttributeSize(idTagFsStrings[3]) != attr_size)
                throw std::runtime_error("id_selection_list size != group_ids size");
            selListGroupIds.resize(attr_size);
            reader->fileAttribute(std::string(idTagFsStrings[3]), selListGroupIds.data(), attr_size);

            tagIdsInList(ids, 0, ids.size(), selList, selListGroupIds);
            if (reader->rank() == 0) std::cout << "Tagged " << std::to_string(attr_size) << " particles from list\n";
        }
        catch (std::exception& e)
        {
            if (reader->rank() == 0) std::cout << "No selection IDs from list applied because: " << e.what();
        }

        reader->closeStep();
        simData.hydro.id = std::move(ids);
    }
};

template<class Dataset>
struct SimInitializers
{
    using InitPtr = std::unique_ptr<ISimInitializer<Dataset>>;

    static InitPtr makeEvrard(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeEvrardCooling(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeFile(std::string testCase, int initStep, IFileReader* reader);
    static InitPtr makeFileSplit(std::string testCase, int numsplits, IFileReader* reader);
    static InitPtr makeGreshoChan(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeKelvinHelmholtz(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeIsobaricCube(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeNoh(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeSedovGlass(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeSedovGrid();
    static InitPtr makeTurbulence(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeWindShock(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makePolytrope(std::string glassBlock, std::string settingsFile, IFileReader* reader);
    static InitPtr makeTDEOrbitInit(const std::string& filePath, int initStep, IFileReader* reader);
};

extern template struct SimInitializers<SimulationData<cstone::execution::Cpu>>;
extern template struct SimInitializers<SimulationData<cstone::execution::Gpu>>;

} // namespace sphexa
