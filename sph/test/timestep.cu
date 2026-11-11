/*! @file
 * @brief  Tests for block time-steps related GPU device kernels
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>

#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/primitives_gpu.h"
#include "sph/particles_data.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/ts_global.hpp"

using namespace cstone;
using namespace sph;

TEST(TimestepGpu, Divv)
{
    using T                                  = double;
    thrust::device_vector<LocalIndex> groups = std::vector<int>{10, 20, 40};
    GroupView                         grpView{10, 40, 2, rawPtr(groups), rawPtr(groups) + 1};

    thrust::device_vector<T> divv(40);
    thrust::sequence(divv.begin(), divv.end(), 100);
    thrust::device_vector<float> groupDt(groups.size() - 1, 1e10f);

    float Krho = 0.2;
    groupDivvTimestepGpu(Krho, grpView, rawPtr(divv), rawPtr(groupDt));

    thrust::host_vector<float> probe = groupDt;

    EXPECT_NEAR(probe[0], Krho / 119, 1e-10);
    EXPECT_NEAR(probe[1], Krho / 139, 1e-10);
}

TEST(TimestepGpu, Acc)
{
    using T                                  = double;
    thrust::device_vector<LocalIndex> groups = std::vector<int>{10, 20, 40};
    GroupView                         grpView{10, 40, 2, rawPtr(groups), rawPtr(groups) + 1};

    thrust::device_vector<T> ax(40), ay(40), az(40), h(40);

    thrust::sequence(ax.begin(), ax.end(), 100);
    thrust::sequence(ay.begin(), ay.end(), 200);
    thrust::sequence(az.begin(), az.end(), 300);
    thrust::fill(h.begin() + groups[0], h.begin() + groups[1], 1.0);
    thrust::fill(h.begin() + groups[1], h.begin() + groups[2], 2.0);

    thrust::device_vector<float> groupDt(groups.size() - 1, 1e10f);

    float etaAcc = 0.2;
    groupAccTimestepGpu(etaAcc, grpView, rawPtr(ax), rawPtr(ay), rawPtr(az), rawPtr(h), rawPtr(groupDt));

    thrust::host_vector<float> probe = groupDt;

    EXPECT_NEAR(probe[0], etaAcc * std::sqrt((1.0) / std::sqrt(norm2(Vec3<T>{100 + 19, 200 + 19, 300 + 19}))), 1e-8);
    EXPECT_NEAR(probe[1], etaAcc * std::sqrt((2.0) / std::sqrt(norm2(Vec3<T>{100 + 39, 200 + 39, 300 + 39}))), 1e-8);
}

//! @brief M2b guard: rhoTimestep on an empty local range returns +inf before the divv availability
//! check — a zero-fluid rank has neither owned particles nor an allocated divv buffer
TEST(TimestepGpu, RhoTimestepEmptyRange)
{
    sphexa::ParticlesData<cstone::execution::Gpu> d;
    ASSERT_TRUE(d.divv.empty());

    auto dt = rhoTimestep(3, 3, d);
    EXPECT_TRUE(std::isinf(dt));

    // the guard must precede the divv.empty() throw: an empty range never throws
    EXPECT_NO_THROW(rhoTimestep(0, 0, d));
}

//! @brief control: a non-empty range still yields the finite Krho / |max(divv)| timestep
TEST(TimestepGpu, RhoTimestepFinite)
{
    sphexa::ParticlesData<cstone::execution::Gpu> d;
    d.divv = std::vector<float>{100, 119, 101};

    auto dt = rhoTimestep(0, d.divv.size(), d);
    EXPECT_FALSE(std::isinf(dt));
    EXPECT_NEAR(dt, d.Krho / 119, 1e-10);
}

//! @brief M2b guard: gpu::computeMomentumEnergy with numGroups == 0 must skip the reduceDt launch
//! entirely (a zero-block launch leaves a pending non-sticky cudaErrorInvalidConfiguration that
//! surfaces at the next unrelated call) and must leave dtCourant untouched
TEST(TimestepGpu, MomentumEnergyZeroGroups)
{
    sphexa::ParticlesData<cstone::execution::Gpu> d;
    d.dtCourant = std::vector<float>{1, 2, 3, 4, 5};

    GroupView grp{0, 0, 0, nullptr, nullptr};
    gpu::computeMomentumEnergy<false>(grp, nullptr, d, Box<double>{0, 1});

    // the skipped reduce leaves dtCourant intact and minDtCourant at the +inf seed
    EXPECT_EQ(toHost(d.dtCourant), (std::vector<float>{1, 2, 3, 4, 5}));
    EXPECT_TRUE(std::isinf(d.minDtCourant));

    // no pending CUDA error may survive the call
    checkGpuErrors(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

//! @brief M2b guard: minMax on an empty device range returns {max, lowest} without touching memory
TEST(TimestepGpu, MinMaxEmptyRange)
{
    DeviceVector<double> empty;
    auto [emptyMin, emptyMax] =
        minMax(execution::gpuDefaultStream, rawPtr(empty), rawPtr(empty) + empty.size());
    EXPECT_EQ(emptyMin, std::numeric_limits<double>::max());
    EXPECT_EQ(emptyMax, std::numeric_limits<double>::lowest());

    DeviceVector<double> v = std::vector<double>{3.0, -1.0, 2.0};
    auto [minV, maxV]    = minMax(execution::gpuDefaultStream, rawPtr(v), rawPtr(v) + v.size());
    EXPECT_EQ(minV, -1.0);
    EXPECT_EQ(maxV, 3.0);
}
