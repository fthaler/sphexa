/*! @file
 * @brief Tests for the CPU empty-range timestep guards (M2b hardening, M4 item 5 unit coverage)
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <vector>

#include "sph/particles_data.hpp"
#include "sph/ts_global.hpp"

namespace sphexa
{

//! @brief M2b guard: rhoTimestep on an empty local range returns +inf before the divv availability
//! check — a zero-fluid rank has neither owned particles nor an allocated divv buffer
TEST(TimestepCpu, RhoTimestepEmptyRange)
{
    ParticlesData<cstone::execution::Cpu> d;
    ASSERT_TRUE(d.divv.empty());

    auto dt = sph::rhoTimestep(3, 3, d);
    EXPECT_TRUE(std::isinf(dt));

    // the guard must precede the divv.empty() throw: an empty range never throws
    EXPECT_NO_THROW(sph::rhoTimestep(0, 0, d));
}

//! @brief control: a non-empty range still yields the finite Krho / |max(divv)| timestep
TEST(TimestepCpu, RhoTimestepFinite)
{
    ParticlesData<cstone::execution::Cpu> d;
    d.divv = std::vector<sph::SphTypes::HydroType>{100, 119, 101};

    auto dt = sph::rhoTimestep(0, d.divv.size(), d);
    EXPECT_FALSE(std::isinf(dt));
    EXPECT_NEAR(dt, d.Krho / 119, 1e-10);
}

//! @brief a non-empty range without an allocated divv buffer still throws (unchanged semantics)
TEST(TimestepCpu, RhoTimestepThrowsWithoutDivv)
{
    ParticlesData<cstone::execution::Cpu> d;
    EXPECT_THROW(sph::rhoTimestep(0, 5, d), std::runtime_error);
}

} // namespace sphexa
