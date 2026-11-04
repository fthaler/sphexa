# Milestone M2b — SPH empty-range hardening (zero-fluid ranks)

Lands the three SPH-layer guards that make a rank owning **zero fluid particles** survivable:
the `rhoTimestep` empty-range guard (both CPU and GPU paths), the type-generic
`cstone::minMax(Gpu)` hardening, and the `reduceDt` zero-group guard. Depends on **M2** (the
fluid SPH phase must exist — these guards harden its range-reduced calls) and must land
**before** the zero-fluid-rank config (b) runs and before M3/M4's GPU zero-fluid tests, which
depend on all three. Scope is deliberately tiny: three guarded call sites, no new plumbing,
plus the config (b) acceptance runs.

## Scope

- `sph/include/sph/ts_global.hpp`: empty-range guard in `rhoTimestep(first, last, d)`, both
  paths (`d.useGpu` branches), placed **before** the `divv.empty()` throws.
- `domain/include/cstone/primitives/primitives_gpu.cu`: type-generic `minMax(Gpu)` early
  return on empty ranges.
- `sph/include/sph/hydro_ve/momentum_energy_gpu.cu`: guard the `reduceDt<<<>>>` launch on
  `grp.numGroups > 0`.
- Acceptance runs: config (b) — a 2-rank mixed run where one rank owns zero fluid particles —
  on a CPU build **and** a CUDA build (the GPU part of this milestone needs a CUDA build).
- Nothing else: no field/propagator changes, no M2 mechanics, no M3 integrate changes.

## Context you need

**What config (b) is (runnable scenario):** a 2-rank `ve-pt` mixed run (M2's
`--nonfluid-frac` mechanism) configured so that one rank's `fluidLayout.back() == 0` —
`nFluid == 0` on that rank. Consequences exercised: empty fluid exchange requests
(`fluidHalos_.exchangeRequests` posts/matches nothing), empty group construction, and the
empty-range SPH calls `rhoTimestep(fluidFirst, fluidLast, d)` (plan's step-7 signature:
`first = fluidFirst`, `last = fluidLast`) and `computeMomentumEnergy` with
`fluidFirst == fluidLast == 0`. M2's config (a) zero-count-leaf case is a rank that *still
has fluid* — distinct from this. Construction: M2's spread rule (`id % stride`) never
empties a rank, so config (b) uses the contiguous id-keyed variant — mark `id < F·N` with
`F = 0.5` on 2 ranks (ids are rank-offset sequences, so rank 0's entire id range is marked
non-fluid → zero fluid on rank 0). Any deterministic id-keyed rule emptying exactly one rank
is acceptable; it must already pass `sync` (empty receive counts) before the SPH guards are
reached.

**Why only two call sites need guards** (the rest is naturally safe): GPU group data is
always resized to ≥ 1 element (`groups_gpu.cu:93,106`), so on a zero-fluid rank
`numGroups = 0` and group-indexed kernels no-op via zero-block guards
(`positions_gpu.cu:180`); the CPU path sets `numGroups = 1` with a single empty group
(`sph/include/sph/groups.hpp:65-70`) — harmless, nothing iterates its empty range;
`haloPeers` derives flags from layout ranges (`halo_peers.hpp:33`) — a rank with no fluid
peers posts and matches nothing. Exceptions needing their own guards: `reduceDt` (Fix 3)
and `rhoTimestep` (Fix 1).

**Facts already true in the code:** `accelerationTimestep` is already guarded
(`last <= first → infinity`, `ts_global.hpp:52`); CPU `minMax(Cpu)` on an empty range
returns `{+inf, -inf}` (`primitives/primitives_acc.hpp:64-65`).

## Implementation

### Fix 1 — `rhoTimestep` empty-range guard (`sph/include/sph/ts_global.hpp`, both paths)

Add, at the top of `rhoTimestep` (before the `divv.empty()` throws at `:82` (GPU) and
`:89` (CPU), so zero-*total*-particle ranks — where `divv` is legitimately unallocated —
are covered too):

```cpp
if (first >= last) return INFINITY;   // zero-fluid rank: identity for the caller's min-reduce
```

Semantics: an empty range contributes `INFINITY`, the identity for the caller's `min`
reduction — `computeTimestep` (`ts_global.hpp:101-113`) folds it into `std::min` and the
`MPI_MIN` Allreduce at `:111`, so the empty rank cannot drag the global dt. The guard must
precede the throws (not just the loops) because a zero-*total*-particle rank has `divv`
size 0 and must return `INFINITY`, not throw on `divv.empty()`.

Failure modes without the guard, verbatim from the plan:

- **CPU** (`:79,91-96`): the openmp `reduction(max : maxDivv)` never executes its loop, so
  `maxDivv` stays `-INFINITY` → `dt = Krho / |maxDivv| = 0` → `0` poisons the `MPI_MIN`
  reduce (`:111`) — a **global stall** (every rank's global dt collapses to 0; the run
  stops advancing).
- **GPU** (`:83-85`): `cstone::minMax(Gpu)` unconditionally copies back
  `thrust::minmax_element`'s iterators — on an empty range `minMaxElements.second` points
  *at* `last`, and the `cudaMemcpyAsync` reads the element at `last`
  (`primitives/primitives_gpu.cu:184-193`) → garbage `dt` (out-of-bounds read only when the
  range ends at the buffer end, i.e. only on a zero-fluid rank where
  `fluidLast == fluidFirst == 0` sits at the compaction start).

### Fix 2 — `cstone::minMax(Gpu)` hardening (`domain/include/cstone/primitives/primitives_gpu.cu`)

Add a type-generic early return at the top of the `minMax(execution::Gpu, ...)` template
(`:183-194`):

```cpp
if (first == last) return {std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()};
```

- **Why not `INFINITY`:** the template is instantiated for `unsigned` too
  (`primitives_gpu.cu:198`) — `INFINITY` would be a narrowing conversion there. `max()` /
  `lowest()` is the type-generic identity pair.
- **Semantics:** matches CPU's empty-range result `{+inf, -inf}`
  (`primitives/primitives_acc.hpp:64-65`) *in behavior* (max/min identities for any
  subsequent reduce) if not in exact values.
- Keep the Fix 1 `rhoTimestep` guard as the **primary** fix — this hardening removes the OOB
  read hazard class for all `minMax(Gpu)` callers, but `rhoTimestep` must not rely on
  min-element identity `numeric_limits<T>::max()` flowing through `Krho / |maxDivv|`
  arithmetic for its `INFINITY` contribution.

### Fix 3 — `reduceDt` zero-group guard (`sph/include/sph/hydro_ve/momentum_energy_gpu.cu:92-94`)

Wrap the launch:

```cpp
if (grp.numGroups > 0)
{
    constexpr LocalIndex threads = 256;
    const LocalIndex     blocks  = cstone::iceil(grp.numGroups, threads / GpuConfig::warpSize);
    reduceDt<<<blocks, threads>>>(grp.groupStart, grp.groupEnd, grp.numGroups, rawPtr(d.dtCourant), groupDt);
}
```

The kernel already self-guards (`if (grpIdx >= numGroups) return;`, `:58`) — the hazard is
the *launch configuration*, not the kernel body: `blocks = iceil(0, 8) = 0` is an
invalid-configuration launch, currently unchecked.

Failure mode, verbatim from the plan: neither a crash nor a wrong dt — `inf` is the correct
zero-fluid value (both paths seed the reduction with
`numeric_limits<float>::infinity()`, `momentum_energy.hpp:53`, `momentum_energy_gpu.cu:88`)
— the hazard is the pending non-sticky `cudaErrorInvalidConfiguration` surfacing at the
next unrelated `cudaGetLastError()` (e.g. the ij-loop's checks, `loop.cuh:513,567`) →
spurious `exit(EXIT_FAILURE)` elsewhere; `checkGpuErrors` inspects only the wrapped call's
return (`errorcheck.cuh:15-25`), so the launch error is invisible at its own site. This is
the only unguarded launch on the ve path; zero-fluid ranks hit it in the M2b/M4 GPU tests.
With the guard, the untouched `minDt_ve_device` keeps its seeded `inf` value and the
following `cudaMemcpyFromSymbol` (`:96`) delivers the correct `d.minDtCourant = inf`.

## Explicitly out of scope (deferred)

- **M2** step-1/step-3 fluid mechanics: `(ptype, m)` exchange, `fluidHalos_` + scratch
  buffers, peer-consistency asserts, saveFields rework — M2b builds on them, changes none.
- **M3** integrate changes: full-layout `driftGroups_`, masked energy update in
  `computePositions` (`positions.hpp`/`positions_gpu.cu`/`sph_gpu.hpp`), masked
  acceleration zeroing, fluid-only h update, timestep wiring.
- **M4 item 5**: the dedicated GPU unit test for a rank owning zero fluid particles — a
  *consumer* of these guards (M2b's config (b) is the CPU part plus the full-app GPU run;
  the gtest-level GPU edge-case test is M4's).

## Verification / acceptance

Config (b) on **CPU and GPU** (GPU requires a CUDA build): 2-rank run, one rank with
`fluidLayout.back() == 0`. **Construction gap — resolve before running:** M2's
`markNonFluidFraction` deliberately spreads with `id % stride`, which yields a
homogeneous mixture on every rank and can therefore *never* produce a zero-fluid
rank. Plan-gap options, in order of preference:

1. Temporary contiguous selection variant (`id < f*N`, the corner-chunk case the
   plan warns about for glass ICs — `id` = global SFC order there) behind an extra
   env var or flag, used only for this config; or
2. a hand-marked IC dump (set `ptype != 0` on a spatially contiguous sub-block so
   one rank's owned range is fully non-fluid), loaded via `FileInit`; or
3. a cstone-level 2-rank test with hand-set `ptype` on raw vectors (as in M2's
   harness) exercising the empty-request/empty-group path — sufficient for the
   CPU guards, insufficient for the full-app GPU run.

The command below assumes option 1 (`--nonfluid-frac 0.5` + contiguous variant):

```
mpirun -np 2 <build>/sphexa --prop ve-pt --init evrard --glass ./50c.h5 -n 50 \
    -s 10 -w 10 -o vept_b.h5 --nonfluid-frac 0.5
```

with the contiguous (non-spread) marking variant so rank 0 owns zero fluid particles
(`-w 10` is required or no dump is ever written, `sphexa.cpp:106-107`). Expected outcomes:

- **Computing rank** (rank 1): crash-free through all steps, finite dt (the `MPI_MIN`
  reduce yields a finite global dt), dumps written.
- **Empty rank** (rank 0): crash-free; local courant contribution `inf` (correct — nothing
  to compute); run completes, no hang, no stall, finite global dt on both ranks.
- **GPU build additionally**: no spurious CUDA errors — in particular no
  `cudaErrorInvalidConfiguration` surfacing as a later `exit(EXIT_FAILURE)` in the ij-loop
  checks (`loop.cuh:513,567`).

Regression: the M2 all-fluid config and config (a) (mixed, every rank has fluid) must behave
identically before/after these guards — all three are no-ops for non-empty ranges.

## Risks & gotchas

- **Guard placement matters**: the `rhoTimestep` guard must precede the `divv.empty()`
  throws (`:82,:89`), not just the loops — fixing only the loop bounds would leave the
  zero-total-particle config throwing.
- **Don't "fix" `rhoTimestep` by relying on Fix 2**: `minMax(Gpu)`'s empty-range maximum is
  `numeric_limits<T>::max()` (an integer-ish identity), not `inf`; the arithmetic
  `Krho / |maxDivv|` and the downstream `MPI_MIN` expect the `INFINITY` contribution from
  Fix 1. Fix 2 is defense-in-depth against the OOB read, Fix 1 defines the semantics.
- **The `reduceDt` bug is invisible at its own site**: `checkGpuErrors` checks only the
  wrapped call (`errorcheck.cuh:15-25`); a bare `<<<0, 256>>>` launch does not fail there.
  Any debugging of "random" GPU exits in M2b/M4 should first check whether this launch ran
  unguarded.
- **CPU empty-rank `numGroups == 1`** (`groups.hpp:65-70`) is *not* a bug — do not "fix" it
  to 0; group construction and the halo machinery are validated as-is by config (b).
- `accelerationTimestep` needs no change (`ts_global.hpp:52`) — resist refactoring it to
  share code with the new guard.
