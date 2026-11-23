# Milestone M5 — unconverged-particle removal (deferred follow-up)

**Status: COMPLETE — closed 2026-11-18 (G1–G6 all green, user-run).**

Follow-up from the M3 v1 decision (`removeUnconvergedParticles` rejected at startup + unconditional
throw): with the flag set (IC file attribute, ve parity — `particles_data.hpp:190`, no CLI flag),
ve-pt now removes unconverged **fluid** particles at the next sync instead of throwing. Hard
requirement (unchanged from the plan): only fluid particles with no fluid neighbor are ever
removed (`nc <= 1`, where `nc` is self-inclusive — see below); gravity particles (`ptype != 0`) are
never marked and never removed. Without the flag, the "Neighbor search did not converge" throw is
byte-identical to ve.

## Scope

1. `ve_hydro_pt.hpp`: drop the v1 `sync()` rejection; restore ve's conditional throw in integrate
   step 3; debug-only stray-mark guard; header-comment rework.
2. First-ever full-propagator MPI test (`domain/test/integration_mpi/unconverged_removal.cpp`).
3. App-level gate tooling (`particle-types/gates/m5/`, h5cc pattern; user-run; `--smoke` runs 1-rank without an allocation).
4. Docs (this file, `progress.md`, `overview.md`, `CONTEXT.md`).

## Design rationale (verified facts)

- **Marking is structurally fluid-only.** The only marking site is `updateSmoothingLength`
  (`sph/include/sph/find_neighbors.hpp:31-35`, `nc <= 1` → `keys[i] = removeKey`), called in
  integrate step 3 over the phase-7 **fluid groups**: owned fluid in the compacted layout
  (`ptype == 0` by construction of `stablePartition`) with the fluid tree view, so `nc` counts
  **fluid** neighbors only. Gravity particles live at `[nFluid:bufSize)` and are never visited;
  halo slots are never marked. A debug guard (`assertRemoveMarksFluidOnly_`) asserts this in
  debug builds whenever unconverged particles are reported.
- **Removal machinery is cstone-standard, unchanged.** Marks survive key recomputation
  (`sfc.hpp:286` CPU, `sfc_gpu.cu` GPU); marked particles sort beyond the tree cover (key
  `1 << 3*maxTreeLevel` ≥ every valid SFC key, and the tree spans `[0, 2^63)`), are counted into no
  leaf (`calculateNodeCount` ranges), fall outside every rank's assignment
  (`createSendRanges` lower-bounds on assignment boundaries) and are dropped by the next
  `domain.sync`/`syncGrav`, which shrinks all conserved arrays. `partition_` rebuilds
  `ordering_`/`orderingInv_` at the new `bufSize` each step, so the shrink cannot trip the
  size-mismatch throws; the step-4 restore is a gather via `orderingInv_`, never a key sort —
  precisely why `removeKey`-carrying particles survive the restore until the sync removes them.
- **Criterion stays ve's `nc <= 1`** (shared `updateSmoothingLength`). `nc` is self-inclusive:
  `updateSmoothingLengthIterative` stores `nc[i] = 1 + findNeighbors(...)`
  (`sph/include/sph/kernels.hpp:49,90`; the cstone return value excludes self,
  `domain/include/cstone/findneighbors.hpp:83`), so `nc <= 1` means **zero** fluid neighbors.
  `nc == 0` is unreachable (the `.nc` floor is 1, `kernels.hpp:88`); a particle with exactly one
  fluid neighbor has `nc == 2` and survives.
- Flag default stays `false` → behavior identical when the flag is unset (all M0–M4 gates
  unaffected).

### ve-parity nuances (documented, not fixed)

- Dumps written between marking (integrate step 3) and removal (next sync) still contain the
  doomed particles. Keys are not dumped → a restart from such a dump recomputes keys from
  positions and resurrects them — identical to ve.
- Conserved totals (mass/energy) intentionally drop while removals occur.

## Implementation

### 1. `main/src/propagator/ve_hydro_pt.hpp`

- `sync()`: the v1 rejection (`if (d.removeUnconvergedParticles) throw "unsupported in ve-pt v1"`)
  is replaced by a comment: with the flag set, removal happens at the next sync; without it,
  unconverged particles throw in integrate step 3.
- integrate step 3: ve parity with `ve_hydro.hpp:215-218`:
  ```cpp
  bool haveUnconvergedParticles = updateSmoothingLength(groups_.view(), d);
  if (haveUnconvergedParticles)
  {
      assertRemoveMarksFluidOnly_(domain, simData);
      if (not d.removeUnconvergedParticles) { throw std::runtime_error("Neighbor search did not converge\n"); }
  }
  ```
- `assertRemoveMarksFluidOnly_` (new, debug-only, gated like `assertFluidHalosConsistent`): counts
  `keys == removeKey` over the two non-owned ranges `[0:fluidFirst)` + `[fluidLast:bufSize)` of the
  permuted keys (`fluidLayoutHost_[domain.startCell()]`/`[domain.endCell()]`) and asserts zero.
  Gpu path uses `cstone::count(Gpu, ...)` (existing `uint64_t` instantiation,
  `primitives_gpu.cu:428`); Cpu path uses `std::count` — **deviation from the plan**, which
  assumed a CPU `cstone::count` inline in `primitives_acc.hpp`; no such function exists in this
  cstone version (Gpu-only, `primitives_gpu.h:171`). Asserts (aborts the rank) rather than throws,
  like the halo peer-consistency assert. No new `timer.step()`.
- File header: the v1-limitations paragraph is replaced by the removal description + the two
  ve-parity nuances above.
- **Halo-mark fix (found by the first 2-rank run of `GlobalUnconvergedRemoval`)**: `sync()` now
  calls `clearHaloRemovalMarks_` when a removal is pending (`bool removalPending_`, set at the
  integrate-step-3 marking site, cleared in `sync()` after `domain.sync`). The removal sync compacts
  the marks out of the owned range, but `domain::setupHalos` re-derives the halo keys through
  `computeSfcKeys`, which deliberately *preserves* `removeKey` (`sfc.hpp:286`, `sfc_gpu.cu:30`). A
  dropped particle's stale slot can therefore end up in `[endIndex, nParticlesWithHalos)` with its
  mark intact — a live mark in a non-owned slot that trips `assertRemoveMarksFluidOnly_` and leaves
  the halo keys inconsistent with their positions. The fix zeroes the halo key range with
  `cstone::fill` (the recompute skips slots that already hold a removal key, so the mark must be
  gone first) and recomputes it with `computeSfcKeys` on `cstone::sfcKindPointer(rawPtr(d.keys))` —
  the strong-key pointer conversion is required, `sfc3D` instantiates on `KeyType::ValueType` and
  does not accept a raw integer key type. Cost: one pass over the halo range, removal steps only.

### 2. `domain/test/integration_mpi/unconverged_removal.cpp` (+ CMake)

First full-propagator MPI harness: constructs `Domain` + `RemovalSimData` (a minimal
`SimulationData` without the cooling fields: `hydro` + `comm`) + `HydroVePtProp` directly and
drives it in sphexa.cpp loop order (`activateFields` → sync → computeForces → integrate → sync).
CPU + GPU variants share the template functions; GPU target mirrors `zero_fluid_rank_gpu`
(`SPH_EXA_PTYPE_GPU_TEST`, links `sph_gpu ryoanji` on top of `cstone_gpu`).

- IC (31 particles, open box `[0, 10]^3`, `ng0 = 8` so `ngmin = 2`): 8-particle fluid clump
  (2x2x2 grid, spacing 0.04, h = 0.08; search radius 2h = 0.16 > body diagonal 0.069 → each member
  sees all 7 mates, nc = 8 = ng0 → `updateH` fixed point → survives), 3 isolated fluid
  singletons (h = 0.05, ≥ 4 away from any other fluid → even the capped h growth ~8x cannot
  rescue → nc = 1 → doomed), 20 gravity particles (ptype = 1; never counted — the fluid tree view
  counts fluid slots only).
- Positive: flag set → after the final sync the global owned count dropped by exactly 3, the
  surviving id set is exactly clump + gravity (Allgatherv over owned ranges), zero `removeKey`
  marks in any buffer; a second computeForces + integrate proves post-removal stability (no new
  marks, no throw).
- Negative: same IC, flag unset → integrate throws "Neighbor search did not converge" on the
  ranks owning doomed particles. The throw happens after `applyPartition_` (a local gather), so
  no collective is pending; the per-rank outcome is Allreduced and asserted ≥ 1 rank.
- **MPI + gtest deadlock trap**: a bare `ASSERT_EQ` is a per-rank early return. When one rank
  failed the mark check while its peer entered the next `computeForces` collective, the job hung
  until the SLURM step timeout (observed as a 596 s "hang" that was not a simulation hang). The
  pass/fail decision is therefore `MPI_Allreduce`d (`globalLiveMarks`) and both ranks branch
  identically; `reportLocalMarks` prints `owned=/halo=/tail=` mark counts per rank on failure.
- Mark accounting uses the buffer index space: owned `[startIndex, endIndex)`, halo
  `[endIndex, nParticlesWithHalos)`, and a stale tail above `nParticlesWithHalos` (fields are not
  shrunk until the next `d.resize()`; `reallocate()` never shrinks). Only owned + halo marks are
  defects.
- `zero_fluid_rank.cpp` GPU case: `groups_gpu.cu` accepts only `warpSize` (32 under `__CUDACC__`),
  but this host TU sees `GpuConfig::warpSize == 64` (`gpu_config.cuh:28`), so
  `2 * GpuConfig::warpSize` handed the dispatcher 128 and threw "Unsupported spatial group size".
  The test now passes the production group size 64 explicitly.
- CMake guard: `if (EXISTS ${CMAKE_SOURCE_DIR}/main/src/propagator/ve_hydro_pt.hpp)` — standalone
  cstone builds (`project()` inside `domain/` makes `PROJECT_SOURCE_DIR` the domain dir; only
  main-tree builds have the main/ sibling) skip the test.

### 3. App-level gate tooling (`particle-types/gates/m5/`, see `run_gates.sh`)

- **IC source deviation from the plan**: the plan said "copy an existing IC (50c.h5)" — but
  `50c.h5` is the bare **glass block** (x/y/z only) fed to `--init evrard --glass`; it cannot be
  restarted via FileInit (missing conserved fields). The tools instead copy a **full dump**:
  `mixed-1rank.h5` (complete ve-pt snapshot: all conserved fields + `ptype` + step attributes
  incl. `gravConstant = 1`, `ng0 = 100`, `iteration = 10`) for the removal ICs and the plain `ve`
  dump `ve.h5` for the all-fluid gate.
- `craft_removal_ic.c` (`removal_ic.h`): `H5Ocopy`s the whole Step#0 group, then rewrites
  `ptype` (fluid = the 100 nearest particles to each of 2 cluster centers + the 1 nearest particle
  to each of 3 isolated sites; all others ptype = 1), shrinks the singletons' `h` to 1e-6 and
  zeroes their velocities (the h-shrink is **not** in the plan: without it, the capped h growth at
  ng0 = 100 reaches ~100x the glass h and the singletons converge onto the clusters instead of
  staying unconverged), and sets the step attributes `removeUnconvergedParticles = <flag>` and
  `iteration = 0` (so `-s 3 -w 1` dumps at iterations 1 and 2; removal lands at the iteration-1
  sync). The all-fluid mode only writes the attribute.
- `check_removal.c`: re-derives the doomed/cluster sets from the crafted IC (shared
  `selectFluid`), reads `id`+`ptype` from the last output step and asserts: count == IC - doomed;
  every gravity id survived with ptype preserved; doomed ids gone; cluster ids present. Exit 0/1.
- `check_pair.c`: bitwise HDF5 comparison of two dumps (`--vept` allows the extra `ptype` dataset),
  used by G5.
- `run_gates.sh`: the full G1–G6 command list (`all|G1..G6`), crafting the ICs into `$WORK`
  (default `build/gate-m5`, gitignored) and forcing `MPICH_GPU_SUPPORT_ENABLED` per launch. It also
  has `--smoke`: a 1-rank, no-allocation end-to-end run of the whole toolchain (negative throw,
  positive removal + `check_removal`, ve/ve-pt 20-step pair + `check_pair`) so the tooling can be
  validated on a login node with `MPICH_GPU_SUPPORT_ENABLED=0`.
- **GPU-aware-MPI env scoping (found by the first G1/G6 user runs)**: the driver used to
  `export MPICH_GPU_SUPPORT_ENABLED=0` globally, which leaked into the ctest gates. The GPU test
  binaries are compiled with `CSTONE_HAVE_GPU_AWARE_MPI` (`domain/CMakeLists.txt:31`), so
  `useGpuDirect` is true (`mpi_cuda.cuh:28-32`) and cstone passes **device** pointers to
  `MPI_Allreduce`; with GPU support disabled cray-mpich memcpys from the device address and every
  GPU-path test segfaults inside `MPIDI_Cray_shared_mem_coll_reduce` (core backtrace:
  `__memcpy_sve` ← `MPIR_Typerep_icopy` ← `MPIR_CRAY_Allreduce` ← `updateOctreeGlobalGpu` ←
  `Domain::sync`). The override is now per launch: `0` for the CPU app gates (G2/G3/G5/smoke), `1`
  for G1/G4/G6.
- **G6 tree**: G6 ran `ctest --test-dir build-cpu-gate`, which only builds a subset of the test
  targets (10 reported as "Not Run"). It now uses `--full-build` (default `$GPU_BUILD`). The full
  suite also needs a ≥3-node allocation — the tests launch up to 12 ranks at
  `--ntasks-per-node=4`.
- **G6 exclusion**: `ryoanji_cpu_unit_tests` fails on this host with 6 pre-existing numeric
  violations (`Gravity.Cartesian_M2P`, `P2PmsoftBase`, `P2PmsoftH`,
  `EwaldGravity.CombineMultipoleTrace`, `ewaldEvalMultipoleComplete`,
  `SingleParticleChangingGrid`; 1e-10/1e-6 tolerances) that reproduce standalone on the login node
  with the current `build/` binary — independent of M5, of the MPI environment and of the gate
  driver. G6 excludes it by default (`G6_EXCLUDE`, default `^ryoanji_cpu_unit_tests$`; override
  with `--g6-exclude REGEX`, or `G6_EXCLUDE= ./run_gates.sh G6` to run everything).
  `ryoanji_unit_tests` (the GPU suite) passes.

## Acceptance gates

| Gate | What | Command (summary) | Status |
|---|---|---|---|
| G1 | integration test CPU + GPU green | `run_gates.sh G1` (2 ranks, GPU-aware MPI) | **PASS** (2026-11-18, user-run) |
| G2 | negative: 2-rank CPU, flag unset → throw | `run_gates.sh G2` | **PASS** (2026-11-18, user-run; exit 134 + expected message) |
| G3 | removal, CPU 2-rank → `check_removal` green | `run_gates.sh G3` | **PASS** (2026-11-18, user-run; 65494→65491, 3/3 doomed removed, 200/200 cluster fluid kept, 0 non-fluid drops) |
| G4 | removal, GPU (`sphexa-cuda`) → `check_removal` green | `run_gates.sh G4` | **PASS** (2026-11-18, user-run; same report as G3) |
| G5 | no-op equivalence: all-fluid + flag, ve vs ve-pt bitwise | `run_gates.sh G5` | **PASS** (2026-11-18, user-run; 15 datasets bitwise-identical, ptype all zero) |
| G6 | regression: existing suites green (default-off) | `run_gates.sh G6` (full suite from `build/`, ≥3 nodes) | **PASS** (2026-11-18, user-run; 39/39 in 118.8 s, `ryoanji_cpu_unit_tests` excluded as the pre-existing host failure) |

## Gotchas carried into implementation

- Criterion stays ve's `nc <= 1` with self-inclusive `nc` — this means **zero** fluid neighbors
  (a particle with exactly one fluid neighbor has `nc == 2` and survives).
- No CLI/help/settings changes in sphexa.cpp; no new `timer.step()` calls; no fluid exchange inside
  `integrate` (M3 invariant); `find_neighbors.hpp` untouched (shared with ve/std propagators).
- `-s N` is an absolute iteration cap; the crafted ICs reset `iteration = 0` so `-s 3 -w 1`
  snapshots at iterations 1–2, both post-removal.
- Deterministic IC ⇒ deterministic survivor set; a cascade removing a clump member is a real
  signal (clump too sparse), not checker noise.
- Run clang-format in-tree (never /tmp); compare pre-existing violation counts before "fixing".

## Out of scope

ptype-filtering of shared observables; halo-`ax` readers; epoch exhaustion; upstream ve notes;
changing the flag default; any cstone modification.
