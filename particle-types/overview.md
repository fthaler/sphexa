# Particle Type Support (`ve-pt`) — Milestone Overview

This directory splits `../particle-types-plan.md` (the authoritative, fully
cross-referenced plan) into per-milestone work packages. Read this file first;
then work from exactly one milestone file. When in doubt, the plan file wins.

Companion background: `../particle-types.md` (the original design sketch) and
`../CONTEXT.md` (repo domain model).

## Goal

New propagator `ve-pt` (`main/src/propagator/ve_hydro_pt.hpp`) supporting two
particle classes via a new `uint8` field `ptype`: `0` = fluid (SPH + gravity),
non-zero = gravity-only. Non-fluid particles keep their IC `h` (gravity
softening) and are integrated KDK from gravity accelerations only.

## Milestones and dependency chain

| File | Scope | Depends on |
|---|---|---|
| `milestone-0.md` | M0 — new cstone primitives (stable partition, fluid layout helper, masked zero-fill, `GATHER_RANGES` uint8, `gatherArrays` hardening) + unit tests | — |
| `milestone-1.md` | M1 — `ve-pt` skeleton round trip: ptype field end-to-end, partition→restore, bitwise `ve` equivalence, CPU only | M0 |
| `milestone-2.md` | M2 — fluid SPH phase: halo `ptype`+`m` exchange, fluid tree view/groups/pipeline, `fluidHalos_`, saveFields rework, `--nonfluid-frac`, 2-rank tests | M1 |
| `milestone-2b.md` | M2b — SPH empty-range hardening (`rhoTimestep`, `minMax(Gpu)`, `reduceDt`) + zero-fluid-rank config (b) | M2 |
| `milestone-3.md` | M3 — integrate: drift groups, masked energy update, masked acceleration zeroing, fluid-only h update, GPU bitwise gate | M2b (guards); GPU gate needs M0+M2 GPU paths |
| `milestone-4.md` | M4 — polish: tolerant restart reader, proper per-init option, restart acceptance gates, GPU edge cases | M3 (item 5 needs M2b) |

## Design decisions taken (shared, do not re-litigate)

- Non-fluid particles are integrated (KDK) from gravity accelerations only;
  their `h` stays constant. Gravity **accumulates** into `ax/ay/az`, so
  non-fluid accelerations are masked to zero before each gravity traversal
  (computeForces step 10, M3).
- Test inits gain `markNonFluidFraction(d, frac)` (id-keyed selection,
  rank-count-independent) at **M2** behind a temporary `--nonfluid-frac` CLI
  flag; the proper per-init option lands at M4.
- Restarts from files without `ptype` default it to 0 (all fluid): tolerance
  lives in `restoreDataset` (skip missing datasets by name + zero-fill, via
  the new `IFileReader::datasetNames()` — M4 item 1); `readField` itself stays
  **throwing** (`ifile_io_hdf5.cpp:406-411`).
- Halo `m` is exchanged with the full pattern right after `sync`, in the same
  call as `ptype`: sync delivers halos only for `x, y, z, h`
  (`domain.hpp:481-489`, contract `:132-139`), and `computeXMass` +
  gravity's upsweep consume `m` over the whole buffer incl. halos
  (`gravity_wrapper.hpp:128-133` GPU, `:62-70` CPU; as `nbody.hpp:130`) —
  they need valid halo `m`. Replaces ve's homogeneous `fillMassHalos`;
  non-fluid halo slots must **not** be left untouched.
- **Every ve-pt dump carries `ptype`**: `conservedFields()` still excludes it
  (default output selection stays ve-identical, `sphexa.cpp:139`), but ve-pt's
  `saveFields` appends `"ptype"` — dumps write only `outputFieldIndices`
  (ve's output loop, `ve_hydro.hpp:222-287`), so without this a mixed-run
  restart silently zero-fills everyone to fluid. Canonical statement in
  particle data; column mechanics in saveFields.

## Departures from the sketch in `particle-types.md`

The sketch prescribes stable type re-sorting (`particle-types.md:19,31`) —
kept. Three departures:

1. The sketch's "build a local tree using all SPH particles" (`:34`, overview
   `:20-21`) is **not** done — the fluid view reuses the focus tree with a
   swapped layout (computeForces step 5, M2).
2. Its "re-sort local particles by SFC key" (`:37`) is replaced by the
   **inverse permutation** (steps 4/9) — the `removeKey` values written into
   `keys` make any key re-sort incorrect.
3. Its "find the split … as well as halo offsets for SPH particles"
   (`:32-33`) is replaced by the fluid-only leaf-count scan (`fluidLayout`) +
   a second `cstone::Halos` instance (steps 2-3, M2).

## Per-step flow of `ve-pt` and where each step lands

`computeForces` steps (numbering per the plan):

1. Sync on **all** particles (`syncGrav` iff `d.g != 0`, else `domain.sync`
   with `m` folded into conserved) + `domain.exchangeHalos(std::tie(ptype, m),
   ax, keys)` + step-8 marking site + `d.resize` → **M2** (sync/resize
   skeleton in **M1** without the ptype/m exchange)
2. Fluid leaf counts + `fluidLayout` (device-side on GPU, host on CPU) →
   primitive in **M0**, wired in **M2**
3. `fluidHalos_.exchangeRequests(...)` on a second `cstone::Halos` instance +
   invariants (a)-(d) + scratch spec → **M2**
4. Fused `stablePartition` + apply `ordering_` to the fixed gather set
   (+ flag-gated trio call) → primitive in **M0**, wired in **M1** (fixed
   list) / **M2** (trio)
5. Fluid tree view (focus tree + swapped layout pointer) + `totalBodies`
   plumbing → **M2**
6. `fillMassHalos` dropped exactly when the `(ptype, m)` exchange lands →
   **M2** (kept in **M1**)
7. SPH pipeline on the fluid range with `fluidHalos_` exchanges + state-aware
   field rotation → **M2**
8. Output trio `divv/curlv/gradh` permanently allocated (trigger flag,
   `setDependent` marking) → **M2**
9. Apply cached inverse permutation → full SFC order; reset `d.treeView` →
   **M1** (trio addition in **M2**)
10. Masked zero-fill of non-fluid owned `ax/ay/az` before gravity → **M3**

`saveFields` → ve version + ptype append in **M1**; rework (pass-2 `gradh`
symmetry, pass-3 drop) in **M2**. `integrate` steps 1-4 → **M3**.

## Cross-cutting invariants (load-bearing — restate and assert in code)

- **Two index spaces**: fluid-compacted (`[fluidFirst:fluidLast)`) vs full SFC
  (`[0:bufSize)`). Exactly four layout transitions per step (steps 4/9,
  integrate 3/4), all via one helper pair `partition_/unpartition_`. Never
  hand a fluid index to a full-layout routine (esp. `rhoTimestep`, gravity,
  output) or vice versa. `ordering_` stays valid between step 9 and integrate
  step 3 (no `sync` between); it is a per-step cache, rebuilt in the next
  computeForces step 4, never reused across a `sync`.
- **Step-1 ordering**: sync → mark (step 8) → `d.resize` → ptype/m exchange →
  fluid counts (step 2). A stale halo `ptype` yields a wrong `fluidLayout`
  surfacing as an MPI size-mismatch hang, not a clean error.
- **MPI hygiene (step-3 invariant list)** — emergent fencing from the two
  `Alltoall`s plus complete-on-return exchanges; violations hang or corrupt:
  (a) no main-instance `exchangeHalos`/`exchangeRequests` from the step-3
  Alltoall until the next sync's `exchangeRequests` Alltoall; (b) no fluid
  `exchangeHalos` after the step-7 exchanges until the next `sync`'s
  `exchangeRequests` (`domain.hpp:211-214`); (c) `fluidHalos_.exchangeRequests`
  before its first `exchangeHalos`; (d) `fluidHalos_` gets its own GPU scratch
  buffers; `ax`/`keys` never aliased.
- **Whole-leaf halo delivery premise**: the full sync delivers complete halos
  per flagged leaf (`exchange_keys.hpp:42-43,80-88`), so per-leaf fluid counts
  are peer-consistent and `fluidLayout` needs no communication.
- **Never restore by re-sorting on `keys`**: unconverged particles get
  `keys[i] = removeKey` (`find_neighbors.hpp:32`, `find_neighbors_gpu.cu:34`);
  restore is always the cached inverse permutation as a gather.
- **Run-constant flags**: the step-8 trigger flag and the two-list gather
  selection are derived from `d.outputFieldIndices` (set once,
  `sphexa.cpp:139`, before the loop `:167`). A mid-run true→false flip throws
  at the leg-1 `acquire`; test the flag boolean, not membership lookups.
- **Stale-content invariant**: `d.resize` leaves grown non-conserved fields
  holding stale contents — safe because nothing consumes them between the
  resize and their first writer (per-field analysis in the plan/M2).
- **Halo accelerations of non-fluid particles stay garbage forever** — never
  read today (step 10); any future halo-ax reader breaks silently.
- **`ptype` is written in every ve-pt dump** and registered conserved (never
  dependent) so it survives repartition and restart.

## Overall file map

| Area | Files | Lands in |
|---|---|---|
| New primitives | `domain/include/cstone/primitives_acc.hpp`, `primitives_gpu.h`, `primitives_gpu.cu`, `halos/gather_halos_gpu.cu` | M0 |
| Particle data | `sph/include/sph/particles_data.hpp` (+ init zero-fill anchors in `main/src/init/*`) | M1 |
| New propagator | `main/src/propagator/ve_hydro_pt.hpp`, `ve_hydro_pt.cpp`, `propagator.h`, `factory.hpp`, `main/src/propagator/CMakeLists.txt`, `compare_develop.py:67` | M1 (fluid members M2, drift groups M3) |
| Fluid mechanics | same propagator + `cstone::Halos` peer getters (`halos.hpp`) | M2 |
| SPH layer | `sph/include/sph/positions.hpp`, `positions_gpu.cu`, `sph_gpu.hpp`, `ts_global.hpp`, `hydro_ve/momentum_energy_gpu.cu` | M2b (guards) / M3 (positions mask) |
| Init/restart | `main/src/init/utils.hpp`, `sphexa.cpp`, `main/src/init/file_init.hpp`, `io/ifile_io.hpp`, `ifile_io_hdf5.cpp` | M2 (`markNonFluidFraction`) / M4 (tolerant reader, per-init option) |

## Conventions

- `file:line` references are relative to the repo root at plan-writing time;
  trust the plan's references (they were verified then) — spot-check only when
  something essential looks inconsistent.
- Timers (`timer.step(...)`, no registration): `fluidLayout`, `partition`,
  `fluidView`, `fluidHaloExchange`, `partitionRestore`.
- Acceptance gates run as plain `sphexa` CLI invocations (exact commands in
  the milestone files) plus small h5py checks; `compare_develop.py` cannot
  drive the gates (rtol diff, no exit code).
- M1 gate is **CPU only**; the GPU gate (same two-run bitwise check) lands at
  M3. 2-rank gtest harnesses live at cstone level
  (`domain_2ranks.cpp` via `addCstoneMpiTest`) — no 2-rank ve-pt propagator
  harness exists.
- Buffer geometry reminder: local buffer is `[0:start)` prefix halos,
  `[start:end)` owned, `[end:size)` suffix halos; SFC-sorted over the whole
  buffer after sync (`domain.hpp:141-144,499-500`,
  `buffer_description.hpp:29-45`). The compaction is **stable**: fluid subset
  stays SFC-sorted in leaf order with the same three-segment geometry.

## Shared risks (details in milestone files)

- Two index spaces (largest bug source) — see invariants above.
- Stale neighbor lists across layout switches: any kernel using
  `d.neighborhood` or `groups_` must only run in the layout it was built in.
- Zero-fluid ranks: group machinery naturally safe; exceptions needing their
  own guards: `reduceDt` and `rhoTimestep` (M2b). M2b/M4 cover CPU and GPU.
- `removeUnconvergedParticles`: v1 = startup error if flag set + unconditional
  throw on unconverged particles (M3).
- Fluid particles with `nc <= 1`: documented unsupported in v1 (M4).
- SPH quantities of non-fluid particles are stale/undefined in v1 for
  fluid-only `-f` outputs; default output set holds only finite values.
  `ptype`-filtering of shared observables is a follow-up (do not filter in v1
  — `nbody` has no `ptype`).
- Non-fluid `h` stays at IC value; feeds `egrav` (`gravity_wrapper.hpp:143`)
  and `accelerationTimestep` (`ts_global.hpp:66,105`) — init option asserts it
  is finite/sane.
- Performance: only MPI-latency-scale addition is the single `(ptype, m)`
  full-pattern exchange per step; the four full-buffer permutations per step
  are bandwidth-negligible. No mitigation beyond the timers.
- **Config (b) construction gap (plan omission)**: M2's `markNonFluidFraction`
  spreads with `id % stride`, so it can never produce a zero-fluid rank —
  M2b's config (b) needs a temporary contiguous selection variant, a
  hand-marked IC, or a cstone-level test instead (options in
  `milestone-2b.md`, Verification).
- Upstream note (pre-existing in `ve`, avoided by ve-pt): ve's saveFields
  pass 2 writes a garbage `gradh` column when gradh is a requested output, and
  pass 3 writes through gradh's empty home buffer whenever divv/curlv are
  requested (`iad_divv_curlv_gradh.hpp:49`/`_gpu.cu:48`) — an OOB write.
  Consider filing upstream.
