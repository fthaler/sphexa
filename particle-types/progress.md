# `ve-pt` Implementation Progress

Tracker for the particle type implementation. Authoritative scope and acceptance
criteria live in `overview.md` (shared invariants, dependency chain) and the
`milestone-*.md` files; this file only records **where we are**.

## How to use

- Checkboxes: `- [ ]` not started · `- [~]` in progress (add a one-line note)
  · `- [x]` done (verified per the milestone's Verification section).
- Gate rows: record the exact command(s) run, build type (debug/release, CPU/CUDA),
  rank count, and date in the session log at the bottom; check only when the
  milestone's full acceptance is met.
- Statuses in the table below: `not-started` / `in-progress` / `done` /
  `blocked` (state the blocker). Keep at most one milestone `in-progress`.
- Do not re-litigate design decisions here — they are fixed in `overview.md`.

## Milestone status

| M | File | Scope | Depends on | Status | Gate |
|---|---|---|---|---|---|
| M0 | `milestone-0.md` | cstone primitives (stable partition, fluid layout, masked fill, uint8 gather, `gatherArrays` hardening) + tests | — | done | unit tests, debug+release (+CUDA) |
| M1 | `milestone-1.md` | `ve-pt` skeleton, ptype end-to-end, partition→restore, CPU bitwise `ve` equivalence | M0 | not-started | CPU bitwise gate |
| M2 | `milestone-2.md` | fluid SPH phase: `(ptype, m)` exchange, `fluidHalos_`, fluid tree view/groups/pipeline, saveFields rework, `--nonfluid-frac`, 2-rank tests | M1 | not-started | all-fluid bitwise + 2-rank mixed configs |
| M2b | `milestone-2b.md` | SPH empty-range hardening (`rhoTimestep`, `minMax(Gpu)`, `reduceDt`) + zero-fluid-rank config | M2 | not-started | config (b), CPU + CUDA |
| M3 | `milestone-3.md` | integrate: drift groups, masked energy update, masked accel zeroing, fluid-only h update, GPU bitwise gate | M2b | not-started | GPU bitwise gate + 30% mixed smoke |
| M4 | `milestone-4.md` | tolerant restart reader, per-init option, restart gates, GPU edge cases | M3 (item 5 needs M2b) | not-started | restart gates 1+2, GPU zero-fluid test |

---

## M0 — cstone primitives (`milestone-0.md`)

Implementation:

- [x] `stablePartition` — fused, both permutations, CPU (`primitives_acc.hpp`) + GPU (`primitives_gpu.h`/`.cu`); exclusive-scan-of-**non-fluid**-flag formula
- [x] Fluid-layout helper — segmented `ptype == 0` count over `domain.layout()` + exclusive scan; fills `fluidLayoutHost_` + `fluidLayoutAcc_`, realloc to `numLeafNodes + 1` per call (`computeFluidLayout(exec, layout, ptype, acc, host, growthRate)` in `primitives_acc.hpp`/`primitives_gpu.h`, device count in `primitives_gpu.cu`)
- [x] Masked zero-fill — CPU `fillMasked` next to `fill`; GPU kernel + explicit instantiation (float/double dest, uint8_t mask)
- [x] `GATHER_RANGES(uint8_t, unsigned)` in `gather_halos_gpu.cu:60-74`
- [x] `gatherArrays` hardening — release-active `throw` on `array.size() < ordering.size()`, covering **both** `reorderArray` branches, strict `<`
- [x] CMake: no library changes (all GPU code stays in the existing `primitives_gpu.cu`); the three **test** CMakeLists were updated to register the new tests — deviation from plan item 6, required to build them

Tests / acceptance:

- [x] `stablePartition` unit tests, CPU + GPU: all-fluid, all-non-fluid, single fluid, alternating; both outputs written ∀i; `orderingInv_[ordering_[j]] == j`; **stability**; gather puts fluid at `[0:nFluid)` in order
- [x] Fluid-layout brute force: zero-count leaves, `nFluid < bufSize` padding, all-fluid ⇒ `fluidLayout == layout`
- [x] `gatherArrays` hardening test: size-0 field + non-empty ordering **throws**, both branches, run in **debug AND release**; existing `Layout.gatherArrays` stays green + shorter-ordering positive case
- [x] Masked-fill unit test, CPU + GPU: masked written, unmasked sentinel intact, empty-range no-op
- [x] 2-rank MPI gtest syncing/exchanging a `uint8_t` field through domain `exchangeHalos` (links only with the new instantiation) — `integration_mpi/exchange_ptype.cpp`, CPU + GPU targets, 2 ranks green
- [x] Domain unit-test suite green in debug + release; GPU tests under CUDA build — 339/339 CPU (debug+release, plain+OMP); `component_units_cuda` green (release+debug); 2-rank uint8 exchange green (CPU + GPU)

---

## M1 — `ve-pt` skeleton, CPU bitwise gate (`milestone-1.md`)

Implementation:

- [ ] `particles_data.hpp`: `FieldVector<uint8_t> ptype` + `"ptype"` appended at the **end** of both `fieldNames` and `dataTuple()`
- [ ] Init zero-fill `ptype` at all 12 anchors (14 paths): `RadialProfile` base; `GreshoChan`, `IsobaricCubeGlass`, `KelvinHelmholtzGlass`, `NohGlassSphere`, `SedovGrid` (after **second** resize), `SedovGlass`, `TurbulenceGlass`, `WindShockGlass`; `FileInit`/`TDEOrbitInit` (missing-ptype branch); `FileSplitInit` (always-fill)
- [ ] `ve_hydro_pt.hpp`: aliases (`ConservedFields` + `"ptype"`), `conservedFields()` filters `"ptype"`, members `ordering_`/`orderingInv_` (`AccVector<LocalIndex>`) + keys-sized `partitionScratch_`; **no** M2 members yet
- [ ] `sync()` verbatim copy (branching incl. `m` folded into conserved at `g == 0`); no ptype/m exchange yet
- [ ] `computeForces`: partition (step 4) after `d.resize`, flat-tuple `gatherArrays` apply with per-field `size == bufSize` **throw**, restore (step 9) via `orderingInv_` + `d.treeView` reset; SPH/gravity exactly as `ve`; `fillMassHalos` **kept**
- [ ] `saveFields`: ve's three passes + `"ptype"` appended to `indicesDone`/`namesDone` iff absent
- [ ] `integrate` copied unchanged
- [ ] `ve_hydro_pt.cpp` factory + explicit instantiation; registration: `propagator.h`, `factory.hpp` (`ve-pt`), `propagator/CMakeLists.txt` `PROP_SOURCES`, `compare_develop.py` literal (do **not** touch `FIELDS`)
- [ ] Timers firing: `partition`, `partitionRestore` (other three reserved for M2)

Acceptance (CPU build, 1 rank, no `-f`, `-w 10` required):

- [ ] Gate: `ve` vs `ve-pt` evrard glass bitwise-identical per dataset (h5py check in `milestone-1.md`), ve-pt `ptype` all zero, same step groups

---

## M2 — fluid SPH phase (`milestone-2.md`)

computeForces steps:

- [ ] Step 1: `domain.exchangeHalos(std::tie(d.ptype, d.m), ax, keys)` right after sync; single step-8 marking site before `d.resize`; step-1 ordering asserted (sync → mark → resize → exchange → fluid counts)
- [ ] Step 2: fluid leaf counts + `fluidLayout` (device-side on GPU with D2H; both spaces filled; realloc per call); `nFluid = fluidLayout.back()`; timer `fluidLayout`
- [ ] Step 3: `fluidHalos_.exchangeRequests(...)` on second `Halos` instance (lazy-constructed in first `sync()` from `rank_` + `simData.comm`); invariants (a)–(d) pinned in code; dedicated send+receive `AccVector<uint64_t>` scratch
- [ ] Step 4: fused partition with real `nFluid` (replaces M1 `d.size()` placeholder); flag-gated trio `gatherArrays` call; per-field `size == bufSize` throw fold; timer `partition`
- [ ] Step 5: fluid tree view (focus tree + swapped layout pointer) assigned to `d.treeView`; `totalBodies = nFluid` plumbing through `neighborhood*`/`find_neighbors*`; timer `fluidView`
- [ ] Step 6: `fillMassHalos` **dropped exactly here**
- [ ] Step 7: SPH pipeline on `[fluidFirst:fluidLast)` with `fluidHalos_.exchangeHalos` (pre-sized arrays assert at call sites); state-aware field rotation coupled on the trigger flag; timer `fluidHaloExchange`
- [ ] Step 8: output trio trigger (all-three-or-nothing) + `setDependent` marking + permanent allocation; trio stays out of `DependentFields`
- [ ] Step 9: inverse-permutation restore incl. flag-gated trio call; `d.treeView` reset; timer `partitionRestore`

Other:

- [ ] `saveFields` rework: pass-2 `gradh` symmetry (both branches per the c13 starvation argument); pass 3 **dropped**
- [ ] `markNonFluidFraction(d, frac)` in `init/utils.hpp` (id-keyed, `id % stride` spread, GPU D2H/H2D mechanics, h sanity assert) + `--nonfluid-frac` in `sphexa.cpp` + `printHelp`
- [ ] `cstone::Halos` const getters `outgoing()`/`incoming()`
- [ ] Propagator members: `fluidHalos_`, `fluidLayoutHost_`/`fluidLayoutAcc_`, two fluid-halo scratch buffers, trigger flag

Tests / acceptance:

- [ ] 2-rank cstone gtests (raw vectors): partition→fluidHalos→restore round trip, zero-count-leaf handling, zero-fluid-rank request emptiness
- [ ] Peer-consistency assertion (debug-only Allreduce over send/recv count matrix + local checks)
- [ ] Gate: all-fluid config still bitwise-identical to `ve` (same command + h5py check as M1)
- [ ] Gate: 2-rank MPI test — conserved quantities identical across round trip; peer-consistency asserts pass
- [ ] Gate: 2-rank mixed config (a) low-fraction — crash-free, conserved momentum (not bitwise)
- [ ] Gate: 2-rank mixed config (b) zero-count-leaf — crash-free, correct halo coverage
- [ ] Mixed configs pass `-f` incl. `divv,curlv,gradh` at least once (exercises step-8 path; requires saveFields rework first)

---

## M2b — SPH empty-range hardening (`milestone-2b.md`)

Implementation:

- [ ] Fix 1: `rhoTimestep` empty-range guard (`ts_global.hpp`), both paths, **before** the `divv.empty()` throws, returns `INFINITY`
- [ ] Fix 2: `minMax(Gpu)` type-generic early return `first == last` → `{max(), lowest()}` (`primitives_gpu.cu`)
- [ ] Fix 3: `reduceDt<<<>>>` launch guarded on `grp.numGroups > 0` (`momentum_energy_gpu.cu:92-94`)

Acceptance:

- [ ] Config (b) construction gap resolved (option 1 contiguous variant / option 2 hand-marked IC / option 3 cstone-level test)
- [ ] Gate: config (b) 2-rank run on CPU build — empty rank crash-free, `inf` local courant, finite global dt, no stall
- [ ] Gate: config (b) on CUDA build — additionally no spurious CUDA errors (no `cudaErrorInvalidConfiguration`)
- [ ] Regression: M2 all-fluid config and config (a) behave identically before/after the guards

---

## M3 — integrate (`milestone-3.md`)

Implementation:

- [ ] A: `computePositions` per-particle energy mask — `ptype` param through CPU (`positions.hpp`) + GPU (`positions_gpu.cu` kernel/launcher/`POS_GPU` + 4 instantiations, extern in `sph_gpu.hpp`); `nullptr` precedent for all other call sites
- [ ] B: computeForces step 10 — masked zero-fill of non-fluid owned `ax/ay/az` over `[startIndex():endIndex())` **after** phase-9 restore, **before** gravity, run unconditionally
- [ ] C: overridden `integrate` — step 1 timestep over owned range (full SFC); step 2 KDK drift all owned (GPU: warp-sized groups into new `driftGroups_` member over full `d.treeView`); step 3 fluid-only `updateSmoothingLength` under cached `ordering_` with phase-7 `groups_`; step 4 second inverse permutation (fixed list + trio); `driftGroups_` member landed
- [ ] D: `removeUnconvergedParticles` v1 — unconditional throw on unconverged; startup error "unsupported in ve-pt v1" at first `sync()` when flag set

Acceptance:

- [ ] Gate: all-fluid bitwise equivalence vs `ve` on a **GPU build** (same two-run command + h5py check)
- [ ] Gate: 30% non-fluid smoke run, 1 rank + 2-rank variant — kinetic energy and linear/angular momentum drift judged; unpolluted observables
- [ ] Observables: `eInt` non-fluid offset exactly 0; `etot` carries time-varying `egrav` offset (finite, not constant); non-fluid `temp`/`du_m1`/`alpha` keep `initFieldsAtRest` values; non-fluid `h` finite/sane

---

## M4 — polish (`milestone-4.md`)

Implementation:

- [ ] Item 1: `IFileReader::datasetNames()` virtual + `H5PartReader` impl + `UnimplementedReader` override; `restoreDataset` skips missing datasets by name — `ptype` zero-filled, any other missing field = hard error; `readField` stays throwing (no per-rank catch)
- [ ] Item 2: proper per-init option — `markNonFluidFraction` hooked beside `runTagging` after `initImpl` (`isim_init.hpp`), `InitSettings` entry, CLI `--nonfluid-frac` kept as the setter; zero-fill stays at the M1 anchors
- [ ] Item 5: GPU unit test for a rank owning zero fluid particles (2-rank, needs M2b guards)
- [ ] Item 5: timing cleanup — five timers wrap exactly their regions (`fluidLayout`, `partition`, `fluidView`, `fluidHaloExchange`, `partitionRestore` + integrate 3/4)
- [ ] Item 6: document `nc <= 1` unsupported in v1; document stale/undefined SPH outputs for non-fluid particles

Acceptance:

- [ ] Gate 1 (item 3): mixed run → restart from first snapshot → re-run → bitwise snapshot equality **incl. `ptype`** (match snapshots by `iteration` attribute; same rank count)
- [ ] Gate 2 (item 4): restart a plain ve dump with `--prop ve-pt` — completes; `ptype` column all zero; conserved fields restored
- [ ] Item 2: per-init settings path reproduces the M2 flag's marking; idempotent re-marking on ptype-bearing restarts

---

## Deferred / follow-ups (not in any milestone)

- [ ] `ptype`-filtering of shared observables (`conserved_quantities.hpp:85-101`, `conserved_gpu.cu:92-97`) — do **not** filter in v1
- [ ] Actual removal of unconverged particles (removeKey + key sort) — v1 throws instead
- [ ] Halo-`ax` readers (non-fluid halo accelerations stay garbage forever)
- [ ] Epoch exhaustion vs `MPI_TAG_UB` (pre-existing; base `P2pTags::haloExchange = 7000`)
- [ ] Upstream: ve saveFields pass-2 garbage `gradh` column + pass-3 OOB write (`iad_divv_curlv_gradh.hpp:49`/`_gpu.cu:48`) — consider filing
- [ ] Fluid instance tag/epoch base escalation (only if M2 MPI hygiene ever breaks)

## Gotcha quick list (from milestone files; see each for detail)

- `f` = exclusive scan of the **non-fluid** flag — literal flag reading corrupts every partition (M0).
- `fieldNames`/`dataTuple()` order match is load-bearing; `static_assert` compiles out on GCC ≤ 11 (M1).
- CPU halo-`ptype` correctness in M1 is a `std::vector` zero-init accident — never extend that gate to GPU; never drop the init zero-fills (M1).
- `fillMassHalos` must survive until M2 step 1, then be dropped exactly there (M1/M2).
- Never cache raw pointers across a `gatherArrays` apply (swap branch toggles buffer identity) (M1/M2).
- Two index spaces; exactly four layout transitions per step, all via `partition_`/`unpartition_`; `ordering_` is a per-step cache, never reused across a `sync` (M2/M3).
- Release-safe checks **throw**, not `assert` (M0/M1/M2).
- `SedovGrid` fill goes after the second resize (M1).
- Do not run acceptance with `-f divv,curlv,gradh` until M2's saveFields rework (ve pass-3 OOB) (M1).
- `compare_develop.py`: add only the `"ve-pt"` literal; never `ptype` in `FIELDS`; cannot drive gates (M1/M4).
- Do not blanket-fill `ptype` after `initImpl` (clobbers restored non-zero ptype) (M4).
- Do not apply `nFluid` gating to gravity; no fluid exchange inside `integrate` (M3).

---

## Session log

Append one line per work session: `- YYYY-MM-DD — <milestone> — what was done / gate outcome (build, ranks)`.

- 2026-11-04 — — `progress.md` created from `overview.md` + milestone files; all milestones not-started.
- 2026-11-04 — M0 — implemented all five primitives + tests (`primitives_acc.hpp`, `primitives_gpu.h/.cu`, `gather_halos_gpu.cu`, `layout.hpp`; new tests `unit/primitives/stable_partition.cpp`, `unit_cuda/primitives/stable_partition.cu`, `integration_mpi/exchange_ptype.cpp`). CPU suites green: 339/339 debug+release (plain + OMP). CUDA cstone builds (release+debug, standalone, `--extended-lambda` needed for standalone builds) compile clean. GPU + 2-rank MPI gate runs pending (no GPU/MPI in dev environment).
- 2026-11-04 — M0 — fixed two issues found at the GPU/MPI gates: (1) host-side write to device memory in `computeFluidLayoutGpu` (last-element zero moved into `fluidLeafCountsKernel`); (2) exchange_ptype verification assumed buffer index == global index — the post-sync buffer is SFC-ordered (Hilbert ≠ listed order), now verifies per particle via unique x. **Gates green**: CPU 339/339 debug+release (plain+OMP); `component_units_cuda` release+debug; 2-rank `exchange_ptype` CPU + GPU (release+debug) + `exchange_halos`/`domain_2ranks` regression. M0 done.
- 2026-11-04 — — M0 verified: standalone cstone builds `build-cstone-{debug,release}` (CPU) and `build-cstone-cuda{,-debug}` (CUDA, with `-DCMAKE_CUDA_FLAGS=--extended-lambda`).
- 2026-11-04 — M0 — code-review fixes: `<cassert>` include in `primitives_acc.hpp`; shared test-verification header `unit/primitives/stable_partition_checks.hpp` (dedupes CPU/GPU checkers + random driver); GPU `stablePartition` tests now also run the spec'd gather check; `computeFluidLayout(Gpu)` turned into extern template + explicit instantiation (launch helper now file-local); deduped owned-range fill in `exchange_ptype.cpp`; comment wording aligned with CONTEXT.md; CMake checkbox corrected above.
- 2026-11-04 — M0 — second code-review round: `stable_partition.cu` gather check now round-trips the ordering through a `DeviceVector<LocalIndex>` (device span into `gather` — fixes host-pointer-into-kernel spec defect); gather verification deduped into `stable_partition_checks.hpp` as `checkGatherResult` (host buffers, shared by CPU/GPU tests); unique-x global lookup deduped in `exchange_ptype.cpp` via file-local `globalIndex` helper; `extern` added to the `computeFluidLayout(Gpu)` declaration in `primitives_gpu.h`; `constexpr`→`const` for `thisExampleRanks` (matches `domain_2ranks.cpp`); SFC-comment typo fixed. Verified: full CPU unit suite 339/339 in `build-cstone-debug` and `build-cstone-release`; `component_units_cuda` compiles+links in `build-cstone-cuda` and `build-cstone-cuda-debug`; `exchange_ptype` (CPU debug+release) and `exchange_ptype_gpu` (CUDA release+debug) compile clean. CUDA/MPI compile-only this session (no GPU/mpirun available).
