# Milestone M4 — polish: tolerant restart reader, proper per-init option, acceptance gates, GPU edge cases

M4 is the closing milestone: **no new core mechanics** (all landed M0–M3). It makes restarts tolerant
(old ptype-less ve dumps restart with `ptype` zero-filled), promotes M2's temporary `--nonfluid-frac` CLI
hack into the proper per-init option, lands the two restart acceptance gates (mixed-run restart bitwise
equality; old-dump restart), adds the GPU unit test for a rank owning zero fluid particles (CPU
counterpart was M2b's config (b); needs M2b's guards), reviews the five timers, and documents the v1
limitations (`nc <= 1` unsupported; stale SPH outputs for non-fluid particles). **Depends on M3**; item 5
additionally needs M2b's empty-range guards.

## Scope

1. **Tolerant reader** — `IFileReader::datasetNames()` virtual + `H5PartReader` impl; `restoreDataset`
   skips missing datasets (by name), default-fills `ptype → 0`; hard error for genuinely missing physics
   fields; `readField` stays throwing.
2. **Proper per-init option** — replace M2's interim `--nonfluid-frac` wiring; hook beside `runTagging`
   after `initImpl` (`isim_init.hpp:56-62`).
3. **Restart acceptance test (gate)** — mixed run → restart → bitwise snapshot equality incl. `ptype`.
4. **Old ptype-less dump restart (gate)** — restart an existing ve dump, `ptype` column all-zero.
5. **GPU edge cases** — GPU unit test for a rank owning zero fluid particles; timing cleanup (five timers).
6. **Documentation** — `nc <= 1` unsupported in v1; stale SPH outputs for non-fluid particles.

## Context you need

State after M3 (do not re-touch): computeForces steps 1–10, `integrate` steps 1–4, ve-pt `saveFields`
(ptype appended to every dump, pass-2 `gradh` symmetry, pass-3 dropped), M2b empty-range guards, M3's
masked energy update / acceleration zeroing / fluid-only h update, and the `removeUnconvergedParticles`
v1 behaviors (startup error if flag set; throw on unconverged particles, `ve_hydro.hpp:215-218` analog).

Restart-path facts:

- `--init <file>.h5:<step>` — `file:step` is the **file-step index** into the dump's snapshot list, not
  the simulation iteration (`main/src/init/factory.hpp:115-118`). `iteration`/`ttot`/`minDt`/`minDt_m1`
  are restored from file attributes (`sph/include/sph/particles_data.hpp:186-193`).
- `restoreDataset` (`main/src/init/file_init.hpp:43-77`) pairs `fieldNames[i]` ↔ `data()[i]` and restores
  every conserved field found by name (`isConserved` gate `file_init.hpp:52`; `ptype` registered
  conserved in M1). **Today a pre-ptype dump hard-fails in `readField`** (`io/ifile_io_hdf5.cpp:406-411`)
  — item 1 removes that. `IFileReader` (`io/ifile_io.hpp:83-105`) has **no** dataset-listing API;
  `fileutils::datasetNames()` (`io/h5part_wrapper.hpp:99`) exists but needs a raw `h5_file_t`.
- M1 zero-fill anchors (unchanged): `cstone::fill(d.exec, d.ptype.begin(), d.ptype.end(), 0)` after the
  final `d.resize` at every init path (precedent `rung`, `file_init.hpp:258`); `FileInit`/`TDEOrbitInit`
  fill only in the missing-ptype branch (dead until item 1); `FileSplitInit` bypasses `restoreDataset`
  (`file_init.hpp:249-263`) and always fills. `DeviceVector::resize` does **not** initialize device
  memory (`noinit_thrust.cuh:29-55`) — GPU restart correctness depends on the fill.

M2 interim mechanism (what item 2 replaces — the selection rule is kept, only the wiring changes):
`markNonFluidFraction(d, frac)` in `main/src/init/utils.hpp` — **id-keyed selection** via
`generateParticleIDs` (`utils.hpp:129-145`; host-side Allgather + exclusive scan; ids are rank-offset
sequences, so rank-count-independence comes from the rule being a function of `id`, never per-rank
highest-index or per-rank-random). For glass ICs `id` = global SFC order, so `id < f*N` selects a
spatially contiguous corner chunk — use `id % stride` to spread. Introduced behind a temporary
`--nonfluid-frac F` flag parsed in `sphexa.cpp` (`parser.get(...)`, no Config struct; called right after
`simInit->init(...)`, `sphexa.cpp:136`, before the first sync `:157`; update `printHelp`), fresh ICs only.
GPU mechanics (keep): `generateParticleIDs` is host-MPI → D2H ids → host selection → H2D `ptype`
(`memcpyD2HAsync`/`memcpyH2DAsync`) — no new kernel, no new `.cu` (`utils.hpp` is a g++ TU).

Other facts: default output set = `conservedFields()` (`sphexa.cpp:139`); without `-w` no dump is ever
written (`isOutputTriggered`, `sphexa.cpp:106-107,179-183`); `compare_develop.py` cannot drive the gates
(rtol diff, no exit code) — use h5py, and do **not** add `ptype` to its `FIELDS` list.

## Implementation

### Item 1 — Tolerant reader (init/restart)

Goal: pre-ptype dumps (any ve dump) restart with `ptype` zero-filled (all fluid); a dump missing a
*physics* field still errors out loudly. Adopt **option (a)**: a new `IFileReader::datasetNames()`
virtual (`io/ifile_io.hpp:83-105`):

- `H5PartReader`: ~5 lines; the class holds `h5File_` (`io/ifile_io_hdf5.cpp:451`); mirror the
  attribute-listing pattern at `ifile_io_hdf5.cpp:338-348` (the raw `h5_file_t` is already there — why
  `fileutils::datasetNames()` becomes usable).
- `UnimplementedReader`: trivial override (`io/ifile_io.hpp:107`); the ASCII io is writer-only
  (`io/ifile_io_ascii.cpp:45`), never on a read path.
- **Option (b) (catch-the-throw) is REJECTED**: a per-rank catch of `readField`'s throw can diverge
  collectively — one rank defaults while others read real values. Absence decided via `datasetNames()`
  is rank-consistent: every rank runs the same query against the same file and takes the same branch.

In `restoreDataset` (`file_init.hpp:43-77`): fetch `datasetNames()` once; in the per-field restore loop,
when the dataset is absent **by name**: skip the restore and default-fill — for `ptype` fill 0 via
`cstone::fill(d.exec, d.ptype.begin(), d.ptype.end(), 0)` (precedent `rung`, `file_init.hpp:258`); for
any other field this is a **hard error** (clear message naming the missing dataset). `readField` itself
stays **throwing** (`ifile_io_hdf5.cpp:406-411`) — tolerance lives entirely in `restoreDataset`'s name
check, not in the reader's field accessor. This activates the M1 anchors: `FileInit`/`TDEOrbitInit`'s
missing-ptype-branch fill becomes live; `FileSplitInit` keeps its own always-fill.

### Item 2 — Proper per-init option (replaces the M2 mechanism)

- **Hook point**: apply `markNonFluidFraction` inside `ISimInitializer::init`, beside `runTagging`,
  after `initImpl` (`isim_init.hpp:56-62`). The option becomes a per-init setting — an entry in
  `InitSettings` (`std::map<std::string, double>`, `main/src/init/settings.hpp:42`; the settings path is
  the one that already persists selections as file attributes, cf. the tagging strings
  `settings.hpp:44-47`) — so it travels with the IC.
- **What changes vs M2**: hook moves from `sphexa.cpp:136` into the initializer base — uniform across
  fresh ICs and restarts. The M2 "fresh ICs only" restriction is dropped: **id-based selection is
  idempotent on ptype-bearing restarts** (re-marking the same ids yields the same `ptype`); on a restart
  from a ve dump (no `ptype`), item 1 zero-fills first, then the option marks. CLI: keep
  `--nonfluid-frac F` (`parser.get(...)`, `printHelp`) as the fresh-IC setter; it feeds the per-init
  settings entry.
- **The zero-fill must NOT live here**: the hook only marks non-zero — the ptype zero-fill stays at the
  M1 init-path final-resize anchors (a blanket post-`initImpl` fill would clobber restored non-zero
  `ptype` on restart).
- **Keep unchanged**: the id-keyed rule and `id % stride` spreading for glass ICs; rank-count
  independence; GPU mechanics (D2H ids → host selection → H2D `ptype`); the assert that marked `h` stays
  finite/sane (non-fluid `h` feeds `egrav`, `gravity_wrapper.hpp:143`, and `accelerationTimestep`,
  `ts_global.hpp:66,105`). The mixed-IC option **asserts the chosen IC avoids the throw**
  (`ve_hydro.hpp:215-218`) — the `nc <= 1` precondition (below) is made likely by the cut, not guaranteed.

### Item 3 — Restart acceptance test (gate)

Mixed run with the per-init option, then restart from its first snapshot and re-run the same number of
steps with the same dump interval (commands + assertions under *Verification / acceptance*). Why bitwise
equality is a valid claim: dump@k is a **consistent start-of-step-k state** — the ij-loop
(`computeMomentumEnergy`) writes only `du`/`gradP`/`dt`
(`sph/include/sph/hydro_ve/momentum_energy_kern.hpp:224-235`); the v-kick/u-update happen in `integrate`
(`sph/include/sph/positions.hpp:131`). The restart re-executes the courant-factor step from an identical
state — pure, deterministic, bitwise. Restart restores `iteration`/`ttot`/`minDt`/`minDt_m1`
(`particles_data.hpp:186-193`). Requires the **same rank count** for both runs.

### Item 4 — Old ptype-less dump restarts with `ptype` zero-filled (GATE)

Restart an existing ve dump (`--init <ve-dump>.h5:0`) with `--prop ve-pt` — the dump has no `ptype`
dataset, so item 1's tolerant reader must skip it and zero-fill; the restarted run's `ptype` column must
read all-zero and the run must complete. Depends on item 1.

### Item 5 — GPU edge cases + timing cleanup

- **GPU unit test for a rank owning zero fluid particles** (the CPU part is M2b's config (b); **needs
  M2b's guards**): 2-rank GPU test, one rank owning zero fluid particles — empty exchange requests,
  empty group construction. Shape: the cstone-level raw-vector 2-rank harness (`domain_2ranks.cpp` via
  `addCstoneMpiTest`, the M2 pattern — hand-set `ptype` on raw vectors) or a GPU ve-pt mixed config with
  a fraction forcing an empty rank. Relies on M2b: `rhoTimestep` empty-range guard on both paths, placed
  before the `divv.empty()` throws (`sph/include/sph/ts_global.hpp:82,89`); type-generic `minMax(Gpu)`
  early return (`first == last` → `{max, lowest()}`); the `reduceDt` zero-group guard
  `if (grp.numGroups > 0)` (`sph/include/sph/hydro_ve/momentum_energy_gpu.cu:92-94`). Assert: crash-free,
  finite dt on computing ranks, `inf` courant on the empty rank (correct), **no spurious CUDA errors** —
  the old `reduceDt` hazard was a pending non-sticky `cudaErrorInvalidConfiguration` surfacing at the
  next unrelated `cudaGetLastError()` (`checkGpuErrors` inspects only the wrapped call,
  `errorcheck.cuh:15-25`).
- **Timing cleanup**: review the five timers — `fluidLayout`, `partition`, `fluidView`,
  `fluidHaloExchange`, `partitionRestore` (`timer.step(...)` only, no registration). Each must wrap
  exactly its region (steps 2, 4, 5, 3/7, 9 + integrate 3/4); nothing timed ad hoc.

### Documentation (nc ≤ 1; stale SPH outputs)

- Document `nc <= 1` configurations as **unsupported in v1** (see Risks): guard or document; the
  mixed-IC option asserts the chosen IC avoids the throw (`ve_hydro.hpp:215-218`).
- Document that SPH quantities of non-fluid particles (`prho`, `divv`, `xm`, `nc`, `dtCourant`, ...) are
  **stale/undefined in v1** — written on the fluid range only. Garbage columns appear only for
  fluid-only fields requested via `-f`; the **default output set holds only finite values**:
  `temp`/`du_m1`/`alpha` keep their `initFieldsAtRest` values (`utils.hpp:163` / `:151,153`);
  `x/v/m/h/id` are synced/IC. Acceptable for smoke tests; `ptype`-filtering is a follow-up.

## Explicitly out of scope (deferred / follow-ups)

- Any new core mechanics (all landed M0–M3); the step-8 mechanism (output trio permanent allocation); the
  M2 `markNonFluidFraction` *selection rule* itself (item 2 only re-wires where/when it applies).
- `ptype`-filtering of shared observables (`conserved_quantities.hpp:85-101`, `conserved_gpu.cu:92-97`)
  — follow-up; do not filter in v1 (`nbody` has no `ptype`).
- The `removeUnconvergedParticles` removal mechanism (removeKey + key sort would work; the v1 throw is
  conservative and documented — M3 decision).
- Halo-`ax` readers (non-fluid halo accelerations stay garbage forever, never read today), epoch
  exhaustion vs `MPI_TAG_UB`, the pre-existing ve saveFields pass-2/pass-3 bugs (upstream note).

## Verification / acceptance

**Gate 1 — item 3 (restart bitwise equality).** Both runs with the **same rank count**:

```bash
sphexa --prop ve-pt --init evrard --glass ./50c.h5 -n 50 -s 10 -w 5 -o mixed.h5 --nonfluid-frac 0.3
sphexa --prop ve-pt --init mixed.h5:0 -s 10 -w 5 -o rest.h5
```

(`-w` is required — without it no dump is ever written, `sphexa.cpp:106-107,179-183`. Use the proper
per-init option once item 2 lands; the M2 flag drives the same id-keyed rule, so the gate is identical.)
h5py assertions: each snapshot carries an `iteration` attribute; `mixed.h5:0` = file-step index 0
(`factory.hpp:115-118`). Take `rest.h5`'s **last** snapshot, read its `iteration` `k`, locate the
`mixed.h5` snapshot with the same `iteration == k` (identity via the dumped `iteration` attribute, not
positional index), and assert **every** dataset in the two snapshots — **including `ptype`** — is
`np.array_equal` (bitwise); `iteration`/`ttot`/`minDt`/`minDt_m1` are the restored values
(`particles_data.hpp:186-193`).

**Gate 2 — item 4 (old dump restart).** Reuse (or produce) a plain ve dump — no `ptype` dataset:

```bash
sphexa --prop ve --init evrard --glass ./50c.h5 -n 50 -s 10 -w 10 -o ve.h5
sphexa --prop ve-pt --init ve.h5:0 -s 10 -w 10 -o vept-old.h5
```

Assertions: the run completes (pre-M4 it hard-fails in `readField`, `ifile_io_hdf5.cpp:406-411`); h5py
shows `vept-old.h5`'s `ptype` column **all zero**; all other fields are restored from the file
(spot-check one conserved field against `ve.h5`'s file-step-0 snapshot).

**Item 5**: the GPU zero-fluid-rank unit test passes in the GPU build; a GPU config-(b) run is crash-free
with `inf` courant on the empty rank and no spurious CUDA errors. **Item 2**: a fresh mixed IC via the
per-init settings path reproduces the M2 flag's marking (same ids marked); a restart of a mixed dump with
the option set re-marks idempotently (`ptype` unchanged bitwise vs the dump).

## Risks & gotchas

- **`nc <= 1` (at most one fluid neighbor)**: `updateSmoothingLength` reports such particles unconverged
  directly (`sph/include/sph/find_neighbors.hpp:26-35`) → the hard "Neighbor search did not converge"
  throw (`ve_hydro.hpp:215-218`). The fluid halo set was fixed by the full sync — **no amount of h-growth
  can conjure missing fluid neighbors**. The init option makes the ≥ 2-fluid-neighbors precondition
  likely but does **not** guarantee it (low-density outer shells after the cut can drop below the
  threshold); guard or document as unsupported in v1.
- **Do not catch `readField` throws per-rank** (rejected option (b)) — collective divergence; use
  `datasetNames()` so absence is rank-consistent.
- **Do not blanket-fill `ptype` after `initImpl`** — clobbers restored non-zero `ptype` on restart;
  zero-fill stays at the per-path final-resize anchors, and `FileSplitInit` keeps its own always-fill
  (it bypasses `restoreDataset`, `file_init.hpp:249-263`).
- **GPU restarts**: the item-1 zero-fill must run on `d.exec` (GPU fill) — `DeviceVector::resize` does
  not initialize device memory (`noinit_thrust.cuh:29-55`), host zero-init does not apply.
- **Snapshot identity in the h5py check**: match by the `iteration` attribute, never by dataset index;
  bitwise claims hold only for identical rank counts.
- **Stale SPH outputs for non-fluid particles** (documented, not fixed): fluid-only `-f` columns are
  garbage for non-fluid particles; observables stay unfiltered — `eInt` offset exactly 0 (temp branch,
  ve never allocates `u`), `etot` carries a time-varying `egrav` offset (check finiteness, not constancy).
- `compare_develop.py` cannot drive the gates (rtol diff, no exit code); keep `ptype` out of its `FIELDS`
  list (ve runs would show a key-set mismatch).
