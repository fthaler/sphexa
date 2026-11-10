# Milestone M3 — integrate: drift groups, masked energy update, masked acceleration zeroing, fluid-only h update, GPU bitwise gate

M3 completes the `ve-pt` per-step integration. Three work fronts: (1) the
computeForces **step 10** masked zero-fill of non-fluid owned `ax/ay/az`
before the gravity traversal (gravity *accumulates*; phase 7 wrote fluid
accelerations only); (2) the SPH-layer **per-particle energy-update mask** in
`computePositions` so non-fluid particles update `x`/`v` without corrupting
`temp`/`du_m1`, plus `ve-pt`'s overridden **`integrate`** — timestep over the
owned range, type-agnostic KDK drift with warp-sized full-layout drift groups
on GPU (new `driftGroups_` member), fluid-only `h` update on the phase-7
groups under the cached `ordering_`, and the second inverse permutation;
(3) both `removeUnconvergedParticles` v1 behaviors (unconditional runtime
throw; startup error when the config flag is set). Acceptance: all-fluid
bitwise equivalence with `ve` **now on a GPU build too** (validates the
complete GPU path), plus a 30% non-fluid smoke run judged on kinetic energy
and momentum. Depends on **M2b** (SPH empty-range guards) and, for the GPU
gate, the M0+M2 GPU paths.

## Scope

- `main/src/propagator/ve_hydro_pt.hpp`: computeForces step 10; overridden
  `integrate` (steps 1-4); both `removeUnconvergedParticles` behaviors; the
  new `driftGroups_` member (the second `GroupData`).
- SPH layer: `sph/include/sph/positions.hpp`, `positions_gpu.cu`,
  extern decl in `sph_gpu.hpp` — the per-particle energy mask and its
  plumbing through every `computePositions` call site (all compile unchanged
  except `ve-pt`).
- No changes to gravity, timestep, or observable code — M3 only wires and
  masks.

## Context you need

- **Two index spaces** (overview invariant, load-bearing here): fluid-compacted
  (`[fluidFirst:fluidLast)`) vs full SFC (`[0:bufSize)`). Exactly four layout
  transitions per step (computeForces steps 4/9, integrate steps 3/4), all via
  the single helper pair `partition_`/`unpartition_`. `ordering_` stays valid
  between computeForces step 9 and integrate step 3 (no `sync` in between); it
  is a per-step cache, rebuilt in the next computeForces step 4, never reused
  across a `sync`.
- **State at end of computeForces** (M2 output): buffer in full SFC order
  (phase-9 restore applied); `d.treeView` reset to `domain.octreeProperties()`;
  `groups_` holds the phase-7 **fluid** groups — valid only in the compacted
  layout under the same cached `ordering_`; `d.neighborhood` likewise built in
  the compacted layout. `d.neighborhood` and `groups_` live **outside**
  `dataTuple()` (`particles_data.hpp:261-262` vs `:288-298`; `groups_` is a
  propagator member, `ve_hydro.hpp:67`) — never permuted.
- **`ve`'s integrate is the template** (`ve_hydro.hpp:205-220`):
  `computeTimestep(first, last, d)` →
  `computePositions(groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)})`
  → `updateSmoothingLength(groups_.view(), d)` + the unconverged throw
  (`:215-218`) → `timer.step("UpdateQuantities")`. `ve-pt` overrides and
  restructures it per the steps below.
- **Gravity accumulates** into `ax/ay/az` (`traversal_cpu.hpp:220`,
  `traversal_gpu.cu:530`); `ugrav` accumulates identically (null-guarded,
  `traversal_cpu.hpp:219`).
- **GPU group kernels process one warp per group** (`positions_gpu.cu:125-130`)
  — a single whole-range group integrates only the first warp. CPU groups are
  trivial (`sph/include/sph/groups.hpp:63-70`: `numGroups = 1`,
  `groupStart`/`groupEnd` pointing at first/last).
- **Non-fluid `h` stays at its IC value** by design (gravity softening); it
  feeds `egrav` (`gravity_wrapper.hpp:143`, hence `etot`) and
  `accelerationTimestep` (`ts_global.hpp:66,105`). The init option asserts it
  is finite/sane.
- **Mask primitives from M0**: the masked zero-fill (CPU variant in
  `primitives_acc.hpp`, GPU kernel + explicit instantiation in
  `primitives_gpu.cu`) landed in M0 with unit tests; M3 wires it (step 10).
- **`removeUnconvergedParticles` flag** arrives via file attribute
  (`particles_data.hpp:190`). In `ve` the flag only suppresses the throw;
  removal happens via `removeKey` + the next sync's key sort (preserving
  markings, `sfc.hpp:286`).

## Implementation

### A. computePositions energy mask (SPH layer, plan: positions mask)

Per-particle energy-update mask in `computePositions` so non-fluid particles
update `x`/`v` without corrupting `temp`/`du_m1`. The mask gates the **whole**
energy branch on **both** paths (CPU + GPU): `energyUpdate` reads garbage
`du`/`du_m1` and overwrites `temp`/`u`; `du_m1[i] = du[i]` is unconditional —
`positions_gpu.cu:156,163` GPU, `positions.hpp:158,177` CPU. The
`idealGasCv`/`mui` read is conditional and `mui` is never allocated in `ve`.
A range toggle **cannot** work: in the full SFC layout owned fluid and
non-fluid particles are interleaved.

Plumbing:
- Gate `updateTempHost`/`updateIntEnergyHost` on
  `ptype == nullptr || ptype[i] == 0` (signatures gain
  `const uint8_t* ptype`, `positions.hpp:140,163`).
- Host `computePositions`: `const uint8_t* ptype = nullptr` after `rung`
  (`positions.hpp:207-227`).
- Extern declaration `sph_gpu.hpp:84-87`; kernel + launcher + `POS_GPU` macro
  and its **four instantiations** (`positions_gpu.cu:119,168,185-195`).
- Precedent: `rung`'s nullptr pattern (`positions.hpp:208`,
  `positions_gpu.cu:62,132`).
- All other call sites compile unchanged (`std_hydro.hpp:180`,
  `std_hydro_grackle.hpp:227`, `std_disk.hpp:133`, `nbody.hpp:160`,
  `ve_hydro.hpp:213`; `ve_hydro_bdt.hpp:370` passes `rung` positionally as
  trailing arg — appending `ptype` after it keeps it valid).
- **`ve-pt`'s overridden `integrate` passes `rawPtr(d.ptype)`.**
- `driftKernel`'s own energy branch (`positions_gpu.cu:78-88`) is a separate
  unmasked path used only by `bdt` in all-fluid configs — document, don't
  touch.
- Side effect kept by the mask: non-fluid `temp = 0` from `initFieldsAtRest`
  (`utils.hpp:163`) survives → the non-fluid `eInt` observable offset is
  exactly 0 (see verification).

### B. computeForces step 10 — masked acceleration zero-fill before gravity

In `computeForces`, **after the phase-9 restore** (full SFC layout), **before
gravity**: zero `ax/ay/az` for **non-fluid owned particles** — masked fill on
`ptype != 0` over `[startIndex():endIndex())`, using the M0 masked zero-fill
primitive. Run it **unconditionally**, so non-fluid accelerations are
well-defined even with `d.g == 0`.

Why it must be unconditional and masked: the traversal **accumulates** into
the accelerations (`traversal_cpu.hpp:220`, `traversal_gpu.cu:530`); in `ve`
that is safe because `computeMomentumEnergy` overwrites `ax` for the whole
owned range first (`ve_hydro.hpp:185`), but phase 7 wrote fluid accelerations
only, so non-fluid slots hold sync/halo-exchange scratch garbage (`ax`/`ay`/
`az` are dependent, `ve_hydro.hpp:76`; `ax` is also the exchange send buffer).

- The potential `ugrav` accumulates identically (null-guarded,
  `traversal_cpu.hpp:219`) — benign only because `ugrav` is in neither field
  list (always null). If ever resident it must be masked the same way **and**
  join the step-4/9 gather set (same for Ewald's `p`).
- Non-fluid **halo** accelerations remain garbage — never read today (momentum
  kernels exclude `ax`, `momentum_energy_kern.hpp:232-235`; positions/
  timestep/observables are owned-only) — any future halo-`ax` reader breaks
  silently (risks).
- Gravity itself exactly as `ve_hydro.hpp:189-202` (`mHolder_` groups from
  `domain.focusTree()` + `domain.layout()`, `gravity_wrapper.hpp:121-126`) —
  the full-buffer view is **correct for mixed populations**
  (`layout.back() == bufSize`); do **not** apply the `nFluid` gating here.
  Fluid = SPH + gravity, non-fluid = gravity only.

### C. integrate (override) — steps 1-4

Entry state: full SFC layout; `ordering_`/`orderingInv_` valid from this
step's computeForces (no `sync` in between); `groups_` = phase-7 fluid groups
(compacted layout); `d.treeView` = full focus tree.

**Step 1 — timestep.** `computeTimestep(first, last, d)` over the **owned**
range `[startIndex():endIndex())` in full SFC layout — halos excluded (their
`ax/ay/az` are stale and would poison the min-reduce; `minDtCourant`/
`minDtRho` were computed from fluid only in computeForces).
`accelerationTimestep` (`ts_global.hpp:48-71`) limits the gravity-only
particles via `h_i²/|a_i|²` (`:66`; needs sane non-fluid `h` — asserted at
init).

**Step 2 — position update (KDK), type-agnostic, for ALL owned particles in
full SFC layout.** `d.neighborhood` and `groups_` live outside `dataTuple()`
— never permuted; the phase-7 fluid `groups_` are valid only in the compacted
layout under the same cached `ordering_`.
- On **CPU** the trivial owned-range GroupView suffices
  (`sph/include/sph/groups.hpp:63-70`).
- On **GPU** it does **not** — kernels process one warp per group
  (`positions_gpu.cu:125-130`), so a single whole-range group would integrate
  only the first warp. Build **warp-sized groups over the full owned range**
  into the **second** `GroupData` member (`driftGroups_`, new — see
  D-landing below) with the **full** `d.treeView` (reset in phase 9) — i.e.
  the `computeSpatialGroups` path (`groups.hpp:42-55`,
  `cstone::computeGroupSplits` over `d.treeView.leaves`/`layout` +
  `d.traversalStack`) — **preserving the phase-7 `groups_` for step 3**.
  The group-split build reads `h` for all owned particles
  (`groups.hpp:47`) — the non-fluid-`h` sanity assert covers it.
- Do **not** reorder the h-update before the drift — a **layout constraint**:
  the h update runs on the phase-7 fluid groups in the compacted layout
  (step 3), the drift is type-agnostic over the full SFC layout (step 2).
  There is **no data dependency** between them (drift uses `h` only for the
  FBC correction, the h update reads `nc` + the old `h`).
- Energy update (`temp`/`du_m1`) for **fluid only**, masked per particle by
  `ptype == 0` — the step-A mask, `computePositions(..., rawPtr(d.ptype))`.
  Fluid `du` values were restored to full layout in phase 9.

**Step 3 — fluid-only h update in the compacted layout.** Re-apply the cached
`ordering` (via the `partition_` helper) and
`updateSmoothingLength(groups_.view(), d)` over the **phase-7 fluid groups**
— fluid owned particles are only contiguous (`[fluidFirst:fluidLast)`) in the
compacted layout. Consume them here, **still under the same cached
`ordering_`**, without rebuilding anything in between. `removeKey` markings
land in the permuted keys — fine, the restore is a permutation, not a key
sort (canonical rule: overview departures).

**Step 4 — second inverse permutation.** Apply the inverse permutation again —
same fixed gather list + flag-gated trio call (**now including `h`/`keys`
modified in step 3**) — → SFC order for the next `sync`. Precondition: no
resize between syncs (`domain.hpp:104-111`); postcondition: the next sync
resizes to the new count (`domain.hpp:123-124`) — hence `ordering_`/
`orderingInv_` are per-step caches, rebuilt in the next computeForces step 4,
never reused across a `sync`.

**Propagator member landing in M3**: the second `GroupData` `driftGroups_`
(full-layout KDK drift groups; GPU needs warp-sized groups, integrate
step 2) — added alongside `groups_` (`ve_hydro.hpp:67` idiom).

### D. removeUnconvergedParticles — both v1 behaviors

1. **Runtime**: unconverged particles throw **unconditionally** — the
   `ve_hydro.hpp:215-218` analog with ve's
   `&& not d.removeUnconvergedParticles` clause dropped:
   `if (haveUnconvergedParticles) throw std::runtime_error("Neighbor search did not converge\n");`
   (`updateSmoothingLength` reports unconverged for fluid particles with at
   most one fluid neighbor, `find_neighbors.hpp:26-35` — the `nc <= 1` case,
   documented unsupported in v1).
2. **Config**: flag set → **startup error "unsupported in ve-pt v1"**,
   checked at the **first `sync()`** (the flag arrives via file attribute,
   `particles_data.hpp:190`).

In `ve` the flag only suppresses the throw; removal happens via `removeKey` +
the next sync's key sort (preserving `removeKey` markings, `sfc.hpp:286`) —
that mechanism would actually work in `ve-pt`; the v1 throw is conservative
and documented.

## Explicitly out of scope (deferred)

- **computeForces step-8 mechanism / fluid phase** (M2 — assumed landed:
  fluid tree view, fluid groups/pipeline, `fluidHalos_` exchanges, trio
  permanent allocation, saveFields rework).
- **SPH empty-range guards** (M2b — assumed landed: `rhoTimestep` empty-range
  guard on both paths, `minMax(Gpu)` hardening, `reduceDt` zero-group guard;
  zero-fluid-rank config (b)).
- **Tolerant restart reader / proper per-init option** (M4 items 1-2); the
  `--nonfluid-frac` CLI flag is M2's interim mechanism.
- **`ptype`-filtering of shared observables** — follow-up (used by every
  propagator, incl. `nbody` which has no `ptype`):
  `conserved_quantities.hpp:85-101`, `conserved_gpu.cu:92-97`.
- **`driftKernel`'s unmasked energy branch** (`positions_gpu.cu:78-88`) —
  bdt-only, all-fluid configs; document, don't touch.
- **Actual removal of unconverged particles** via `removeKey` + key sort —
  follow-up; v1 throws instead. `nc <= 1` guard (M4).

## Verification / acceptance

### 1. All-fluid bitwise equivalence — now on a GPU build too

The same two-run bitwise gate as M1, run on a **GPU build** (validates the
complete GPU path: fused partition kernel, uint8 halo exchange, device-side
fluid layout, warp-sized drift groups). Two runs, no `-f` (default outputs =
`conservedFields()`, `sphexa.cpp:139`):

```
sphexa --prop ve    --init evrard --glass ./50c.h5 -n 50 -s 10 -w 10 -o ve.h5
sphexa --prop ve-pt --init evrard --glass ./50c.h5 -n 50 -s 10 -w 10 -o vept.h5
```

`-w 10` is required: without it no dump is ever written (`isOutputTriggered`
never fires, `sphexa.cpp:106-107,179-183`). Then a small h5py check: every
ve-named dataset `np.array_equal` (bitwise) between the dumps, and ve-pt's
`ptype` all zero. Presuppositions: equal-mass ICs (evrard glass; with varying
masses ve-pt intentionally diverges — ve's homogeneous `fillMassHalos`
mis-values halo masses) and the default output set (requesting
`divv`/`curlv`/`gradh` takes different IAD/saveFields branches — intentional
divergences). `compare_develop.py` cannot drive the gate (rtol diff, no exit
code) — use the h5py check.

**Amended 2026-11-10 (user decision, option 1):** the GPU build is not bitwise-reproducible
run-to-run (`ve` vs `ve`, same binary, back-to-back `-n 50 -s 10 -w 1` runs: v/x differ from
iteration 1 — v maxabs ~6e-11, x ~4e-15, temp/du_m1/alpha/h/id/m bitwise-identical at step 0;
energy fields join from step 1 via the AV velocity dependence; x snowballs to ~1.7e-13 by step
8; h bitwise-equal throughout). The seed is in the ax path with the momentum kernel exonerated
by clean step-0 du/temp — GPU gravity traversal the prime suspect; pre-existing (gravity
untouched by M3). The gate criterion is therefore re-specified as a **tolerance gate**: same
two-run command and structure checks (same step groups; ve-pt = ve ∪ {ptype}; `ptype` all
zero), with integer datasets (`id`) bitwise-exact and floating-point datasets
maxabs ≤ **1e-6** — `check_gate.py --tolerance 1e-6`. Calibration (measured noise, step 10,
-n 50): x/y/z ~1.7e-13, v ~3.5e-10, alpha 4.5e-8, du_m1 1.9e-9, temp 7e-20, h/id/m bitwise 0 —
≥20× margin; valid for the -n 50 10-step evrard configuration, revisit for longer runs. The
CPU bitwise gate remains exact and is the primary correctness check.

### 2. Mixed smoke run — 30% non-fluid

```
sphexa --prop ve-pt --nonfluid-frac 0.3 --init evrard --glass ./50c.h5 -n 50 -s 10 -w 10 -o mixed.h5
```

(`--nonfluid-frac` is M2's interim mechanism, fresh ICs only.) Judge
**kinetic energy and linear/angular momentum drift** — unpolluted: computed
from `x`/`v`/`m` only. Run a 2-rank variant too (M2 config (a) shape: mixed
low-fraction, crash-free, conserved momentum — not bitwise, paths diverge by
design). Note: mixed full-sphexa configs must pass `-f` including
`divv,curlv,gradh` at least once (an output set without them never triggers
the step-8 permanent-allocation path) if exercising that branch.

### Observables

| Observable | Expected | Why |
|---|---|---|
| `eInt`, non-fluid offset | **exactly 0** | `eInt` takes the **`temp` branch** — ve never allocates `u` (`conserved_quantities.hpp:92-101` CPU + `conserved_gpu.cu:93-95` GPU) — via non-fluid `temp = 0` (`utils.hpp:163`), kept by the M3 energy mask |
| `etot` | **`egrav` offset, time-varying** | non-fluid particles move under gravity; IC `h` → `gravity_wrapper.hpp:143` — check **finiteness, not constancy** |
| `Ekin`, linear/angular momentum | judge the drift | computed from `x`/`v`/`m` only (unpolluted) |
| non-fluid `temp`/`du_m1`/`alpha` | keep `initFieldsAtRest` values (`utils.hpp:163` / `:151,153`) | the energy mask; the stale-output-finiteness argument |
| non-fluid `h` | finite/sane | asserted at init (M2); feeds `egrav` (`gravity_wrapper.hpp:143`) and `accelerationTimestep` (`ts_global.hpp:66,105`) |

Do **not** filter the shared observables by `ptype` in v1 (used by every
propagator, incl. `nbody` which has no `ptype`; follow-up):
`conserved_quantities.hpp:85-101`, `conserved_gpu.cu:92-97`.

## Risks & gotchas

- **Two index spaces in one propagator** — the largest bug source; keep the
  single helper pair `partition_`/`unpartition_` doing *all* layout
  transitions and never hand a fluid index to a full-layout routine (esp.
  `rhoTimestep`, gravity, output) or vice versa. Exactly four transitions
  (steps 4/9, integrate 3/4), all via the helper pair; `ordering_` stays
  valid between step 9 and integrate step 3 (no `sync` between).
- **Stale neighbor lists across layout switches** — any kernel using
  `d.neighborhood` or `groups_` must only run in the layout it was built in
  (drives the `driftGroups_` split; integrate step 2). A single whole-range
  GPU group integrates only the first warp (`positions_gpu.cu:125-130`) —
  silent partial integration, no error.
- **`ordering_` validity window** — per-step cache only: never reuse across a
  `sync` (the sync resizes: precondition no resize between syncs,
  `domain.hpp:104-111`; postcondition resizes to the new count,
  `:123-124`).
- **removeUnconvergedParticles v1 semantics** — startup error if the flag is
  set; unconditional throw on unconverged particles. In `ve` the flag only
  suppresses the throw; removal via `removeKey` + the next sync's key sort
  (`sfc.hpp:286`) would actually work in `ve-pt`; the v1 throw is
  conservative and documented.
- **Non-fluid `h`** stays at its IC value by design (gravity softening);
  feeds `egrav` (`gravity_wrapper.hpp:143`, hence `etot`) and
  `accelerationTimestep` (`ts_global.hpp:66,105`) — the init option asserts
  it is finite/sane.
- **SPH quantities of non-fluid particles** (`prho`, `divv`, `xm`, `nc`,
  `dtCourant`, ...) are stale/undefined in v1 — written on the fluid range
  only; garbage columns appear only for fluid-only fields requested via
  `-f` (the **default output set holds only finite values**: `temp`/`du_m1`/
  `alpha` keep their `initFieldsAtRest` values, `utils.hpp:163` / `:151,153`;
  `x/v/m/h/id` are synced/IC). Acceptable for smoke tests, document it;
  `ptype`-filtering is a follow-up.
- **Non-fluid halo accelerations remain garbage forever** — never read today
  (step 10); any future halo-`ax` reader breaks silently.
- **Do not apply `nFluid` gating to gravity** — the full-buffer view
  (`mHolder_` groups from `domain.focusTree()` + `domain.layout()`) is
  correct for mixed populations (`layout.back() == bufSize`).
- **Zero-fluid ranks**: the group machinery is naturally safe — GPU group
  data is always resized to ≥ 1 element (`groups_gpu.cu:93,106`) so
  `numGroups = 0` and group-indexed kernels no-op (zero-block guards,
  `positions_gpu.cu:180`); the CPU path sets `numGroups = 1` with a single
  empty group (`groups.hpp:65-70`). The `driftGroups_` build must follow the
  same pattern; `reduceDt`/`rhoTimestep` are M2b's guards.
- **MPI hygiene**: do not add a fluid data exchange inside `integrate` —
  the step-3 fencing windows (overview invariant) are emergent from the two
  `Alltoall`s plus complete-on-return exchanges; an exchange outside them
  makes colliding tag sequences corrupt.
