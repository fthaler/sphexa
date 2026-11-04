# Milestone M2 — fluid SPH phase (`fluidHalos_`, fluid tree view, `saveFields` rework, `--nonfluid-frac`)

Split the ve-pt SPH path onto the **fluid-only index space**: exchange halo
`ptype`+`m` with the full pattern right after `sync`, compute `fluidLayout`
(fluid-only leaf counts + exclusive scan), wire a second `cstone::Halos`
instance (`fluidHalos_`) for the mid-phase exchanges, run the SPH pipeline on
`[fluidFirst:fluidLast)` under a fluid tree view (focus tree + swapped layout
pointer), permanently allocate the `divv/curlv/gradh` trio behind a trigger
flag, and restore via the cached inverse permutation (now incl. the trio).
M2 also owns the **`saveFields` rework** (pass-2 `gradh` symmetry, pass-3
drop), the **interim mixed-IC mechanism** (`--nonfluid-frac`) so mixed runs
do not wait on M4, and the **2-rank verification** (cstone gtests +
full-sphexa 2-rank configs + peer-consistency assertions). Depends on **M1**
(registered `ve-pt`, ptype end-to-end, partition→restore skeleton with
identity net effect, ve's `fillMassHalos` still in place, timers,
`compare_develop.py` literal) and consumes the **M0** primitives
(`stablePartition`, fluid-layout helper, masked zero-fill,
`GATHER_RANGES(uint8_t, unsigned)`, `gatherArrays` hardening) — the only
cstone API change in M2 is the `Halos` peer getters. Read `overview.md` first
for shared background; load-bearing invariants are restated verbatim below.

## Scope

- `computeForces` steps 1–9 as they land in M2 (step 10 — masked zero-fill of
  non-fluid owned `ax/ay/az` before gravity — is **M3**; gravity itself stays
  exactly as `ve_hydro.hpp:189-202`).
- `saveFields` rework: pass-2 `gradh` symmetry + pass-3 drop.
- Interim mixed-IC mechanism: `--nonfluid-frac F` + `markNonFluidFraction`.
- Propagator members: `fluidHalos_` (lazy), `fluidLayoutHost_`/`fluidLayoutAcc_`,
  dedicated fluid-halo scratch buffers, partition scratch, trigger flag.
- `cstone::Halos` const getters `outgoing()`/`incoming()` + peer-consistency
  assertions; 2-rank cstone gtests; full-sphexa 2-rank configs.

## Context you need

- **Buffer geometry** (post-sync): `[0:start)` prefix halos, `[start:end)`
  owned, `[end:size)` suffix halos (`domain.hpp:499-500`,
  `buffer_description.hpp:29-45`), SFC-sorted over the whole buffer
  (`domain.hpp:141-144`). The compaction must be **stable**: the fluid subset
  stays SFC-sorted and in leaf order with the same three-segment geometry;
  `fluidFirst = fluidLayout[startCell]`, `fluidLast = fluidLayout[endCell]`.
- **Whole-leaf halo delivery premise**: the full sync delivers complete halos
  per flagged leaf (`exchange_keys.hpp:42-43,80-88`), so per-leaf fluid counts
  are peer-consistent and `fluidLayout` needs **no communication**; zero-fluid
  leaves are never requested (`extractMarkedElements` emits only non-empty
  ranges, `layout.hpp:112-141`).
- **Two index spaces** (largest bug source): fluid-compacted
  (`[fluidFirst:fluidLast)`) vs full SFC (`[0:bufSize)`). Exactly four layout
  transitions per step (steps 4/9 here; integrate 3/4 in M3), **all** via one
  helper pair `partition_/unpartition_` doing *all* layout transitions; never
  hand a fluid index to a full-layout routine (esp. `rhoTimestep`, gravity,
  output) or vice versa. `ordering_` stays valid between step 9 and integrate
  step 3 (no `sync` between); it is a per-step cache, rebuilt in the next
  computeForces step 4, never reused across a `sync`.
- **Never restore by re-sorting on `keys`**: unconverged particles get
  `keys[i] = removeKey` (`find_neighbors.hpp:32`, `find_neighbors_gpu.cu:34`);
  restore is always the cached inverse permutation applied as a gather.
- **Run-constant flags**: the step-8 trigger flag and the two-list gather
  selection are derived from `d.outputFieldIndices` (set once,
  `sphexa.cpp:139`, before the loop `:167`). Test the flag boolean, not
  membership lookups; a mid-run true→false flip throws at the leg-1 `acquire`.
- **Field-state mechanics**: conserved fields are never released/acquired
  (`field_states.hpp:39,123-132`); `acquireOne` throws on non-`unused`
  (`:145-149`) and *swaps* buffers with its donor (`:165-178`); `releaseOne`
  throws on non-`dependent` (`:123-132`); `d.resize` (re)allocates exactly
  non-`unused` fields (`particles_data.hpp:330-352`; `isAllocated`
  `field_states.hpp:61-64`).
- **Presuppositions from M0/M1**: `gatherArrays` throws (release-active) on
  `array.size() < ordering.size()` guarding both `reorderArray` branches
  (`layout.hpp:237-258`); `stablePartition` writes both permutations in one
  fused pass with `nFluid` as parameter; `ve-pt` registered, `ptype` in
  `ConservedFields` + `dataTuple()`/`fieldNames`, partition→restore of the
  fixed gather list with identity net effect, ve's `fillMassHalos` still
  called (it must break **exactly here**, step 6), `ordering_`/`orderingInv_`
  and the keys-sized partition scratch already exist.
- ve baseline being mirrored: `ve_hydro.hpp:113-202` (computeForces),
  `:222-287` (saveFields passes).

## Implementation

Timers (add `timer.step(...)`, no registration): `fluidLayout`, `partition`,
`fluidView`, `fluidHaloExchange`, `partitionRestore`.

### computeForces

#### Step 1 — sync, `(ptype, m)` halo exchange, marking site, resize

Sync on **all** particles, branching exactly as `ve_hydro.hpp:113-122` —
`domain.syncGrav(keys, x, y, z, h, m, conserved..., dependent...)` when
`d.g != 0`, plain `domain.sync(...)` with `m` folded into the conserved tuple
otherwise (different focus-tree convergence criteria; required for the M1/M2
bitwise equivalence at `g == 0` too) — already in place from M1. M2 adds,
on the focus-tree pattern set by the main sync:

```cpp
domain.exchangeHalos(std::tie(d.ptype, d.m), ax, keys);
```

— halo `ptype` (static, once per sync) and halo `m` (GPU path consumes M0's
`GATHER_RANGES(uint8_t, unsigned)`, `halos/gather_halos_gpu.cu:60-74`).
**Halo `m` rationale**: sync delivers halos only for `x, y, z, h`
(`domain.hpp:481-489`, contract `:132-139`), and `computeXMass` (neighbor
masses) + gravity's upsweep — consuming `m` over the whole buffer incl. halos
(`gravity_wrapper.hpp:128-133` GPU, `:62-70` CPU; as `nbody.hpp:130`) — need
valid halo `m`. Replaces ve's homogeneous `fillMassHalos`; non-fluid halo
slots must **not** be left untouched.

**Single marking site** (step 8): right after the sync, mark the IAD trio
`divv`/`curlv`/`gradh` dependent iff the step-8 trigger flag is set — marking
must precede `d.resize`, else the fields stay zero-length through step 4's
`gatherArrays` (OOB on GPU). Then `d.resize(domain.nParticlesWithHalos())` at
the `ve_hydro.hpp:135` position: `sync` resizes only the arrays passed to it
(`domain.hpp:123,466,502`); `ptype` is sized by the conserved tuple, the
dependent outputs *solely* by this resize (step 4 gathers all allocated
fields); grown non-conserved fields hold stale contents — safe per the
step-8 stale-content invariant below.

**Step-1 ordering invariant** (assert it; plan wording): *sync → mark → resize
→ ptype/m exchange → fluid counts (step 2)* — a stale halo `ptype` yields a
wrong `fluidLayout`, surfacing as an MPI size-mismatch hang in the fluid
exchanges, not a clean error. **The ordering is asserted, not defensive**: it
is what makes the peer-consistent `fluidLayout` (step 2/3) possible at all.

**Keys-as-receive-scratch safety**: `ax`/`keys` as the exchange's scratch pair
is safe — the GPU packed receive reaches into the owned range, but nothing
reads keys before the next `sync` recomputes them (`assignment.hpp:109-113`,
`domain.hpp:485-488`, `neighborhood.hpp:50-51`; CPU ignores scratches,
`halos.hpp:89-96`) — full hazard analysis in the step-3 scratch spec.

Keep `d.treeView = domain.octreeProperties()` for now (it is swapped in step 5
and reset in step 9).

#### Step 2 — fluid leaf counts + `fluidLayout`

Segmented count of `ptype == 0` over the `domain.layout()` segments +
exclusive scan. **No communication** (whole-leaf halo delivery premise). No
public host layout accessor (`domain.hpp:359` device span; host `layout_`
private, `:657-658`) and no host `ptype` copy → on GPU the count runs
**device-side** (segmented-count kernel + device `exclusiveScan(unsigned)`,
`primitives_gpu.h:164-168`, instantiated `primitives_gpu.cu:378`; no
count-by-predicate primitive exists) into `fluidLayoutAcc_`, then D2H into
`fluidLayoutHost_` (`memcpyD2HAsync` + `syncGpu`, staging
`octree_focus_mpi.hpp:562-571`; the stream is already drained by the halo
exchange's `cudaStreamSynchronize`, `exchange_halos_gpu.cuh:110`); CPU
computes on the host. Reallocate **both** to `numLeafNodes + 1` every step
(`domain.hpp:205` pattern).

Not alternatives (do not substitute): `computeNodeLayout` (presence flags +
total counts, no predicate, `layout.hpp:153-185`); `cstone::computeNodeCounts`
(sorted keys + tree, `csarray.hpp:182`).

`nFluid = fluidLayout.back()` feeds the fused partition (step 4) and
`totalBodies` (step 5). Timer: `fluidLayout`.

#### Step 3 — `fluidHalos_` request exchange, invariants (a)–(d), scratch spec

```cpp
fluidHalos_.exchangeRequests(focusTree.treeLeaves(), focusTree.assignment(),
                             fluidLayoutHost_);
```

on a **second `cstone::Halos` instance**. Internals (`halos/halos.hpp:64-80`,
`halos/halo_peers.hpp`, `exchange_keys.hpp`) are pure functions of the layout,
so both ends agree automatically — provided per-leaf fluid counts are
peer-consistent (step 1's halo-`ptype` exchange guarantees it; asserted by
M2's peer-consistency check, Verification section).

**Memory spaces**: `exchangeRequests` consumes a *host-readable* span (Domain
passes its *host* `layout_`, `domain.hpp:211,278`) → pass `fluidLayoutHost_`;
the fluid `OctreeNsView` uses the device pointer (step 5). The GPU overload
static_asserts `IsDeviceVector` (`halos.hpp:104`).

**Tag-collision analysis.** No separate tag base for the request exchange:
`exchangeRequestKeys` hardcodes `P2pTags::haloRequestKeys`
(`exchange_keys.hpp:58`), whose comment (`:94-96`) demands a collective
between consecutive request exchanges — provided by the `MPI_Alltoall` in
`exchangePeers` (`halo_peers.hpp:44-47`) inside `exchangeRequests`. The halo
*data* exchange tags **do** collide (both instances derive
`P2pTags::haloExchange + epoch` from independent `haloEpoch_` counters,
`halos.hpp:123-127`, `exchange_halos.hpp:32`) — safety rests on
complete-on-return exchanges (trailing `MPI_Waitall`,
`exchange_halos.hpp:71-85` / `exchange_halos_gpu.cuh:96-112`) plus the
`Halos` re-entry contract (barrier/collective or epoch increment between
calls, `exchange_halos.hpp:87-89` / `exchange_halos_gpu.cuh:118-119`), not
the Alltoall ordering.

**Pin the ordering as an explicit invariant list (comment/assert in the
propagator) — this list is load-bearing, not defensive**: the fencing is
emergent from the two Alltoalls plus complete-on-return exchanges, and any
violation hangs or corrupts (e.g. a fluid exchange issued in `integrate` would
hang its peer):

- (a) no main-instance `exchangeHalos`/`exchangeRequests` from the step-3
  Alltoall until the next sync's `exchangeRequests` Alltoall (covers the fluid
  exchanges and the dead zone after them);
- (b) no fluid `exchangeHalos` after the step-7 exchanges until the next
  `sync`'s `exchangeRequests` (`domain.hpp:211-214`);
- (c) `fluidHalos_.exchangeRequests` before its first `exchangeHalos`, same as
  the main instance;
- (d) `fluidHalos_` gets **its own** GPU scratch buffers — spec below;
  `ax`/`keys` never aliased.

**Scratch spec.** The hazard is *not* removeKey markings (during the step-7
exchanges `keys` carries compacted SFC keys; removeKey is written only in
integrate step 3, `find_neighbors.hpp:30-33`) and *not* a size check
(`haloExchangeGpu` restores pre-call scratch sizes,
`exchange_halos_gpu.cuh:115-116`; `checkSizesEqual` inspects only data arrays,
`domain.hpp:410-416`). The real hazard: the GPU receive scratch is filled with
packed field data over `[0, maxReceiveBytes)` (`exchange_halos_gpu.cuh:91-99`)
— for multi-field exchanges this reaches past `fluidFirst` into the compacted
**owned** range (and, workload-dependently, past `bufSize`), so aliasing
`keys` would corrupt owned fluid keys that the phase-9 restore then permutes.
Spec: **two dedicated `AccVector<uint64_t>` members (send + receive),
grow-only inside the call, never aliased with `keys`/`ax`**; the CPU overload
ignores scratches (`halos.hpp:90-96`) — GPU-only. The fluid exchanges bypass
domain's `checkSizesEqual` entirely (direct `Halos` calls) — **pre-size assert
(`arrays == bufSize`) at the call sites** (step 7).

#### Step 4 — partition + gather set (trio addition)

Build `ordering_` + `orderingInv_` in **one fused pass** (M0 `stablePartition`,
`nFluid` from step 2) and apply `ordering_` to exactly the gather set:

- **Fixed compile-time list** (M1, unchanged), flattened with `std::tuple_cat`
  into the single flat tuple `gatherArrays` takes (`domain/layout.hpp:234` — a
  nested FieldList tuple would not compile), in the same field order as the
  sync gather (`ve_hydro.hpp:115-121`, keys first):
  `tuple_cat(tie(keys, x, y, z, h), tie(m), get<ConservedFields>(d),
  get<DependentFields>(d))` — `ConservedFields` includes `ptype` (ptype itself
  is gathered); `keys` is `setDependent` (`ve_hydro.hpp:105`) but not in
  `DependentFields` — listed explicitly.
- **Trio call (M2)**: `divv`/`curlv`/`gradh` are in neither ve list
  (`ve_hydro.hpp:73-84`) yet must be permuted when requested — a **second,
  flag-gated `gatherArrays` call** over `std::tie(divv, curlv, gradh)`,
  executed iff the step-8 trigger flag is set (then all three are permanently
  allocated, `size == bufSize`). The flag is **per-run constant**
  (`setOutputFields` runs once, `sphexa.cpp:139`, before the loop `:167`), so
  the **same two-list selection** is used at all four applies (steps 4/9,
  integrate 3/4 in M3). Release-safe per-field assert `size == bufSize` on
  every gathered field (a fold over the tuple; `gatherArrays`' own asserts,
  `layout.hpp:244,252`, compile out in release) — plus M0's release-active
  `throw` on `array.size() < ordering.size()` guarding both `reorderArray`
  branches (`layout.hpp:237-258`), which catches the null-source gather of a
  size-0 field.
- **Scratch**: one keys-sized (`uint64`, `bufSize`) scratch serves the whole
  set (the reuse branch accepts any scratch with element size ≥ the array's
  via `reinterpret_cast`, `layout.hpp:250-256`, including `double` fields,
  `sph/types.hpp:42`); **distinct from the `fluidHalos_` scratch** (step 3(d));
  never resized after step 4; **never cache raw pointers across an apply**
  (the exact-type swap branch for `keys`/`id` toggles buffer identity). The
  GPU flags buffer for the partition scan is carved from this scratch before
  the gathers. Timer: `partition`.

#### Step 5 — fluid tree view + `totalBodies` plumbing

Assemble the **fluid tree view** by reusing the focus tree: copy
`domain.octreeProperties()`, replace the `layout` pointer with the device
`fluidLayout` pointer, assign to `d.treeView` (`OctreeNsView` holds tree
pointers + layout pointer + centers/sizes, `tree/octree.hpp:259-283`; the
layout-swap alone is self-consistent — centers/sizes are key-derived node-cell
bboxes, identical in both views, a safe conservative over-approximation for
overlap bounds). **Do not** build a fresh tree with `computeOctree` on
`keys[0:nFluid]`: the layout is indexed by focus-tree leaves and must match
its tree.

Grouping reads only `d.treeView.leaves/numLeafNodes/layout`
(`sph/include/sph/groups.hpp:46-48`) → the fluid view suffices; CPU
`computeGroups` ignores the tree (`sph/include/sph/groups.hpp:63-70`). The
swap is also load-bearing for `updateSmoothingLengthIterative`: under the
fluid layout it counts **fluid-only** neighbors (the desired semantics —
non-fluid particles stop being j-candidates for the h update), so the view
must be in place **before step 7**.

Halo coverage: the full sync delivers complete halos per leaf (premise above),
so per-leaf fluid counts match the owner's and every fluid particle
(owned + halo) falls inside `fluidLayout`'s leaf ranges; zero-fluid leaves are
never requested (`layout.hpp:112-141`).

**`totalBodies = nFluid` (not `d.size()`)**: the neighborhood build must
receive `nFluid` (= `fluidLayout.back()`), **not** `d.size()` (= `bufSize`):
arrays keep full length through the partition, so with `d.size()` non-fluid
particles load as *real* neighbors — in the traversal j-loops
(`loop.cuh:384,399-404,465,481`; with `totalBodies == layout.back()` tail
j-lanes become inert NaN dummies, `common.hpp:60-65`; i-side already gated by
group bounds) **and in the build phase**, which sizes/bounds j-clusters from
`totalBodies` (`build.cuh:111-115,144,164,349`) — the nFluid fix covers both
phases; `d.size()` would load non-fluid particles as real neighbors in
traversal **and** build. Plumbing: add `LocalIndex totalBodies = 0` (meaning
`d.size()`) to `neighborhood.hpp:50-51`, `neighborhood_gpu.hpp:82-83`,
`find_neighbors.hpp:12-18`, `find_neighbors_gpu.hpp:11-12` /
`find_neighbors_gpu.cu:7-13`; ve-pt passes `nFluid`. The CPU j-candidate path
needs no change (`CpuFullNbList` uses the count only for `hExt` sizing).
Timer: `fluidView`.

#### Step 6 — drop `fillMassHalos` **exactly here**

**No `fillMassHalos` call from M2 on**: halo `m` was exchanged in step 1 and
travels through the partition unchanged. Do **not** call it at all — it fills
`[0:first)` **and** `[last:buffer end)` (`particles_data.hpp:415-429`),
clobbering the non-fluid tail `[nFluid:bufSize)`. Migration: keep ve's call at
M1; drop it exactly when the step-1 `(ptype, m)` exchange lands (M2), or the
M1 bitwise gate breaks.

#### Step 7 — SPH pipeline on the fluid range

Mirror `ve_hydro.hpp:141-187` with `first = fluidFirst`, `last = fluidLast`
and the fluid `OctreeNsView`: `computeGroups`,
`updateSmoothingLengthIterative`, `findNeighborsSfc`, `computeXMass`,
`computeVe`, `computeIadDivvCurlvGradh`, `rhoTimestep(fluidFirst, fluidLast,
d)` (signature `(first, last, d)`; empty-range guard is M2b — ranges are
non-empty here whenever the rank owns/request fluid), `computeEOS`,
`computeAVswitches`, `computeMomentumEnergy`.

All mid-phase `domain.exchangeHalos(...)` calls (xm; vx,vy,vz,kx;
c_ij,divv,c; prho,alpha[,dV]) become
`fluidHalos_.exchangeHalos(domain.exec(), arrays, s1, s2)` with the
**dedicated scratch buffers from step 3(d)** — not ve's `(ax, keys)` pattern
(`ve_hydro.hpp:151,156,168,177,180`); `domain.exchangeHalos` is **unusable** —
hard-wired to the domain's private `halos_` (`domain.hpp:336-337`);
`domain.exec()` is the execution-context pattern (`ve_hydro.hpp:139`).
(`updateSmoothingLengthIterative` performs no halo exchanges,
`find_neighbors.hpp:72-78.`) Assert `arrays == bufSize` at every call site
(step 3: `checkSizesEqual` is bypassed). Timer: `fluidHaloExchange`.

The `dtCourant` reduction stays inside the fluid group range → cannot be
poisoned by non-fluid particles.

**Field rotation** (`ve_hydro.hpp:159-160` release `ay,az` / acquire
`divv,gradh`; `:183-184` reverse) with one change: **both legs are state-aware
and coupled on the step-8 trigger flag** — when the trio is `dependent`
(requested outputs), skip **both** pairs (releasing `ay,az` without consuming
them would strand them `released`; on each leg the release must be skipped
whenever the paired acquire is skipped, since `acquireOne` throws on
non-`unused` (`field_states.hpp:145-149`) and `releaseOne` on non-`dependent`
(`:123-132`)); otherwise run both pairs as `ve` does. Nothing reads `ay`/`az`
between the rotation legs (pure buffer donation; the IAD input tuple has no
`ay`/`az`). The flag is a **run-invariant** (same per-run constancy as
step 4); a mid-run true→false flip would throw at the leg-1 `acquire` (the
trio is `dependent`, not `unused`).

#### Step 8 — output trio trigger + permanent allocation

Output fields `divv`, `curlv`, `gradh` need **no post-momentum recompute** —
provided they are permanently allocated. **Trigger (all-three-or-nothing)**:
`gradh` is additionally consumed by the saveFields EOS recovery pass
(`hydro_ve/eos.hpp:75,88,117,145`), and the marking + permanent allocation are
trio-wide (requesting only `divv` still incurs the curlv-branch FLOPs and
gather traffic). On the first `computeForces` of a run requesting **any** of
them: set the trigger flag (a propagator member derived from
`d.outputFieldIndices` at the top of computeForces) and mark all three
`dependent` via `FieldStates::setDependent` (`field_states.hpp:55-59`;
`setState` `:183-194`) at the **step-1 marking site** (right after the sync,
immediately before the `d.resize` at the `ve_hydro.hpp:135` position).
`activateFields` cannot do it (`setOutputFields` runs later, `sphexa.cpp:134`
vs `:139`); `resize` (re)allocates exactly non-`unused` fields
(`particles_data.hpp:330-352`; `isAllocated` `field_states.hpp:61-64`) — and
**never release them again**. The trio stays **out** of ve's
`DependentFields` sync list (`ve_hydro.hpp:76`) — intentional (their one halo
exchange, divv in step 7, uses the fluid pattern).

**Stale-content invariant**: `d.resize` leaves grown non-conserved fields
holding stale contents (contents survive while size ≤ capacity,
`reallocate.hpp:21-32`; `particles_data.hpp:334-343`) — safe because nothing
**consumes** the trio between the resize and the phase-7 IAD (step 4's
flag-gated gather reads it value-agnostically); first readers are
`rhoTimestep` (`ts_global.hpp:82-94`), `computeEOS` (`eos.hpp:60,102,131`) and
the divv halo exchange (`ve_hydro.hpp:168`), all post-IAD; holds on the
rotation-skip path too. Do **not** port `ve_hydro_bdt`'s `groupDivvTimestep`
(`ts_rungs.hpp:53`, called post-IAD only, `ve_hydro_bdt.hpp:254`). Assert once
that the curlv output-branch gating (`curlv.size() == x.size()`, read at
`iad_divv_curlv_gradh.hpp:49` / `_gpu.cu:48`) stays true through the
compaction.

Then (a) the phase-7 IAD already writes all three with the fresh fluid
neighborhood (kernel selects the `curlv` branch whenever the buffer exists,
`iad_divv_curlv_kern.hpp:110-139`) — final when the phase-9 restore permutes
them; no donor dance, no second IAD pass; (b) the step-7 rotation takes its
skip path every step; (c) saveFields pass 2 finds `gradh` already `dependent`.
Precedent: `ve_hydro_bdt.hpp:100`. This **replaces the rejected donor scheme**:
`dV` fields exist only under `avClean` (`ve_hydro.hpp:80-84`; default
**false**, `sphexa.cpp:104`), `alpha` is conserved and can never be a donor
(`ve_hydro.hpp:73`), and every `acquireOne` *swaps* buffers with its donor
(`field_states.hpp:165-178`) — a release/acquire cycle is not repeatable.

#### Step 9 — inverse permutation restore

Apply the cached **inverse permutation** — same fixed list plus the same
flag-gated trio call — → buffer back in full SFC order; reset
`d.treeView = domain.octreeProperties()`. (`orderingInv_.size() == bufSize` is
guaranteed by the fused kernel writing one entry per `i`; assert it in release
too, the debug asserts are compile-outs.) Timer: `partitionRestore`.

### `saveFields` rework (ve-pt's own version; do not inherit ve's pass structure blindly)

- **Pass 1** writes `outputFieldIndices ∩ isAllocated` as in ve
  (`ve_hydro.hpp:227-249`) **plus `"ptype"` appended if absent** (M1) — at the
  end of the local `indicesDone`/`namesDone` copies (`ve_hydro.hpp:226-228`);
  column mechanics unchanged (`AsciiWriter` sort places ptype last,
  `ifile_io_ascii.cpp:82-84,116`; H5 per-named-dataset).
- **EOS recovery pass** (`computeEOS(first,last)`) is element-wise → safe in
  full SFC layout after restore. Its `gradh` input
  (`hydro_ve/eos.hpp:75,88,117,145`) is valid whenever `gradh` is a requested
  output (permanently allocated, phase-7-written, phase-9-restored); otherwise
  the acquired buffer holds stale donor content — harmless: pass 1 wrote
  `prho`/`c` from valid values and the pass-2 `rho`/`p` outputs are
  `gradh`-independent (`rho = kx*m/xm`, `hydro_ve/eos.hpp:73,86,115,143`) —
  exactly as in `ve`.
- **Pass 3 is DROPPED**: the requested outputs are permanently allocated
  (step 8), written by the phase-7 IAD, attached by the phase-9 restore —
  pass 1 writes the columns directly. (ve's pass 3 writes through gradh's
  zero-length home buffer whenever divv/curlv are requested — OOB; see
  upstream note below. Do not re-add pass 3.)
- **Pass 2 must be symmetric in `gradh`**: when `gradh` is already `dependent`
  (requested output, step 8) it must not be re-acquired (`acquireOne` throws
  on non-`unused`, `field_states.hpp:145-149`) — release only `c11, c12` as
  donors, acquire only `rho, p`, close with
  `release(rho, p); acquire(c11, c12);`. The **c13 starvation argument**:
  releasing three donors for two acquires strands `c13` `released` and the
  closing `acquire(c11,c12,c13)` throws on it; a literal `release(rho,p)` in
  the other branch starves the closing acquire of its third donor ("No
  suitable field available", `field_states.hpp:179-180`). When `gradh` is not
  requested (trigger flag false), run **ve's dance unchanged** — release
  `c11, c12, c13`; acquire `rho, p, gradh`; output; release `rho, p, gradh`;
  acquire `c11, c12, c13` (closing pair `ve_hydro.hpp:260-261`) — restoring
  the pool exactly. (In that branch ve-pt drops ve's partial cij recovery:
  c11/c12 keep rho/p values in `[first:last]`; harmless — nothing consumes
  cij before the next IAD.) Upstream precedent for the skip shape:
  `ve_hydro_bdt.hpp:431-442`.
- **Output writes `[first:last)` from the full SFC layout**; the phase-9
  restore attaches recomputed values to the right particles → restore must
  precede output (non-fluid values remain stale/undefined, risks).

**Upstream note** (pre-existing in `ve`, avoided by ve-pt; consider filing
upstream): ve's saveFields pass 2 writes a garbage `gradh` column when gradh
is a requested output (the acquired buffer holds c13 content,
`ve_hydro.hpp:257`), and pass 3 writes through gradh's empty home buffer
whenever divv/curlv are requested (caller passes `d.gradh.data()`
unconditionally, `iad_divv_curlv_gradh.hpp:49`/`_gpu.cu:48`; all four ij-loop
branches write gradh, `iad_divv_curlv_kern.hpp:110-139`) — an OOB write.

### Interim mixed-IC mechanism (`--nonfluid-frac`, fresh ICs only)

`main/src/init/utils.hpp`: add `markNonFluidFraction(d, frac)` —
**id-keyed selection** (via `generateParticleIDs`, `utils.hpp:129-145`:
host-side Allgather + exclusive scan; ids are rank-offset sequences —
**rank-count-independence** comes from the selection rule being a function of
`id`, never per-rank highest-index or per-rank-random, which is
irreproducible across rank counts and can create zero-fluid ranks mid-test).
For glass ICs `id` = global SFC order (data-dependent — **sanity-check in
M2**), so `id < f*N` selects a spatially contiguous corner chunk, not a radial
shell — use `id % stride` to spread. Introduced behind a temporary
`--nonfluid-frac F` flag (**fresh ICs only** — on a ptype-bearing restart the
mark is already stored; assert this), parsed in `sphexa.cpp` (no Config
struct — `parser.get(...)`, call right after `simInit->init(...)`, `:136`,
before the first sync `:157`; update `printHelp`) — `InitSettings` has no CLI
path today; M4 turns it into the proper per-init option (hooking beside
`runTagging` after `initImpl`, `isim_init.hpp:56-62` — id-based selection is
idempotent on ptype-bearing restarts; the **zero-fill** must not live there).
**GPU mechanics**: `generateParticleIDs` is host-MPI code, so D2H ids → host
selection → H2D `ptype` (`memcpyD2HAsync`/`memcpyH2DAsync`) — no new kernel,
no new `.cu` (`utils.hpp` is a g++ TU). **Assert the marked `h` stays
finite/sane** (non-fluid `h` feeds `egrav`, `gravity_wrapper.hpp:143`, and
`accelerationTimestep`, `ts_global.hpp:66,105` — risks).

### Propagator members landing in M2

`main/src/propagator/ve_hydro_pt.hpp` (existing M1 skeleton; template head as
`ve_hydro.hpp:48`, `KeyType` at `:57`):

- `std::optional<cstone::Halos<KeyType>> fluidHalos_` — `std::optional`
  (or `unique_ptr`) because `Halos` has no default ctor
  (`halos/halos.hpp:46`; `Halos(int, MPI_Comm)`) and the propagator ctor
  receives `std::ostream&` + rank but no `MPI_Comm` (`ve_hydro.hpp:87`), and
  is created before `simData` exists (`sphexa.cpp:123` vs `:127`).
  **Lazy-construct in the first `sync()`** from the protected `rank_`
  (`ipropagator.hpp:211`) and the public `simData.comm`
  (`simulation_data.hpp:62`, set at `sphexa.cpp:127` before the first sync
  `:157`) — the zero-API-change route. **Documented assumption**:
  `simData.comm` == the Domain's comm (both `MPI_COMM_WORLD` on every current
  path, `:127` vs `:154`; no accessor exists to assert it — a sub-comm Domain
  would deadlock; a one-line `Domain::comm()` getter would let the first sync
  assert it). Late construction is safe (`Halos` builds everything inside
  `exchangeRequests`, `halos.hpp:64-80`; nothing in `halos/` holds static
  state).
- `fluidLayoutHost_` + `fluidLayoutAcc_` — **both memory spaces** needed:
  host span for `exchangeRequests` (step 3), device pointer for the
  view/kernels (steps 2/5).
- Two dedicated `AccVector<uint64_t>` **fluid-halo scratch buffers**
  (send + receive, step 3(d)).
- Partition scratch + **GPU flags buffer** (keys-sized `uint64`, `bufSize`;
  its head serves as the flags buffer — already present from M1; M2 adds the
  carve-before-gathers discipline and the trio call).
- The step-8 **trigger flag** member (derived from `d.outputFieldIndices`).
- `ordering_`/`orderingInv_` exist from M1 (AccVector idiom,
  `domain.hpp:45`; element type `cstone::LocalIndex`). **No fluid tree storage
  or `fluidKeys_`**: the fluid view reuses the focus tree. (`driftGroups_`
  second `GroupData` is M3.)
- `cstone::Halos` gains two new **const getters** `outgoing()`/`incoming()`
  (none exist today; `halos.hpp:118-119` are the private data members
  `incomingHaloIndices_`/`outgoingHaloIndices_`) — used by the
  peer-consistency assertion.

### Tests

- **2-rank gtests run at cstone level** (the `integration_mpi` raw-vector
  harness, `domain_2ranks.cpp` via `addCstoneMpiTest` — **no 2-rank ve-pt
  propagator harness exists**): hand-set `ptype` on raw vectors covering the
  partition→fluidHalos→restore mechanics, zero-count-leaf handling, and
  **zero-fluid-rank request emptiness** on cstone types (a rank with an
  all-non-fluid layout must issue an empty request set and match nothing —
  `haloPeers` derives flags from layout ranges, `halo_peers.hpp:33`). The
  **full zero-fluid-rank run config is M2b**; here only the request-emptiness
  mechanism on cstone types. The ve-pt-level mixed round trip is covered by
  the full-sphexa 2-rank configs below.
- **Peer-consistency assertion** (turns a stale-halo-ptype hang into a clean
  assert): after `fluidHalos_.exchangeRequests`, a **debug-only**
  `MPI_Allreduce` over the P×P send/recv count matrix (via
  `outgoing()`/`incoming()`) verifying every sent request arrives, plus local
  checks: `fluidLayout.back() == nFluid`, and incoming ranges tile
  `[0:fluidFirst) ∪ [fluidLast:nFluid)`.

## Explicitly out of scope (deferred)

- **SPH-layer empty-range guards** — `rhoTimestep` guard on both paths (before
  the `divv.empty()` throws, `ts_global.hpp:82,89`), type-generic
  `minMax(Gpu)` hardening, `reduceDt` zero-group guard
  (`momentum_energy_gpu.cu:92-94`) → **M2b**.
- **Zero-fluid-rank run config** (one rank owning zero fluid particles) →
  **M2b** (M2 tests request emptiness at cstone level only).
- **`integrate` changes** — drift groups (`driftGroups_`), masked energy
  update in `computePositions`, masked acceleration zeroing (step 10),
  fluid-only h update, timestep wiring, `removeUnconvergedParticles`
  behaviors → **M3**.
- **GPU bitwise gate** (two-run bitwise check on a CUDA build) → **M3**.
- **Tolerant reader** (`IFileReader::datasetNames()` + `restoreDataset`
  skip-missing/zero-fill) and the **proper per-init option** (replacing
  `--nonfluid-frac`) → **M4**.

## Verification / acceptance

1. **All-fluid config still identical to `ve`** — same two-run bitwise gate
   as M1, default outputs (no `-f`; `conservedFields()`, `sphexa.cpp:139`),
   **`-w 10` required** (else no dump is ever written,
   `sphexa.cpp:106-107,179-183`):
   ```
   sphexa --prop ve    --init evrard --glass ./50c.h5 -n 50 -s 10 -w 10 -o ve.h5
   sphexa --prop ve-pt --init evrard --glass ./50c.h5 -n 50 -s 10 -w 10 -o vept.h5
   ```
   then the small h5py bitwise check per dataset. Caveat: with all
   `ptype = 0`, `fluidLayout == layout`, so **zero-count leaves and
   `nFluid != d.size()` are NOT exercised yet** (2-rank mixed tests cover
   them).
2. **2-rank MPI test** (cstone gtest level): identical conserved quantities
   across the partition→fluidHalos→restore round trip; the peer-consistency
   assertions above pass.
3. **Two 2-rank mixed full-sphexa configs** (`mpirun -np 2`, `--nonfluid-frac`,
   evrard glass as in the M1 gate):
   - (a) **mixed low-fraction**: crash-free, conserved momentum — **not
     bitwise** (paths diverge by design);
   - (b) **zero-count-leaf case** (a rank still has fluid, but some leaves it
     owns hold only non-fluid particles) — crash-free, correct halo coverage;
     distinct from the zero-fluid rank, which is M2b's config (b).
   - **Mixed configs must pass `-f` including `divv,curlv,gradh` at least
     once**: an output set without them never triggers the step-8
     permanent-allocation path (trigger, trio gather, rotation skip, pass-2
     `gradh`-dependent branch). These configs are exactly the ones that would
     hit ve's pass-3 OOB write — the M2 saveFields rework must land first.

## Risks & gotchas

- **Two index spaces in one propagator** (fluid-compacted vs full SFC) — the
  largest bug source; keep the single helper pair `partition_/unpartition_`
  doing *all* layout transitions and never hand a fluid index to a
  full-layout routine (esp. `rhoTimestep`, gravity, output). Exactly four
  transitions (steps 4/9, integrate 3/4), all via the helper pair;
  `ordering_` stays valid between step 9 and integrate step 3 (no sync
  between).
- **Stale neighbor lists across layout switches** — any kernel using
  `d.neighborhood` or `groups_` must only run in the layout it was built in
  (drives the saveFields design and the `driftGroups_` split; integrate
  step 2, M3).
- **MPI hygiene**: the step-3 invariant list (a)–(d) is load-bearing; the
  fencing is emergent from the two `Alltoall`s plus complete-on-return
  exchanges (`mpi_wrappers.hpp:78`; `mpi_cuda.cuh:75-93`; `MPI_Waitall` on
  sends is send-buffer safety only, **not** remote delivery — what makes the
  exchanges complete-on-return is the trailing `MPI_Waitall`,
  `exchange_halos.hpp:71-85` / `exchange_halos_gpu.cuh:96-112`). If the
  region sequence ever becomes rank-dependent or an exchange moves outside
  the fenced windows (e.g. a fluid data exchange in `integrate`), the
  colliding tag sequences corrupt — **escalation path**: give the fluid
  instance its own tag/epoch base (small cstone API change). Follow-up:
  epochs are unbounded vs `MPI_TAG_UB ≥ 32767` (base
  `P2pTags::haloExchange = 7000`) — exhaustion horizons are a pre-existing ve
  issue too.
- **Zero-fluid ranks**: the group machinery is naturally safe — GPU group
  data is always resized to ≥ 1 element (`groups_gpu.cu:93,106`) so
  `numGroups = 0` and group-indexed kernels no-op (zero-block guards,
  `positions_gpu.cu:180`); the CPU path sets `numGroups = 1` with a single
  empty group (`sph/include/sph/groups.hpp:65-70`); `haloPeers` derives flags
  from layout ranges (`halo_peers.hpp:33`) — a rank with no fluid peers posts
  and matches nothing. Exceptions needing their own guards: `reduceDt` and
  `rhoTimestep` (SPH layer) — **zero-fluid-rank guards DEFERRED to M2b**;
  M2b/M4 cover CPU and GPU.
- **Upstream note** (ve's saveFields): pass 2 garbage `gradh` column +
  pass 3 OOB write — see the saveFields section; do not re-add pass 3;
  consider filing upstream.
