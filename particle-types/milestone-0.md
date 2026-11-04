# Milestone M0 — cstone primitives: fused stable partition, fluid-layout helper, masked zero-fill, uint8 gather ranges

First milestone; **no dependencies**. Adds, entirely inside cstone, the five primitive-level pieces the `ve-pt`
propagator consumes in M1–M3: (1) a fused `stablePartition` writing **both** permutations of the fluid compaction
in one pass, (2) the fluid-layout helper computing `fluidLayout` (offset per focus-tree leaf, host **and** device
spaces), (3) a masked zero-fill (CPU + GPU), (4) the `GATHER_RANGES(uint8_t, unsigned)` halo-gather instantiation,
and (5) a release-active `gatherArrays` hardening (throw on an array shorter than the ordering). Delivered with CPU
unit tests under `domain/test/unit`, GPU unit tests for the fused partition kernel, and a 2-rank MPI gtest
exercising uint8 through the existing domain `exchangeHalos` path. No propagator is touched; `ve` is unchanged.

Ref convention (per `overview.md`): cstone-header refs like `layout.hpp:231` are relative to
`domain/include/cstone/`; `sph/` and `main/src/` refs are repo-root relative. Verified full paths are given where
this milestone edits a file.

## Scope

1. `stablePartition(exec, ptype, nFluid, ordering-out, orderingInv-out)` — policy-dispatched (CPU + GPU), both
   permutations in one fused pass. Files: `primitives/primitives_acc.hpp` (CPU), `primitives_gpu.h` +
   `primitives_gpu.cu` (GPU, explicit instantiations).
2. Fluid-layout helper: segmented count of `ptype == 0` over the `domain.layout()` segments + exclusive scan,
   filling host + device buffers, both reallocated to `numLeafNodes + 1` per call. Same files.
3. Masked zero-fill: CPU variant in `primitives_acc.hpp` next to `fill` (`:29-33`), GPU kernel + explicit
   instantiation in `primitives_gpu.cu`.
4. `halos/gather_halos_gpu.cu:60-74`: add `GATHER_RANGES(uint8_t, unsigned)`.
5. `domain/layout.hpp:231-261` (`gatherArrays`/`reorderArray`): release-active throw on
   `array.size() < ordering.size()`, guarding both branches.
6. Unit tests (see Verification). CMake: **no changes** expected (step 6 below).

## Context you need

- **Terminology.** `ptype` is a `uint8` field: `0` = fluid (SPH + gravity), non-zero = gravity-only. `bufSize` =
  full local buffer length incl. halos; `nFluid` = number of fluid particles in the whole buffer
  (`nFluid = fluidLayout.back()`, a host-side scalar). `ordering_` = gather permutation
  (`dst[i] = src[ordering_[i]]` compacts fluid to the front); `orderingInv_` = its inverse
  (`orderingInv_[ordering_[j]] = j`; restore is the inverse applied as a gather). **Element type:
  `cstone::LocalIndex`** (`= unsigned`, `tree/definitions.h:27`) for both permutations — matches `gatherArrays`'
  `span<const LocalIndex>` (`domain/layout.hpp:232`) and the existing unsigned gather instantiations. The
  propagator (M1) applies `ordering_` via `gatherArrays` with outputOffset 0.
- **Buffer geometry / stability premise** (canonical): the local buffer is `[0:start)` prefix halos,
  `[start:end)` owned, `[end:size)` suffix halos — halos on *both* sides (`domain.hpp:499-500`,
  `buffer_description.hpp:29-45`) — SFC-sorted over the whole buffer after sync (`domain.hpp:141-144`). The
  compaction must therefore be **stable**: the fluid subset stays SFC-sorted and in leaf order, with the same
  `[halo-pre | owned | halo-post]` geometry — `fluidFirst = fluidLayout[startCell]`,
  `fluidLast = fluidLayout[endCell]`, where `fluidLayout` is the offset-per-focus-tree-leaf layout from
  **fluid-only leaf counts** (segmented count of `ptype == 0` over `domain.layout()` + exclusive scan; **no
  communication** — the full sync delivered complete halos per flagged leaf: whole-leaf halo delivery,
  `exchange_keys.hpp:42-43,80-88`). The M0 brute-force tests encode exactly this invariant.
- **Device residency.** `ordering_`/`orderingInv_` and the partition scratch need `DeviceVector` storage on GPU
  (the GPU gather kernel reads the ordering on device, `primitives_gpu.cu:75-89`) — the `AccVector` idiom
  (`std::conditional_t<HaveGpu, DeviceVector<T>, std::vector<T>>`, `domain.hpp:45`). A host/device duality is
  needed only for `fluidLayout`. The propagator members are declared in M1; M0's primitives must accept that
  storage (raw pointers / spans).
- **Instantiation facts** (verified at plan-writing time): no new *gather* instantiation needed — `gather(Gpu,
  unsigned ordering)` covers `uint8_t`, `double`, `float`, `uint64_t` (= `unsigned long` on LP64) etc.
  (`primitives_gpu.cu:105-113`); `sequence(unsigned)` (`:272`) covers iota. No other missing instantiation found.

## Implementation

### 1. `stablePartition` — fused, both permutations

Policy-dispatched; GPU declaration in `primitives_gpu.h`, CPU overload in `primitives/primitives_acc.hpp`.
Signature shape (names adjustable, semantics fixed): `stablePartition(exec, const uint8_t* ptype, LocalIndex n,
LocalIndex nFluid, LocalIndex* ordering, LocalIndex* orderingInv, unsigned* workspace)` — `n == bufSize`; all
buffers length `n`; covers `[0, bufSize)` including halos; `workspace` ≥ `n` unsigned (GPU scan buffer,
CPU-ignored).

- **Formula (canonical):** `f` = exclusive scan of the **non-fluid** flag — `f[i]` counts `ptype != 0` in `[0:i)`;
  a literal flag reading silently corrupts every partition — `dest = ptype[i]==0 ? i - f[i] : nFluid + f[i]`;
  write `ordering_[dest] = i` **and** `orderingInv_[i] = dest` simultaneously (stability by construction; covers
  `[0, bufSize)` incl. halos, outputOffset 0).
- **CPU** (`execution::Cpu`): counting-sort loop + trivial `orderingInv[ordering[i]] = i` loop.
- **GPU:** (a) flags into `workspace` (`workspace[i] = ptype[i] != 0`); (b) in-place
  `exclusiveScan(exec, unsigned)` over it (`primitives_gpu.h:164-168`, already instantiated
  `primitives_gpu.cu:378`) → `f` in place; (c) one kernel computing `dest` per `i`, writing both outputs. **No
  scatter primitive required**; keep `scatter` with `unsigned` ordering as an unfused fallback only
  (`primitives_gpu.cu:137-145` instantiates `int` ordering — that instantiation would need adding; do **not** add
  it now). cub `sortByKey` rejected: needs 2n+2n DoubleBuffer alternates + temp beyond the 8n scratch.
- `nFluid` is a host-side input (caller's `fluidLayout.back()`); never recompute it inside the partition.

### 2. Fluid-layout helper (implements computeForces step 2)

Per step 2: segmented count of `ptype == 0` over the `domain.layout()` segments + exclusive scan. No public host
layout accessor (`domain.hpp:359` device span; host `layout_` private, `:657-658`) and no host `ptype` copy → on
GPU the count runs **device-side** (segmented-count kernel + device `exclusiveScan(unsigned)`,
`primitives_gpu.h:164-168`, instantiated `primitives_gpu.cu:378`; **no count-by-predicate primitive exists** —
write one). Then D2H into the host buffer (`memcpyD2HAsync` + `syncGpu`, staging
`octree_focus_mpi.hpp:562-571`; the stream is already drained by the halo exchange's `cudaStreamSynchronize`,
`exchange_halos_gpu.cuh:110`); CPU computes on the host.

- **Both memory spaces needed** — host span for `exchangeRequests` (step 3), device pointer for the fluid
  view/kernels (step 5) — keep the `fluidLayoutHost_` + `fluidLayoutAcc_` naming for the caller-side buffers; the
  helper fills both and **reallocates both to `numLeafNodes + 1` every call** (`domain.hpp:205` pattern).
- Semantics: `fluidLayout[i]` = fluid particles in all leaves before leaf `i`; `fluidLayout.back() == nFluid`;
  zero-count leaves give empty ranges (`fluidLayout[i] == fluidLayout[i+1]`); all-fluid gives
  `fluidLayout == layout`.
- **Not alternatives:** `computeNodeLayout` (presence flags + total counts, no predicate, `layout.hpp:153-185`);
  `cstone::computeNodeCounts` (sorted keys + tree, `csarray.hpp:182`).
- `nFluid = fluidLayout.back()` feeds the fused partition (step 4) and `totalBodies` (step 5, M2).

### 3. Masked zero-fill

- **CPU:** add to `primitives/primitives_acc.hpp` next to `fill` (`:29-33`): a masked fill writing `value` where
  the mask element is non-zero, e.g. `fillMasked(exec, first, last, maskFirst, value)`; empty-range no-op like
  `fill`.
- **GPU:** kernel + **explicit instantiation** in `primitives_gpu.cu`, declared in `primitives_gpu.h` (the
  existing `fill`, `primitives_gpu.h:30-37`, is unconditional — no masked-fill primitive exists).
- Semantics fixed by its single consumer (step 10, M3): zero `ax/ay/az` where `ptype != 0` over
  `[startIndex():endIndex())` — mask = `ptype`, predicate `mask[i] != 0` → write. Keep it generic (value + mask
  pointer) so the instantiation covers `float`/`double` destinations. Wiring into step 10 is M3 — here only the
  primitive + test.

### 4. `GATHER_RANGES(uint8_t, unsigned)`

- `halos/gather_halos_gpu.cu:60-74`: add `GATHER_RANGES(uint8_t, unsigned)` — required for `ptype` through GPU
  `exchangeHalos` (only `int` + `util::array<float,1..4>` instantiated today; `rung` never passes through
  `exchangeHalos` in any propagator).
- Send path only (receive unpacks via `memcpyD2DAsync`, `exchange_halos_gpu.cuh:105-108`; `PackType<uint8_t>` is
  `uint8_t`, `pack_buffers.hpp:53`).
- Lands at M0 (M0's uint8 2-rank test needs it to link); consumed by M2's `(ptype, m)` exchange.

### 5. `gatherArrays` hardening

File `domain/include/cstone/domain/layout.hpp`: `gatherArrays` at `:231-261`, lambda `reorderArray` at
`:237-258`, two branches — exact-type swap `:241-247` (gather from `rawPtr(array)`, then `swap`) and generic
`reinterpret_cast` branch `:248-257`.

- Add one release-active check at the top of `reorderArray` (covers **both** branches): throw on
  `array.size() < ordering.size()`. The existing asserts `:244,252` compile out in release; a size-0
  (unallocated) field gathered with a non-empty ordering reads `rawPtr(array) == nullptr` for
  `ordering.size()` elements — silent OOB today, in **both** branches (the exact-type swap branch `:245` reads
  the null source too).
- Condition is strict `<` so domain's shorter-ordering calls (`domain.hpp:188-189,328`) keep passing — those
  legally gather with `ordering.size() <= array.size()`.
- Do not change `outputOffset` semantics; the propagator-side per-field `size == bufSize` fold is M1.

### 6. CMake — no changes

All GPU paths (segmented ptype-count, stablePartition, masked fill) live in the **existing** `primitives_gpu.cu`
with **explicit instantiations**, following `primitives_gpu.cu:92-145` — `primitives/CMakeLists.txt:6` compiles
exactly that TU, so zero CMake edits. A *new* `.cu` would need edits in `primitives/CMakeLists.txt:6` **and**
`domain/include/cstone/CMakeLists.txt:10-16` — avoid. Rationale: propagator sources compile with g++ even in CUDA
builds (`main/src/propagator/CMakeLists.txt:12,22-27` builds `propagator_gpu` from the same `PROP_SOURCES` `.cpp`
list with `USE_CUDA` defined), so thrust device algorithms called from `ve_hydro_pt.cpp` would fail to
compile/link.

## Explicitly out of scope (deferred)

- **M1** — wiring the primitives into the `ve-pt` propagator: `ordering_`/`orderingInv_` members, partition →
  restore round trip (computeForces steps 4/9), per-field `size == bufSize` fold, flags buffer carved from the
  keys-sized scratch; `ptype` particle-data field.
- **M2** — consumption of `GATHER_RANGES(uint8_t, unsigned)` by the step-1 `(ptype, m)` exchange; fluid-instance
  halo tests; the fluid `OctreeNsView` (step 5), `fluidHalos_` exchanges and `totalBodies` plumbing that consume
  the helper's device pointer.
- **M3** — wiring the masked zero-fill into computeForces step 10 (non-fluid `ax/ay/az` reset before gravity).

## Verification / acceptance

- CPU unit tests under `domain/test/unit/...`, following `unit/domain/layout.cpp:104-123`
  (`TEST(Layout, gatherArrays)`: `ordering{2,1,3,4}`, `outOffset = 1`, float + **uint8** vectors — uint8
  exercised at `:108` — gathered via `gatherArrays(execution::cpu, {ordering.data(), ordering.size()}, outOffset,
  std::tie(a, b), std::tie(scratch))`).
- **stablePartition** (CPU + GPU): edge + randomized patterns — all-fluid (`nFluid == bufSize`), all-non-fluid
  (`nFluid == 0`), single fluid particle, alternating flags. Assert both outputs written for every `i ∈ [0, n)`;
  `orderingInv_[ordering_[j]] == j`; **stability** (fluid keep relative order, non-fluid keep theirs — this
  preserves SFC/leaf order and the `[halo-pre | owned | halo-post]` geometry); a gather with `ordering_` puts
  fluid at `[0:nFluid)` in original order.
- **Fluid-layout brute force:** build a leaf layout over a small buffer, compute per-leaf fluid counts by brute
  force, check `fluidLayout` equals the exclusive scan — **including zero-count leaves** and the
  **`nFluid < bufSize` padding** case (arising for the first time here). The brute-force check enforces the
  **whole-leaf halo-delivery invariant** (per-leaf counts over the full buffer incl. halos are complete and
  peer-consistent — no communication). Also assert `fluidLayout == layout` for the all-fluid case.
- **gatherArrays hardening:** a unit test containing an unallocated (size-0) field with a non-empty ordering must
  **throw**; run in **debug AND release** (the null-source gather is silent OOB today — asserts compile out).
  Cover **both** branches: one size-0 field matching the scratch tuple type exactly (swap branch) and one that
  doesn't (reinterpret_cast branch). Confirm domain's shorter-ordering calls (`domain.hpp:188-189,328`) keep
  passing — keep the existing `Layout.gatherArrays` test green and add a shorter-ordering positive case.
- **Masked-fill unit test:** CPU + GPU — masked positions written, unmasked untouched (pre-fill with a sentinel),
  empty-range no-op.
- **GPU unit tests for the fused partition kernel** (CUDA build): same case matrix as CPU.
- **`GATHER_RANGES(uint8_t, unsigned)`:** exercised through the **existing** domain `exchangeHalos` path — a
  2-rank MPI gtest (cstone-level `addCstoneMpiTest` harness, cf. `domain_2ranks.cpp`) syncing and exchanging a
  `uint8_t` field; it links only with the new instantiation. (The fluid-instance test belongs to M2.)
- Build & run the domain unit-test suite in **debug and release**; GPU tests additionally under the CUDA build.

## Risks & gotchas

- **Formula trap:** `f` must be the exclusive scan of the **non-fluid** flag (`f[i]` counts `ptype != 0` in
  `[0:i)`); a literal flag reading silently corrupts every partition.
- **Simultaneous writes:** `ordering_[dest] = i` and `orderingInv_[i] = dest` in one pass — do not build
  `orderingInv_` as a second scatter pass (hence no new scatter instantiation).
- **Throw vs assert:** `gatherArrays`' asserts (`:244,252`) compile out in release — the hardening must be a real
  `throw`, strictly `<`, or domain's shorter-ordering calls break.
- **Both branches:** the exact-type swap branch also reads a null source for a size-0 field; a test covering only
  the reinterpret_cast branch misses it.
- **`DeviceVector::resize` does not initialize** (`noinit_thrust.cuh:29-55`) — fine for the fused partition and
  the fluid layout (every entry written), but tests must compare full buffers, never assume zero-fill.
- **Both memory spaces for `fluidLayout`:** host span for `exchangeRequests` (step 3), device pointer for the
  view/kernels (step 5) — a single-space helper breaks M2; keep `fluidLayoutHost_`/`fluidLayoutAcc_` naming.
- **Realloc per call:** both `fluidLayout` buffers to `numLeafNodes + 1` every call — the leaf count changes
  after rebalancing; a stale size is a wrong-layout bug surfacing as an MPI size-mismatch hang in M2's fluid
  exchanges.
- **Element type:** `cstone::LocalIndex` for both permutations; `int` orderings won't match `gatherArrays`' span.
