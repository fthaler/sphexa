# Milestone M1 — `ve-pt` skeleton round trip (bitwise `ve` equivalence, CPU only)

Register a new propagator `ve-pt` (`main/src/propagator/ve_hydro_pt.hpp` + `ve_hydro_pt.cpp`, `--prop ve-pt`)
that is a structural copy of `ve` (`ve_hydro.hpp`) plus one new `uint8` particle field `ptype` carried
end-to-end: declared in `particles_data.hpp`, zero-filled at every init/restart path, registered
**conserved**, synced through the unchanged sync branching, and exercised by a stable-partition →
inverse-permutation round trip inside `computeForces` (identity net effect — SPH and gravity stay on the
full buffer, exactly as `ve`). ve-pt's `saveFields` appends `ptype` to every dump. Acceptance: with all
`ptype = 0`, ve-pt reproduces `ve` **bitwise** on a **CPU build** (gate below). **Depends on M0**
(fused `stablePartition`, hardened `gatherArrays`, masked fill, `GATHER_RANGES(uint8_t, unsigned)` all exist).

## Scope

1. `sph/include/sph/particles_data.hpp`: `FieldVector<uint8_t> ptype` + `"ptype"` appended at the end of
   `fieldNames` and `dataTuple()`.
2. Init/restart zero-fill of `ptype` at all 14 initializer paths (12 code anchors, listed below).
3. `ve_hydro_pt.hpp`/`.cpp`: aliases, `sync` (unchanged branching), `computeForces` (partition at step 4,
   restore at step 9, everything else verbatim `ve`), `saveFields` (ve's three passes + ptype append),
   ve's `integrate` unchanged; registration in `propagator.h`, `factory.hpp`, `CMakeLists.txt`,
   `compare_develop.py`; timers `partition` / `partitionRestore`.

Out of scope (deferred to M2/M2b/M3/M4 — see the section at the end): step-8 marking, flag-gated trio
gather, `(ptype, m)` halo exchange, `fluidHalos_`/`fluidLayout` wiring, saveFields rework, `integrate`
changes, GPU gate.

## Context you need

- From M0: `cstone::stablePartition(exec, ptype, nFluid, ordering-out, orderingInv-out)` writing **both**
  permutations in one fused pass (canonical counting formula lives in M0; stability by construction;
  covers `[0, bufSize)` incl. halos, `outputOffset 0`), and hardened `cstone::gatherArrays`
  (`domain/layout.hpp:231-261`) with a **release-active throw on `array.size() < ordering.size()`**
  guarding both `reorderArray` branches (`:237-258`). The masked zero-fill and
  `GATHER_RANGES(uint8_t, unsigned)` primitives exist but are consumed only in M2/M3.
- Buffer geometry: `[0:start)` prefix halos, `[start:end)` owned, `[end:size)` suffix halos; SFC-sorted
  over the whole buffer after sync (`domain.hpp:141-144,499-500`, `buffer_description.hpp:29-45`). M1
  has only **one index space** (full SFC over `[0:bufSize)`); the fluid-compacted space arrives in M2.
- `sync` resizes only the arrays passed to it (`domain.hpp:123,466,502`); dependent outputs are sized
  solely by the `d.resize(domain.nParticlesWithHalos())` at the `ve_hydro.hpp:135` position.
- `DeviceVector::resize` does **not** initialize device memory (`noinit_thrust.cuh:29-55`) → GPU
  garbage; CPU `std::vector` zero-initializes — hence M1's CPU-only gate, and hence the init zero-fills
  below are mandatory anyway (M3's GPU gate reads noinit garbage without them).
- Shared invariants that bind here (background: `overview.md`): `ptype` is conserved, never dependent;
  **every ve-pt dump carries `ptype`**; never restore by re-sorting on `keys` (unconverged particles
  carry `keys[i] = removeKey`, `find_neighbors.hpp:32` / `find_neighbors_gpu.cu:34`).

## Implementation

### Particle data

- `sph/include/sph/particles_data.hpp`: add `FieldVector<uint8_t> ptype;` and append `"ptype"` **at the
  end** of `fieldNames` and `dataTuple()` in matching order (`:274-298`; `FieldVariant` already supports
  uint8 `:82-83`; precedent `rung` `:257`). The order match is load-bearing three ways:
  (1) restart restore pairs `fieldNames[i]` ↔ `data()[i]` (`file_init.hpp:49-64`);
  (2) the output visit pairs name/pointer (`ipropagator.hpp:149-166`);
  (3) the FieldStates donor-swap pairs by index (`field_states.hpp:137-139,165-178`).
  The existing `static_assert` checks only size (`:294-296`) and compiles out on GCC ≤ 11 — do not rely
  on it: appending to **both** is required and sufficient.
- Register as **conserved** (never dependent) in ve-pt's `ConservedFields`: conserved fields are never
  released/acquired (`field_states.hpp:39,123-132`) and are the fields gathered on repartition
  (`domain.sync`'s conserved tuple, `ve_hydro.hpp:115-121`) — repartition survival is the load-bearing
  reason; `rung` (`ve_hydro_bdt.hpp:96`) is the uint8-in-sync precedent. `"ptype"` in `ConservedFields`
  automatically enters the sync conserved tuple — desired.
- **`ptype` is written in every ve-pt dump** (canonical statement): ve-pt's `saveFields` always appends
  `"ptype"` if absent (below); otherwise a mixed-run restart silently zero-fills everyone to fluid.
  Downstream: H5 is per-named-dataset (`writeField` ignores its column, `ifile_io_hdf5.cpp:132-139`);
  `restoreDataset` restores every conserved field found by name (`file_init.hpp:49-64` — tolerant
  skip-missing lands only at M4; today a pre-ptype restart hard-fails in `readField`);
  `compare_develop.py` prints a "keys only in data: ptype" panel and continues (`:214-232`) — **do not**
  add ptype to its `FIELDS` list (ve would produce a key-set mismatch).
- **Zero-fill `ptype` explicitly at every init/restart path** — after the final `d.resize`:
  `cstone::fill(d.exec, d.ptype.begin(), d.ptype.end(), 0)` (precedent `rung`, `file_init.hpp:258`).
  Anchors — 14 initializer paths via 12 code anchors:
  - `RadialProfile` base (`radial_profile.hpp:74`) — covers `EvrardGlassSphere`,
    `EvrardGlassSphereCooling`, `Polytrope`.
  - Direct `ISimInitializer` children: `GreshoChan` (`gresho_chan.hpp:161`), `IsobaricCubeGlass`
    (`isobaric_cube_init.hpp:213`), `KelvinHelmholtzGlass` (`kelvin_helmholtz_init.hpp:212`),
    `NohGlassSphere` (`noh_init.hpp:158`), `SedovGrid` (`sedov_init.hpp:126` — after the **second**
    resize; `:116` is the first), `SedovGlass` (`sedov_init.hpp:189`), `TurbulenceGlass`
    (`turbulence_init.hpp:133`), `WindShockGlass` (`wind_shock_init.hpp:229`).
  - `FileInit`/`TDEOrbitInit`: fill only in the missing-ptype branch (today a pre-ptype restart
    hard-fails in `readField`; M4's tolerant reader changes this).
  - `FileSplitInit`: bypasses `restoreDataset` (`file_init.hpp:249-263`) → its own **always-fill**.
  - No safe central place exists: a blanket post-`initImpl` fill would clobber restored non-zero
    `ptype` on restart. `du_m1`/`alpha` need no fill (`initFieldsAtRest`, `utils.hpp:151,153`).
  - No IO changes needed: uint8 is already in `IO::Types` (`type_config.h:30-31`) and `H5Types`
    (`h5part_wrapper.hpp:49`).

### New propagator (`main/src/propagator/ve_hydro_pt.hpp` + `ve_hydro_pt.cpp`)

- `template<bool avClean, class DomainType, class DataType> class HydroVePtProp :
  public Propagator<DomainType, DataType>` (template head as `ve_hydro.hpp:48`; `KeyType` at `:57`),
  **based on** `ve_hydro.hpp` — copy its structure; do **not** inherit `HydroVeProp`, because every
  method touching `ConservedFields` must resolve ve-pt's alias. Copy the protected aliases `T`,
  `KeyType`, `Tmass`, `MultipoleType`, `Acc`, `MHolder_t` and members `mHolder_`, `groups_`
  (`ve_hydro.hpp:56-67`).
- Aliases: `ConservedFields = FieldList<"temp", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1",
  "alpha", "id", "ptype">` (ve's `:73` + `"ptype"` appended at the end); `DependentFields_`,
  `GradVFields`, `DependentFields` as ve (`:76-84`). The extended alias must flow into **both**:
  - ve-pt's `activateFields` override — ve's base (`ve_hydro.hpp:104-107`) resolves the *base* alias,
    so inheritance alone would never `setConserved("ptype")`; without it `restoreDataset`'s
    `isConserved` gate (`file_init.hpp:52`) skips ptype and `d.resize` never allocates it
    (`field_states.hpp:61-64`);
  - ve-pt's own sync call (`get<ConservedFields>(d)`).
- `conservedFields()` override (ve's `:93-98`) **filters `"ptype"` out** so the default output selection
  stays ve-identical (`sphexa.cpp:139`).
- Members for M1 only: `ordering_` + `orderingInv_` as `AccVector<cstone::LocalIndex>` (idiom
  `std::conditional_t<HaveGpu, DeviceVector<T>, std::vector<T>>`, `domain.hpp:45`; element type
  `cstone::LocalIndex` = `unsigned`, `tree/definitions.h:27`, matches `gatherArrays`'
  `span<const LocalIndex>`, `domain/layout.hpp:232`) and one keys-sized partition scratch
  `AccVector<uint64_t>` (its head serves as the GPU partition-flags buffer in M3). **Deferred, do not
  declare yet:** `fluidHalos_`, `fluidLayoutHost_`/`fluidLayoutAcc_`, the two dedicated fluid-halo
  scratch buffers, and the step-8 trigger flag (all M2); `driftGroups_` (M3). No fluid tree storage or
  `fluidKeys_`: the fluid view reuses the focus tree (M2).
- `sync()` override: copy ve's (`ve_hydro.hpp:110-124`) verbatim — branching exactly as `:113-122`:
  `domain.syncGrav(get<"keys">(d), x, y, z, h, m, get<ConservedFields>(d), get<DependentFields>(d))`
  iff `d.g != 0`, else plain `domain.sync(keys, x, y, z, h,
  std::tuple_cat(std::tie(get<"m">(d)), get<ConservedFields>(d)), get<DependentFields>(d))` — `m`
  folded into the conserved tuple (different focus-tree convergence criteria; required for the M1/M2
  bitwise equivalence at `g == 0` too). Keep `d.treeView = domain.octreeProperties()` (`:123`). ptype
  rides in `ConservedFields` → sized by both branches. M2 adds
  `domain.exchangeHalos(std::tie(ptype, m), ax, keys)` right after the sync — **not now**: in M1, halo
  `ptype` stays zero on CPU (see the gate rationale).
- `integrate`: copy ve's (`ve_hydro.hpp:205-220`) unchanged; M3 replaces it.
- `ve_hydro_pt.cpp`: factory with the `avClean` dispatch + explicit instantiation block, copying
  `ve_hydro.cpp:41-47,58-64` verbatim, renamed:

  ```cpp
  template<class DomainType, class ParticleDataType>
  std::unique_ptr<Propagator<DomainType, ParticleDataType>>
  PropLib<DomainType, ParticleDataType>::makeHydroVePtProp(std::ostream& output, size_t rank, bool avClean)
  {
      if (avClean) { return std::make_unique<HydroVePtProp<true, DomainType, ParticleDataType>>(output, rank); }
      else         { return std::make_unique<HydroVePtProp<false, DomainType, ParticleDataType>>(output, rank); }
  }
  #ifdef USE_CUDA
  template struct PropLib<cstone::Domain<SphTypes::KeyType, SphTypes::CoordinateType, cstone::execution::Gpu>,
                          SimulationData<cstone::execution::Gpu>>;
  #else
  template struct PropLib<cstone::Domain<SphTypes::KeyType, SphTypes::CoordinateType, cstone::execution::Cpu>,
                          SimulationData<cstone::execution::Cpu>>;
  #endif
  ```

- Timers (`timer.step(...)`, no registration): five planned names — `"fluidLayout"`, `"partition"`,
  `"fluidView"`, `"fluidHaloExchange"`, `"partitionRestore"`; M1 fires only `"partition"` (step 4) and
  `"partitionRestore"` (step 9), the other three are reserved for M2.
- Perf note: the only MPI-latency-scale addition overall is the single `(ptype, m)` full-pattern
  exchange per step (M2; the fluid exchanges *replace* domain's mid-phase exchanges, count unchanged);
  the four full-buffer permutations per step move ~1.3–1.5 KB/particle across ve's 32 allocated fields
  (38 with `avClean`) — bandwidth-negligible against the ij-loops/traversal. No mitigation beyond the
  timers.

### computeForces (M1 deltas vs `ve_hydro.hpp:126-203`)

Copy ve's `computeForces` and insert exactly three things; everything else stays in ve's order.

1. **Partition (step 4), right after `d.resize(domain.nParticlesWithHalos())` (`:135` position):**

   ```cpp
   size_t bufSize = domain.nParticlesWithHalos();
   size_t nFluid  = d.size();   // M1 placeholder: all particles fluid; M2 replaces with fluidLayout.back()
   ordering_.resize(bufSize); orderingInv_.resize(bufSize); partitionScratch_.resize(bufSize);
   cstone::stablePartition(domain.exec(), rawPtr(d.ptype), nFluid, ordering_, orderingInv_); // M0 primitive
   timer.step("partition");
   ```

   With all `ptype == 0` the fused pass yields the identity permutation (`f[i] = 0`, `dest = i`); M1
   exercises the round trip, not a behavior change. Keep ve's
   `fillMassHalos(domain.exec(), get<"m">(d), first, last)` (`:139`) **exactly where ve has it**: it
   fills `[0:first)` **and** `[last:buffer end)` (`particles_data.hpp:415-429`); dropping it before
   M2's step-1 `(ptype, m)` exchange leaves halo `m` garbage — `computeXMass` + gravity's upsweep
   consume `m` over the whole buffer incl. halos (`gravity_wrapper.hpp:128-133` GPU, `:62-70` CPU) —
   and breaks this gate. The step-8 marking site is **not** added in M1.

2. **Apply `ordering_` (gather) to the fixed compile-time list**, flattened with `std::tuple_cat` into
   the single flat tuple `gatherArrays` takes (`domain/layout.hpp:234` — a nested FieldList tuple would
   not compile), in the same field order as the sync gather (`ve_hydro.hpp:115-121`, keys first):

   ```cpp
   auto gatherList = std::tuple_cat(std::tie(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d)),
                                    std::tie(get<"m">(d)), get<ConservedFields>(d), get<DependentFields>(d));
   cstone::gatherArrays(domain.exec(), ordering_, 0, gatherList, std::tie(partitionScratch_));
   ```

   - `ConservedFields` includes `ptype` (ptype itself is gathered); `DependentFields` includes
     `dV11-dV33` under `avClean`; `keys` is `setDependent` (`ve_hydro.hpp:105`) but **not** in
     `DependentFields` — list it explicitly.
   - **Release-safe per-field `size == bufSize` assert** as a fold over the tuple — a **throw**, not
     `assert` (both `assert` and `gatherArrays`' own debug asserts, `layout.hpp:244,252`, compile out
     in release; M0's hardened `gatherArrays` throw is the size-0/null-source backstop).
   - **Scratch rules:** one keys-sized (`uint64`, bufSize) scratch serves the whole set (the reuse
     branch accepts any scratch with element size ≥ the array's via `reinterpret_cast`,
     `layout.hpp:250-256`, including `double` fields, `sph/types.hpp:42`); never resized after step 4;
     **never cache raw pointers across an apply** (the exact-type swap branch for `keys`/`id` toggles
     buffer identity). Distinct from M2's fluid-halo scratch. On GPU the partition flags buffer is
     carved from this scratch before the gathers (M3 path).
   - The flag-gated second `gatherArrays` call over `std::tie(divv, curlv, gradh)` is **deferred to
     M2**; in M1 the trio is not permanently allocated and must not be gathered.

3. **Restore (step 9), as the last action of `computeForces` (after ve's gravity block `:189-202`):**

   ```cpp
   assert(orderingInv_.size() == bufSize);  // plus a release-active throw: the fused kernel writes one
                                            // entry per i, but debug asserts compile out
   cstone::gatherArrays(domain.exec(), orderingInv_, 0, gatherList, std::tie(partitionScratch_));
   d.treeView = domain.octreeProperties();
   timer.step("partitionRestore");
   ```

   The restore is the **inverse permutation as a gather** (`orderingInv_[ordering[j]] = j`, written by
   M0's fused pass alongside `ordering_[dest] = i`; `dst[i] = src[orderingInv_[i]]`) — never a re-sort
   on `keys`. Same fixed list, no trio call in M1.

- **SPH pipeline and gravity stay EXACTLY as ve** — full buffer: `computeGroups` …
  `computeMomentumEnergy` (`ve_hydro.hpp:141-187`, field rotations `:159-160` / `:183-184` as-is) and
  gravity as `:189-202` (`mHolder_` groups from `domain.focusTree()` + `domain.layout()`; the
  full-buffer view is correct here, `layout.back() == bufSize`). No `fluidFirst`/`fluidLast`, no
  `fluidHalos_`, no `totalBodies` plumbing in M1.

### saveFields

Copy ve's `saveFields` (`ve_hydro.hpp:222-287`) **including all three passes and the trailing unused-
fields warning**, with one change — the ptype-append rule: after the local copies
`indicesDone = d.outputFieldIndices; namesDone = d.outputFieldNames;` (`:227-228`), append `"ptype"`
(name + its field index) to **both** copies **iff absent from `d.outputFieldIndices`** (a `-f ptype`
run would otherwise double-write it). Pass 1 then writes `outputFieldIndices ∩ isAllocated` plus ptype:
the existing find-based column computation (`:237-238`) yields `col = outputFieldIndices.size()` for
ptype, and `AsciiWriter`'s sort-by-column (`ifile_io_ascii.cpp:82-84,116`) deterministically places it
**last** — no column-lookup change; H5 is per-named-dataset (`ifile_io_hdf5.cpp:132-139`), unaffected.
The `isAllocated` gate (`:235`) keeps unallocated fluid-only outputs out of the file; `ptype` is
conserved → always allocated. Pass-2 `gradh` symmetry and the pass-3 drop are **deferred to M2** — M1
keeps ve's pass structure; the gate runs the default output set, so `-f divv,curlv,gradh` configs are
not covered by M1 (they intentionally diverge and still take ve's pass-3 path with its upstream OOB
write).

### Registration & build

- `propagator.h` (beside `:50`): `static PropPtr makeHydroVePtProp(std::ostream& output, size_t rank, bool avClean);`
- `main/src/propagator/factory.hpp` (beside `:46`):
  `if (choice == "ve-pt") { return PropLib<DomainType, ParticleDataType>::makeHydroVePtProp(output, rank, avClean); }`
- `main/src/propagator/CMakeLists.txt:12`: append to
  `set(PROP_SOURCES nbody.cpp std_hydro_grackle.cpp std_hydro.cpp ve_hydro.cpp ve_hydro_bdt.cpp ve_hydro_pt.cpp)`
  — both `propagator` (`:14`) and `propagator_gpu` (`:22-24`, built from the same list with
  `USE_CUDA`) compile it with g++, so no thrust calls may appear in it.
- `compare_develop.py:67`:
  `type Prop = Literal["nbody", "std", "turbulence", "turbulence-ve", "ve", "ve-bdt", "ve-pt"]`.
  Do **not** touch its `FIELDS` list.

## Explicitly out of scope (deferred)

- **M2**: step-8 trigger flag + trio `setDependent` marking and the step-1 marking site; the flag-gated
  trio `gatherArrays` call in steps 4/9; the `domain.exchangeHalos(std::tie(ptype, m), ax, keys)` call
  (and, exactly then, the `fillMassHalos` drop); `fluidHalos_` (second `cstone::Halos` instance),
  `fluidLayoutHost_`/`fluidLayoutAcc_`, fluid leaf counts (step 2), fluid tree view (step 5),
  fluid-only SPH pipeline (step 7) with fluid halo exchanges + dedicated scratch buffers; saveFields
  rework (pass-2 `gradh` symmetry, pass-3 drop); `--nonfluid-frac` + `markNonFluidFraction`; 2-rank
  cstone-level gtests.
- **M2b**: SPH empty-range hardening (`rhoTimestep`, `minMax(Gpu)`, `reduceDt` guards).
- **M3**: `integrate` rework (warp-sized `driftGroups_`, masked energy update passing
  `rawPtr(d.ptype)`, masked acceleration zeroing — step 10, fluid-only h update, the integrate 3/4
  permutation re-applies); the GPU bitwise gate.
- **M4**: tolerant restart reader (`IFileReader::datasetNames()` + `restoreDataset` skip-missing +
  ptype zero-fill), proper per-init option, restart acceptance gates.

## Verification / acceptance

**CPU build only.** Two runs, **no `-f`** (default outputs = `conservedFields()`, `sphexa.cpp:139`),
same rank count (1):

```
sphexa --prop ve    --init evrard --glass ./50c.h5 -n 50 -s 10 -w 10 -o ve.h5
sphexa --prop ve-pt --init evrard --glass ./50c.h5 -n 50 -s 10 -w 10 -o vept.h5
```

`-w 10` is **required**: without it no dump is ever written (`isOutputTriggered` never fires,
`sphexa.cpp:106-107,179-183`). Then the h5py bitwise check:

```python
import h5py, numpy as np
ve, vept = h5py.File("ve.h5"), h5py.File("vept.h5")
assert sorted(ve.keys()) == sorted(vept.keys())                      # same step groups
for step in ve.keys():
    assert (set(ve[step].keys()) | {"ptype"}) == set(vept[step].keys())
    for name in ve[step].keys():                                     # every ve-named dataset
        assert np.array_equal(ve[step][name][()], vept[step][name][()]), (step, name)
    assert np.all(vept[step]["ptype"][()] == 0)                      # ptype all zero
```

Acceptance: with all `ptype = 0`, ve-pt reproduces `ve` **bitwise** (the identity-permutation round
trip must not perturb any value). Presuppositions:

- **Equal-mass ICs** (evrard glass): with varying masses ve-pt intentionally diverges — ve's
  homogeneous `fillMassHalos` mis-values halo masses until M2.
- **Default output set**: requesting `divv`/`curlv`/`gradh` takes different IAD/saveFields branches —
  intentional divergences (the trio path is M2's).
- `compare_develop.py` **cannot drive the gate** (rtol diff, no exit code) — use the h5py check.

Why the gate is **CPU only**: the step-4 partition reads halo `ptype` that nothing delivers until M2's
exchange (`setupHalos` exchanges only `x, y, z, h`, `domain.hpp:481-489`; contract `:132-139`) —
0 on CPU only by the `std::vector` zero-init accident, garbage on GPU (`noinit_thrust.cuh:29-55`).
The GPU gate (same two-run bitwise check) lands at M3.

## Risks & gotchas

- **Order match is load-bearing**: `fieldNames` and `dataTuple()` must both get `"ptype"` at the end;
  three consumers pair by index (restart restore `file_init.hpp:49-64`, output visit
  `ipropagator.hpp:149-166`, donor-swap `field_states.hpp:137-139,165-178`). The `static_assert`
  (`particles_data.hpp:294-296`) compiles out on GCC ≤ 11 — don't rely on it.
- **CPU halo-`ptype` accident**: M1's CPU correctness rests on `std::vector` zero-init of halo slots.
  Never extend this gate to a GPU build; never drop the init zero-fills as "redundant".
- **`fillMassHalos` must stay** until M2's `(ptype, m)` exchange lands, or the gate breaks (halo `m`
  garbage feeds `computeXMass` and gravity's upsweep).
- **Gather list shape**: one flat tuple via `tuple_cat` (a nested FieldList tuple won't compile,
  `domain/layout.hpp:234`); `keys` must be listed explicitly (it is `setDependent` but not in
  `DependentFields`); `ptype` rides in via `ConservedFields`; order = sync order, keys first.
- **Never cache raw pointers across a `gatherArrays` apply** — the exact-type swap branch
  (`layout.hpp:241-247`) toggles buffer identity for `keys`/`id`.
- **Release-safe checks must throw, not `assert`** — debug asserts compile out in release; run the gate
  on a release build too.
- `nFluid = d.size()` is an M1 placeholder; M2 must replace it with `fluidLayout.back()` (step 2).
- `SedovGrid`'s fill goes after the **second** resize (`sedov_init.hpp:126`), not the first (`:116`).
- `compare_develop.py`: add only the `"ve-pt"` literal — adding `ptype` to `FIELDS` breaks ve
  comparisons (key-set mismatch).
- Do **not** run acceptance with `-f divv,curlv,gradh`: ve's pre-existing pass-3 OOB write
  (`iad_divv_curlv_gradh.hpp:49` / `_gpu.cu:48`) is only removed by M2's saveFields rework.
