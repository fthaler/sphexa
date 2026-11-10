#!/usr/bin/env python3
"""M1/M3 acceptance gate: `ve` vs `ve-pt` comparison (particle-types/milestone-1.md, milestone-3.md).

Usage:
  python3 check_gate.py [ve.h5] [vept.h5]                  # CPU gate: bitwise per dataset
  python3 check_gate.py --tolerance TOL [ve.h5] [vept.h5]  # GPU gate: float datasets maxabs <= TOL
                                                           # (amended criterion, milestone-3.md
                                                           # 2026-11-10; integer datasets stay exact)
"""
import sys

import h5py
import numpy as np

args = sys.argv[1:]
tolerance = None
if args and args[0] == "--tolerance":
    tolerance = float(args[1])
    args = args[2:]
ve_path = args[0] if len(args) > 0 else "ve.h5"
vept_path = args[1] if len(args) > 1 else "vept.h5"

ve, vept = h5py.File(ve_path, "r"), h5py.File(vept_path, "r")

assert sorted(ve.keys()) == sorted(vept.keys()), (
    f"step groups differ:\n  ve  : {sorted(ve.keys())}\n  vept: {sorted(vept.keys())}"
)
print(f"step groups match: {sorted(ve.keys())}")

n_checked = 0
worst = ("", 0.0)
for step in ve.keys():
    assert (set(ve[step].keys()) | {"ptype"}) == set(vept[step].keys()), (
        f"{step}: dataset names differ:"
        f"\n  ve  : {sorted(ve[step].keys())}"
        f"\n  vept: {sorted(vept[step].keys())}"
    )
    for name in ve[step].keys():  # every ve-named dataset
        a, b = ve[step][name][()], vept[step][name][()]
        if a.dtype.kind in "iu":
            assert np.array_equal(a, b), f"{step}/{name} differs bitwise"
        elif tolerance is not None:
            maxabs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            assert maxabs <= tolerance, f"{step}/{name} maxabs {maxabs:.3e} > tolerance {tolerance:g}"
            if maxabs > worst[1]:
                worst = (f"{step}/{name}", maxabs)
        else:
            assert np.array_equal(a, b), f"{step}/{name} differs bitwise"
        n_checked += 1
    assert np.all(vept[step]["ptype"][()] == 0), f"{step}/ptype not all zero"

if tolerance is not None:
    print(f"OK (tolerance {tolerance:g}): {n_checked} datasets within tolerance across "
          f"{len(ve.keys())} steps; ptype all zero; worst dataset {worst[0]} maxabs {worst[1]:.3e}")
else:
    print(f"OK: {n_checked} datasets bitwise-identical across {len(ve.keys())} steps; ptype all zero; extra dataset: ptype")
