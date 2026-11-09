#!/usr/bin/env python3
"""M1 acceptance gate: bitwise `ve` vs `ve-pt` comparison (particle-types/milestone-1.md).

Usage: python3 check_gate.py [ve.h5] [vept.h5]
"""
import sys

import h5py
import numpy as np

ve_path = sys.argv[1] if len(sys.argv) > 1 else "ve.h5"
vept_path = sys.argv[2] if len(sys.argv) > 2 else "vept.h5"

ve, vept = h5py.File(ve_path, "r"), h5py.File(vept_path, "r")

assert sorted(ve.keys()) == sorted(vept.keys()), (
    f"step groups differ:\n  ve  : {sorted(ve.keys())}\n  vept: {sorted(vept.keys())}"
)
print(f"step groups match: {sorted(ve.keys())}")

n_checked = 0
for step in ve.keys():
    assert (set(ve[step].keys()) | {"ptype"}) == set(vept[step].keys()), (
        f"{step}: dataset names differ:"
        f"\n  ve  : {sorted(ve[step].keys())}"
        f"\n  vept: {sorted(vept[step].keys())}"
    )
    for name in ve[step].keys():  # every ve-named dataset
        a, b = ve[step][name][()], vept[step][name][()]
        assert np.array_equal(a, b), f"{step}/{name} differs bitwise"
        n_checked += 1
    assert np.all(vept[step]["ptype"][()] == 0), f"{step}/ptype not all zero"

print(f"OK: {n_checked} datasets bitwise-identical across {len(ve.keys())} steps; ptype all zero; extra dataset: ptype")
