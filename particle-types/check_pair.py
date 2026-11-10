#!/usr/bin/env python3
"""Generic bitwise pair comparison: same step groups, same dataset names, bitwise-equal values.

Usage: python3 check_pair.py a.h5 b.h5
No ptype expectations (use check_gate.py for the M1/M3 ve-vs-ve-pt gate).
"""
import sys

import h5py
import numpy as np

a_path, b_path = sys.argv[1], sys.argv[2]
a, b = h5py.File(a_path, "r"), h5py.File(b_path, "r")

assert sorted(a.keys()) == sorted(b.keys()), (
    f"step groups differ:\n  {a_path}: {sorted(a.keys())}\n  {b_path}: {sorted(b.keys())}"
)
print(f"step groups match: {sorted(a.keys())}")

n_checked = 0
for step in a.keys():
    assert sorted(a[step].keys()) == sorted(b[step].keys()), (
        f"{step}: dataset names differ:"
        f"\n  {a_path}: {sorted(a[step].keys())}"
        f"\n  {b_path}: {sorted(b[step].keys())}"
    )
    for name in sorted(a[step].keys()):
        assert np.array_equal(a[step][name][()], b[step][name][()]), f"{step}/{name} differs bitwise"
        n_checked += 1

print(f"OK: {n_checked} datasets bitwise-identical across {len(a.keys())} steps")
