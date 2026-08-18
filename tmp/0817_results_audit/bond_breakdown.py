import glob
import os

import h5py
import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))


def group_values(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with h5py.File(path, "r") as h5:
        for group_name in h5.keys():
            group = h5[group_name]
            values = {}
            for key in ("Eh", "Ev", "Es_avg", "ns", "ns_avg"):
                if key in group:
                    data = group[key][()]
                    values[key] = np.asarray(data).tolist()
            out[group_name] = values
    return out


for case_path in sorted(glob.glob(os.path.join(ROOT, "testfreefermion_*"))):
    case = os.path.basename(case_path)
    square = os.path.join(case_path, "examples", "Spinless_Fermion_2D_Square", "exp_ctmrg.h5")
    nested = os.path.join(case_path, "examples", "Spinless_Fermion_2D_Square_nested_CTMRG", "exp_nested_ctmrg.h5")
    print("###", case)
    for label, path in (("ctmrg", square), ("nested", nested)):
        print("--", label)
        for group, values in sorted(group_values(path).items()):
            eh = np.asarray(values.get("Eh", []), dtype=float)
            ev = np.asarray(values.get("Ev", []), dtype=float)
            ns = np.asarray(values.get("ns", []), dtype=float)
            print(
                group,
                "Es", values.get("Es_avg"),
                "Eh_sum", float(eh.sum()) if eh.size else None,
                "Ev_sum", float(ev.sum()) if ev.size else None,
                "Eh_mean", float(eh.mean()) if eh.size else None,
                "Ev_mean", float(ev.mean()) if ev.size else None,
                "ns_mean", float(ns.mean()) if ns.size else values.get("ns_avg"),
            )
            if eh.size:
                print("  Eh", eh)
            if ev.size:
                print("  Ev", ev)
