import pathlib
import re

import h5py
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(
    r"D:\C 盘备份\Back up\Work_Save\coding\codex_server_logs\2026\0828\test-hubbard-half-filled-d6-detailed-table.md"
)
BENCH = -1.1760


def fmt(x, n=6):
    return f"{x:.{n}f}"


rows = []
for fn in sorted(ROOT.glob("Fermi_Hubbard_2D_Square_su_1e-2*/exp*.h5")):
    kind = "nested" if "nested" in str(fn.parent) else "ctmrg"
    m = re.search(r"_(\d+)$", fn.parent.name)
    rep = int(m.group(1)) if m else 0
    with h5py.File(fn, "r") as f:
        for key in f:
            chi = int(re.search(r"\d+", key).group())
            g = f[key]
            eraw = float(np.array(g["Es_avg"]))
            navg = float(np.array(g["ns_avg"]))
            eh = np.array(g["Eh"])
            ev = np.array(g["Ev"])
            ecanon = eraw + navg
            rows.append(
                {
                    "kind": kind,
                    "rep": rep,
                    "chi": chi,
                    "Eraw": eraw,
                    "n": navg,
                    "Ecanon": ecanon,
                    "diff": ecanon - BENCH,
                    "Eh": float(eh.mean()),
                    "Ev": float(ev.mean()),
                    "Ehr": float(eh.max() - eh.min()),
                    "Evr": float(ev.max() - ev.min()),
                }
            )

lines = [
    "# Hubbard half-filled D=6 detailed result table",
    "",
    "Source: `/gpfs/home/jgkong/work/2026/0827/test_Hubbard_half-filled-D6/examples`",
    "",
    "All listed expectation-value jobs report `U=2.0, mu=1.0`. Benchmark comparison uses `E_canonical = E_raw + mu*n_avg = E_raw + n_avg`; benchmark reference value used here is `E_bench=-1.1760`.",
    "",
    "## All CTMRG and nested CTMRG points",
    "",
    "| method | run | chi | E_raw | n_avg | E_canonical | Delta vs bench | Eh_mean | Ev_mean | Eh_range | Ev_range | flag |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]

for r in sorted(rows, key=lambda x: (x["kind"], x["rep"], x["chi"])):
    flags = []
    if abs(r["n"] - 1) > 0.002:
        flags.append("n off")
    if r["Ehr"] > 0.08 or r["Evr"] > 0.08:
        flags.append("bond spread")
    if abs(r["diff"]) > 0.20:
        flags.append("outlier")
    flag = ", ".join(flags) if flags else "ok-ish"
    lines.append(
        f"| {r['kind']} | {r['rep']} | {r['chi']} | {fmt(r['Eraw'])} | {fmt(r['n'])} | {fmt(r['Ecanon'])} | {fmt(r['diff'])} | {fmt(r['Eh'])} | {fmt(r['Ev'])} | {fmt(r['Ehr'])} | {fmt(r['Evr'])} | {flag} |"
    )

lines.extend(
    [
        "",
        "## Summary by method and chi",
        "",
        "| method | chi | N | mean E_canonical | median E_canonical | std | min | max | mean n_avg | good count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
)

for kind in ["ctmrg", "nested"]:
    for chi in sorted({r["chi"] for r in rows if r["kind"] == kind}):
        vals = [r for r in rows if r["kind"] == kind and r["chi"] == chi]
        ec = np.array([r["Ecanon"] for r in vals])
        navg = np.array([r["n"] for r in vals])
        good = [
            r
            for r in vals
            if abs(r["n"] - 1) < 0.002 and r["Ehr"] < 0.08 and r["Evr"] < 0.08
        ]
        lines.append(
            f"| {kind} | {chi} | {len(vals)} | {fmt(float(ec.mean()))} | {fmt(float(np.median(ec)))} | {fmt(float(ec.std(ddof=0)))} | {fmt(float(ec.min()))} | {fmt(float(ec.max()))} | {fmt(float(navg.mean()))} | {len(good)} |"
        )

lines.extend(
    [
        "",
        "## CTMRG vs nested paired differences",
        "",
        "| rep | chi | Ecanon CTMRG | Ecanon nested | nested - CTMRG |",
        "|---:|---:|---:|---:|---:|",
    ]
)

by_key = {(r["kind"], r["rep"], r["chi"]): r for r in rows}
for rep in range(1, 7):
    for chi in [16, 32, 48, 64]:
        c = by_key.get(("ctmrg", rep, chi))
        n = by_key.get(("nested", rep, chi))
        if c and n:
            lines.append(
                f"| {rep} | {chi} | {fmt(c['Ecanon'])} | {fmt(n['Ecanon'])} | {fmt(n['Ecanon'] - c['Ecanon'])} |"
            )

lines.extend(
    [
        "",
        "## SU checkpoint",
        "",
        "| checkpoint | E_raw | err1 | err2 | note |",
        "|---|---:|---:|---:|---|",
    ]
)

su_file = ROOT / "Fermi_Hubbard_2D_Square_su_1e-2" / "su.h5"
if su_file.exists():
    with h5py.File(su_file, "r") as f:
        key = "iter5853_δτ0.01"
        g = f[key]
        lines.append(
            f"| {key} | {float(np.array(g['energy'])):.12f} | {float(np.array(g['err1'])):.3e} | {float(np.array(g['err2'])):.3e} | SU local estimate, approximately E+1 at half filling |"
        )
else:
    lines.append("| unavailable | | | | |")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT)
