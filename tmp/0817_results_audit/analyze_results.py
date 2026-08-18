import glob
import hashlib
import json
import math
import os
import re

import h5py


ROOT = os.path.dirname(os.path.abspath(__file__))


def exact(t, gamma, lam, nk=512):
    delta = 2 * math.pi / nk
    total = 0.0
    for ix in range(nk):
        kx = -math.pi + (ix + 0.5) * delta
        cosx = math.cos(kx)
        sinx = math.sin(kx)
        for iy in range(nk):
            ky = -math.pi + (iy + 0.5) * delta
            normal = t * (cosx + math.cos(ky)) - lam
            pairing = gamma * (sinx + math.sin(ky))
            total += math.hypot(normal, pairing)
    return -lam - total / (nk * nk)


def read_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except FileNotFoundError:
        return ""


def parse_params(path):
    text = read_text(path)
    out = {}
    patterns = [
        ("t", r"^t\s*=\s*([-+0-9.eE]+)", float),
        ("gamma", r"^(?:γ|gamma)\s*=\s*([-+0-9.eE]+)", float),
        ("lambda", r"^(?:λ|lambda)\s*=\s*([-+0-9.eE]+)", float),
        ("Dbond", r"^Dbond\s*=\s*(\d+)", int),
    ]
    for key, pattern, caster in patterns:
        match = re.search(pattern, text, re.M)
        if match:
            out[key] = caster(match.group(1))
    match = re.search(r"^peps_param_str\s*=\s*(.+)$", text, re.M)
    if match:
        out["peps_param_str"] = match.group(1).strip()
    return out


def sha16(path):
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def last_su_energy(path):
    values = []
    for line in read_text(path).splitlines():
        match = re.search(r"Estimated ground state energy per site:\s*([-+0-9.eE]+)", line)
        if match:
            values.append(float(match.group(1)))
    return values[-1] if values else None


def ctmrg_energies(path):
    values = []
    for line in read_text(path).splitlines():
        match = re.search(r"Average ground energy per site:\s*([-+0-9.eE]+)", line)
        if match:
            values.append(float(match.group(1)))
        match = re.search(
            r"chi=(\d+) nested energy/site\s+([-+0-9.eE]+) density\s+([-+0-9.eE]+)",
            line,
        )
        if match:
            values.append(
                {
                    "chi": int(match.group(1)),
                    "energy": float(match.group(2)),
                    "density": float(match.group(3)),
                }
            )
    return values


def h5_scalars(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with h5py.File(path, "r") as h5:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith(("/Es_avg", "/ns_avg")):
                try:
                    out[name] = float(obj[()])
                except Exception as exc:
                    out[name] = "ERR " + repr(exc)

        h5.visititems(visit)
    return out


def h5_top_keys(path):
    if not os.path.exists(path):
        return []
    with h5py.File(path, "r") as h5:
        return sorted(list(h5.keys()))


rows = []
for case_path in sorted(glob.glob(os.path.join(ROOT, "testfreefermion_*"))):
    case = os.path.basename(case_path)
    square = os.path.join(case_path, "examples", "Spinless_Fermion_2D_Square")
    nested = os.path.join(case_path, "examples", "Spinless_Fermion_2D_Square_nested_CTMRG")
    params_su = parse_params(os.path.join(square, "Spinless_Fermion_2D_Simple_Update.jl"))
    params_ctmrg = parse_params(os.path.join(square, "Spinless_Fermion_2D_CTMRG.jl"))
    params_nested_su = parse_params(os.path.join(nested, "Spinless_Fermion_2D_Simple_Update.jl"))
    params_nested_ctmrg = parse_params(os.path.join(nested, "Spinless_Fermion_2D_nested_CTMRG.jl"))
    ex = None
    if {"t", "gamma", "lambda"} <= set(params_su):
        ex = exact(params_su["t"], params_su["gamma"], params_su["lambda"])
    square_tensor = os.path.join(square, "tensor_file.h5")
    nested_tensor = os.path.join(nested, "tensor_file.h5")
    square_su = os.path.join(square, "su.h5")
    nested_su = os.path.join(nested, "su.h5")
    rows.append(
        {
            "case": case,
            "params_su": params_su,
            "params_ctmrg": params_ctmrg,
            "params_nested_su": params_nested_su,
            "params_nested_ctmrg": params_nested_ctmrg,
            "exact": ex,
            "su": last_su_energy(os.path.join(square, "nohup.out")),
            "nested_su": last_su_energy(os.path.join(nested, "nohup.out")),
            "ctmrg_log": ctmrg_energies(os.path.join(square, "nohup_ctmrg.out")),
            "nested_log": ctmrg_energies(os.path.join(nested, "nohup_ctmrg.out")),
            "ctmrg_h5": h5_scalars(os.path.join(square, "exp_ctmrg.h5")),
            "nested_h5": h5_scalars(os.path.join(nested, "exp_nested_ctmrg.h5")),
            "tensor_hash_equal": sha16(square_tensor) == sha16(nested_tensor),
            "su_hash_equal": sha16(square_su) == sha16(nested_su),
            "tensor_keys_square": h5_top_keys(square_tensor),
            "tensor_keys_nested": h5_top_keys(nested_tensor),
            "hashes": {
                "square_tensor": sha16(square_tensor),
                "nested_tensor": sha16(nested_tensor),
                "square_su": sha16(square_su),
                "nested_su": sha16(nested_su),
            },
        }
    )

print(json.dumps(rows, ensure_ascii=False, indent=2))
