import json
import subprocess
import sys


raw = subprocess.check_output(
    [sys.executable, "tmp\\0817_results_audit\\analyze_results.py"],
    text=True,
)
rows = json.loads(raw)
print("| case | exact | SU | CTMRG chi16/32/48 | nested chi16/32/48 | ns CTMRG48/nested48 | same tensor |")
print("|---|---:|---:|---|---|---|---|")
for row in rows:
    ctmrg = [value for key, value in sorted(row["ctmrg_h5"].items()) if key.endswith("/Es_avg")]
    nested = [value for key, value in sorted(row["nested_h5"].items()) if key.endswith("/Es_avg")]
    ns_ctmrg = [value for key, value in sorted(row["ctmrg_h5"].items()) if key.endswith("/ns_avg")]
    ns_nested = [value for key, value in sorted(row["nested_h5"].items()) if key.endswith("/ns_avg")]
    def fmt(values):
        return ", ".join(f"{value:.9f}" for value in values) if values else "missing"
    ns = "missing"
    if ns_ctmrg and ns_nested:
        ns = f"{ns_ctmrg[-1]:.9f}/{ns_nested[-1]:.9f}"
    print(
        "| {case} | {exact:.9f} | {su} | {ctmrg} | {nested} | {ns} | {same} |".format(
            case=row["case"],
            exact=row["exact"],
            su="missing" if row["su"] is None else f"{row['su']:.9f}",
            ctmrg=fmt(ctmrg),
            nested=fmt(nested),
            ns=ns,
            same="yes" if row["tensor_hash_equal"] else "no",
        )
    )
