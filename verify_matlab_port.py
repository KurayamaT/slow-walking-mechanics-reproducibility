#!/usr/bin/env python3
"""
verify_matlab_port.py — compare every MATLAB output against its Python original.

Run it yourself:   python3 verify_matlab_port.py

It reads data_python/*.csv when that directory exists (otherwise data/*.csv) as the
Python side and data_matlab/*.csv as the MATLAB side, and for every file
present in both prints, per column, the largest absolute difference between the
two. String columns are compared exactly and the number of differing rows is
printed. Nothing is written and nothing is interpreted: the numbers are the
report. A file missing on either side is listed as such.

The last section re-derives, from the MATLAB CSVs alone, each number the
manuscript or the Supplementary quotes, and prints it beside the quoted value so
the two can be read off against each other.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_PYDIR = "data_python" if os.path.isdir(os.path.join(HERE, "data_python")) else "data"
PY = os.path.join(HERE, _PYDIR)
ML = os.path.join(HERE, "data_matlab")

PAIRS = [
    "master_fixed_khip_extended", "fig2b_traces", "lowspeed_scaling",
    "transition_spectrum", "slowend_modal", "slowend_verification",
    "supp_csweep", "supp_khip_sweep",
    "supp_validation", "supp_validation_branch", "supp_validation_delta",
]

# Outputs with no Python producer: these come from MATLAB alone, so there is
# nothing independent to compare them against and they are not in PAIRS.
MATLAB_ONLY = ["sens_khip_summary", "sens_beta_summary"]


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def isnum(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def compare(name):
    p, m = os.path.join(PY, name + ".csv"), os.path.join(ML, name + ".csv")
    if not os.path.exists(p) or not os.path.exists(m):
        print(f"\n=== {name} ===")
        print(f"  python: {'present' if os.path.exists(p) else 'MISSING'}   "
              f"matlab: {'present' if os.path.exists(m) else 'MISSING'}")
        return
    A, B = read(p), read(m)
    print(f"\n=== {name} ===   rows: python {len(A)}, matlab {len(B)}")
    if len(A) != len(B):
        print("  ROW COUNT DIFFERS — columns not compared")
        return
    if not A:
        return
    for col in A[0]:
        if col not in B[0]:
            print(f"  {col:<22} column absent in matlab")
            continue
        pairs = [(a[col], b[col]) for a, b in zip(A, B)]
        numeric = [(x, y) for x, y in pairs if isnum(x) and isnum(y)]
        if len(numeric) == len(pairs) and numeric:
            d = max(abs(float(x) - float(y)) for x, y in numeric)
            print(f"  {col:<22} max |python - matlab| = {d:.2e}")
        else:
            n = sum(1 for x, y in pairs if x != y)
            print(f"  {col:<22} string column, {n} of {len(pairs)} rows differ")


def matlab_only_note():
    print("\n\n########## MATLAB-only outputs (no Python counterpart) ##########")
    for name in MATLAB_ONLY:
        here = os.path.exists(os.path.join(ML, name + ".csv"))
        print(f"  {name:<24} {'present in data_matlab/' if here else 'MISSING'}"
              f"   — produced by MATLAB only, so not cross-checked")


def quoted_numbers():
    """Re-derive from the MATLAB CSVs the numbers the paper quotes."""
    print("\n\n########## numbers the paper quotes, recomputed from data_matlab/ ##########")
    out = []

    br = read(os.path.join(ML, "master_fixed_khip_extended.csv"))
    # Key the lookups on q, the exact grid value. Keying on the realised s
# would quietly select a neighbouring gait: at q = 0.010 the realised s
# is 0.009881, so abs(s - 0.010) is minimised one row away.
    row = min(br, key=lambda r: abs(float(r["q"]) - 0.010))
    out += [("Table 1 / 4.6 / S3: N_1/2 at s=0.010", "842.2", row["N_half"]),
            ("Table 1: |lambda_max| at s=0.010", "0.99917734", row["lam_max"])]
    plateau = min(br, key=lambda r: abs(float(r["q"]) - 0.402))
    out.append(("4.3: |lambda_max| at the plateau gait s=0.402", "0.7073", plateau["lam_max"]))
    low = min(br, key=lambda r: abs(float(r["q"]) - 0.080))
    out.append(("4.3: |lambda_max| at s=0.080", "0.9439", low["lam_max"]))
    marg = max(1.0 - float(r["lam_max"]) for r in br)
    out.append(("5.1 / S3: largest recovery margin on the branch", "0.355", f"{marg:.4f}"))

    sc = read(os.path.join(ML, "lowspeed_scaling.csv"))
    fit = [r for r in sc if abs(float(r["s_max"]) - 0.05) < 1e-9][0]
    out += [("4.2 / Figure 4: scaling exponent over s<=0.05", "2.013", f"{float(fit['exponent']):.4f}"),
            ("4.2 / Figure 4: coefficient over s<=0.05", "8.92", f"{float(fit['coefficient']):.4f}"),
            ("S5: n over s<=0.05", "41", fit["n"])]

    ts = read(os.path.join(ML, "transition_spectrum.csv"))
    f = lambda r, k: float(r[k])
    real = [r for r in ts if abs(f(r, "l1_im")) < 1e-12]
    cx = [r for r in ts if abs(f(r, "l1_im")) >= 1e-12]
    lo, hi = f(real[-1], "v"), f(cx[0], "v")
    dets = [f(r, "det") for r in ts]
    out += [("3.3 / 4.4 / S2: transition bracket, lower end", "0.040266", f"{lo:.6f}"),
            ("3.3 / 4.4 / S2: transition bracket, upper end", "0.040380", f"{hi:.6f}"),
            ("S2: bracket width", "1.16e-04", f"{hi - lo:.2e}"),
            ("S2: det A variation over the grid", "0.18 %", f"{(max(dets) - min(dets)) / min(dets) * 100:.2f} %"),
            ("S2: emergent imaginary part", "0.012400", f"{abs(f(cx[0], 'l1_im')):.6f}")]

    sv = read(os.path.join(ML, "slowend_verification.csv"))
    nh = [float(r["N_hat"]) for r in sv]
    out += [("S3: modal half-life range over the six runs", "837-863", f"{min(nh):.1f}-{max(nh):.1f}"),
            ("S3: linear prediction", "842.2", f"{float(sv[0]['N_linear']):.1f}"),
            ("S3: modal drift floor", "3.325e-09", f"{float(sv[0]['modal_floor']):.3e}"),
            ("S3: one-step residual", "2.31e-13", f"{float(sv[0]['one_step_residual']):.2e}")]

    ks = read(os.path.join(ML, "sens_khip_summary.csv"))
    by = {float(r["k_hip"]): r for r in ks}
    out += [("S4: N_1/2 at q=0.010, k_hip=-0.30", "1078.2", f"{float(by[-0.30]['Nhalf_at_qmin']):.1f}"),
            ("S4: N_1/2 at q=0.010, k_hip=-0.16", "842.2", f"{float(by[-0.16]['Nhalf_at_qmin']):.1f}"),
            ("S4: N_1/2 at q=0.010, k_hip=+0.10", "443.6", f"{float(by[0.10]['Nhalf_at_qmin']):.1f}"),
            ("S4: N_1/2 at the common realised s, k_hip=-0.30", "1033.2",
             f"{float(by[-0.30]['Nhalf_at_common_s']):.1f}"),
            ("S4: N_1/2 at the common realised s, k_hip=+0.10", "443.6",
             f"{float(by[0.10]['Nhalf_at_common_s']):.1f}"),
            ("S4: common realised step length", "0.010029",
             f"{float(by[-0.16]['s_common']):.6f}"),
            ("S4: v* at k_hip=-0.30", "0.0469", f"{float(by[-0.30]['v_eigtype_transition']):.4f}"),
            ("S4: v* at k_hip=+0.10", "0.0271", f"{float(by[0.10]['v_eigtype_transition']):.4f}")]

    cs = read(os.path.join(ML, "supp_csweep.csv"))
    for c, q in ((0.80, "1095.2"), (1.00, "876.0"), (1.04, "842.2"), (1.20, "729.8")):
        rows = [r for r in cs if abs(float(r["c"]) - c) < 1e-9]
        slowest = min(rows, key=lambda r: float(r["v"]))
        out.append((f"S4: N_1/2 at the slowest gait, c={c:.2f}", q, f"{float(slowest['N_half']):.1f}"))

    va = read(os.path.join(ML, "supp_validation.csv"))
    out.append(("S1: N_1/2 at s=0.010 with delta=1e-5", "718.7",
                f"{float(va[0]['N_half_delta1e5']):.1f}"))

    vb = read(os.path.join(ML, "supp_validation_branch.csv"))
    out += [("S1: |lambda_3| bound along the branch", "4.3e-10",
             f"{max(abs(float(r['lam3_abs'])) for r in vb):.2e}"),
            ("S1: max |lambda^3D - lambda^2D| on the branch", "6.3e-08",
             f"{max(abs(float(r['dev_2d_3d'])) for r in vb):.2e}")]

    print(f"{'quantity':<52} {'paper':>12} {'matlab':>14}   match")
    for label, quoted, got in out:
        try:
            # plain numbers, including scientific notation with a negative exponent
            q, g = float(quoted.rstrip(" %")), float(str(got).rstrip(" %"))
            ok = abs(q - g) <= 0.02 * abs(q)
        except ValueError:
            try:
                ok = abs(float(quoted.split("-")[0].rstrip(" %")) - float(str(got).split("-")[0].rstrip(" %"))) \
                     <= 0.02 * abs(float(quoted.split("-")[0].rstrip(" %")))
            except ValueError:
                ok = quoted.strip() == str(got).strip()
        print(f"{label:<52} {quoted:>12} {str(got):>14}   {'yes' if ok else 'CHECK'}")


if __name__ == "__main__":
    print("Python originals : " + PY)
    print("MATLAB outputs   : " + ML)
    for name in PAIRS:
        compare(name)
    matlab_only_note()
    quoted_numbers()
