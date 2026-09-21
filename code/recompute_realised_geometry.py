#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A.5: nominal branch parameter vs realised heel-strike geometry.

For the reference branch (k_hip = -0.16) this records, at each branch point,
both the nominal label used to prescribe the push-off and the geometry the
walker actually realises at heel-strike:

    q        branch / push-off parameter
    alpha_p  = arcsin(q/2)                  nominal half inter-leg angle
    P        = 1.04 alpha_p tan(alpha_p)    prescribed push-off
    alpha_h  = theta^+ = -theta^-           REALISED half inter-leg angle
    s_real   = 2 sin(alpha_h)               REALISED step length
    v_real   = s_real / T                   REALISED dimensionless speed

Writes realised_geometry.csv and prints every paper-quoted number on both bases.
Python is the independent cross-check implementation; MATLAB remains the
primary pipeline and must regenerate these before the numbers enter the paper.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # newest/code
REPO = os.path.dirname(HERE)                                # newest
sys.path.insert(0, HERE)
os.chdir(REPO)
import revision_numerics as rn  # noqa: E402

KHIP = -0.16

# Dense where the scaling fit and the transition live; coarser on the plateau.
grid = [round(x, 4) for x in np.arange(0.010, 0.1501, 0.001)]
grid += [round(x, 4) for x in np.arange(0.16, 0.8001, 0.01)]
grid = sorted(set(grid))

# Continue outward from q = 0.080, which is the natural seed in solve_gait.
start = min(grid, key=lambda q: abs(q - 0.080))
down = [q for q in grid if q <= start][::-1]
up = [q for q in grid if q > start]

rows = {}


def sweep(seq, seed):
    first_seed = None
    for q in seq:
        d = rn.solve_gait(q, k=KHIP, seed=seed)
        if d is None:
            print("  収束せず q=%.4f" % q)
            seed = None
            continue
        seed = d["z_fp"]
        if first_seed is None:
            first_seed = seed
        alpha_h = float(d["z_fp"][0])
        s_real = 2.0 * np.sin(alpha_h)
        lam = d["lam"]
        nz = [x for x in lam if abs(x) > 1e-8]
        is_complex = any(abs(np.imag(x)) > 1e-10 for x in nz)
        rows[q] = dict(
            q=q, alpha_p=d["alpha_p"], P=d["P"], alpha_h=alpha_h,
            s_real=s_real, T=d["T"], v_nom=q / d["T"], v_real=s_real / d["T"],
            lam_max=d["lam_max"], margin=1.0 - d["lam_max"],
            N_half=-np.log(2.0) / np.log(d["lam_max"]),
            eig_type="complex" if is_complex else "real",
        )
    return first_seed


start_seed = sweep(down, None)
sweep(up, start_seed)

R = [rows[q] for q in sorted(rows)]
out = os.path.join(REPO, "data", "realised_geometry.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(R[0].keys()))
    w.writeheader()
    w.writerows(R)
print("書き出し: %s  (%d 点)" % (out, len(R)))


def fit(xs, ys):
    """least squares log-log with intercept -> (exponent, coefficient)"""
    lx, ly = np.log(np.array(xs)), np.log(np.array(ys))
    n = len(lx)
    b = (n * (lx * ly).sum() - lx.sum() * ly.sum()) / (n * (lx ** 2).sum() - lx.sum() ** 2)
    a = (ly.sum() - b * lx.sum()) / n
    return b, np.exp(a)


print()
print("=" * 74)
print("  実現半脚間角と名目角の比")
for q in (0.010, 0.050, 0.100, 0.200, 0.400, 0.800):
    r = rows.get(round(q, 4))
    if r:
        print("    q=%.3f  alpha_h/alpha_p=%.6f   s_real/q=%.6f   v_real/v_nom=%.6f"
              % (q, r["alpha_h"] / r["alpha_p"], r["s_real"] / q, r["v_real"] / r["v_nom"]))

print()
print("  短歩幅スケーリング（名目パラメータと実現歩幅）")
for label, key, cut in (("名目 q", "q", 0.05), ("実現 s_real", "s_real", 0.05)):
    sel = [r for r in R if r[key] <= cut + 1e-12 and r["q"] >= 0.0099]
    e, c = fit([r[key] for r in sel], [r["margin"] for r in sel])
    print("    %-12s n=%2d  指数=%.4f  係数=%.4f" % (label, len(sel), e, c))

print()
print("  スロー端の局所比")
r = rows[0.010]
print("    (1-|lam|)/q^2      = %.4f" % (r["margin"] / r["q"] ** 2))
print("    (1-|lam|)/s_real^2 = %.4f" % (r["margin"] / r["s_real"] ** 2))

print()
print("  衝突 deficit が恒等式になるか")
for q in (0.010, 0.100):
    r = rows[round(q, 4)]
    d_h = 1.0 - np.cos(2.0 * r["alpha_h"])
    d_p = 1.0 - np.cos(2.0 * r["alpha_p"])
    print("    q=%.3f: 1-cos(2a_h)=%.10e  s_real^2/2=%.10e  差=%.2e   [実現角]"
          % (q, d_h, r["s_real"] ** 2 / 2, d_h - r["s_real"] ** 2 / 2))
    print("            1-cos(2a_p)=%.10e  q^2/2     =%.10e  差=%.2e   [名目角]"
          % (d_p, q ** 2 / 2, d_p - q ** 2 / 2))

print()
print("  転移(complex -> real)の位置")
prev = None
for r in R:
    if prev and prev["eig_type"] != r["eig_type"]:
        print("    q     : [%.4f, %.4f]" % (prev["q"], r["q"]))
        print("    v_nom : [%.6f, %.6f]" % (prev["v_nom"], r["v_nom"]))
        print("    v_real: [%.6f, %.6f]" % (prev["v_real"], r["v_real"]))
    prev = r

print()
print("  Table 1 に相当する代表点")
print("    %-8s %-10s %-10s %-10s %-10s %-12s %-9s" %
      ("q", "s_real", "v_nom", "v_real", "T", "lam_max", "N_half"))
for q in (0.010, 0.080, 0.100, 0.200, 0.400, 0.800):
    r = rows.get(round(q, 4))
    if r:
        print("    %-8.3f %-10.6f %-10.6f %-10.6f %-10.6f %-12.8f %-9.1f"
              % (q, r["s_real"], r["v_nom"], r["v_real"], r["T"], r["lam_max"], r["N_half"]))
