#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A.5 step 3: does the same branch parameter q mean the same realised step length
across branches?

The robustness section compares nine hip-spring branches "at one step length",
using q as the label. If the realised half inter-leg angle alpha_h depends on
k_hip, the same q is NOT the same step length and those comparisons need to be
re-made at a common realised s.

For each k_hip and each push-off coefficient c, this solves the periodic gait at
a set of q values and records the realised step length s = 2 sin(alpha_h), then
interpolates each branch onto a common realised-s grid so the comparison is
like for like.

Python is the independent cross-check implementation; MATLAB remains primary.
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

KHIPS = [-0.30, -0.20, -0.16, -0.12, -0.08, -0.04, 0.00, 0.05, 0.10]
CS = [0.80, 1.00, 1.04, 1.20]
QS = [round(x, 4) for x in np.arange(0.010, 0.0601, 0.002)]   # scaling range
QS += [0.080, 0.100, 0.120, 0.130]


def solve_branch(k, c=1.04):
    """Seeded continuation downward from q=0.130 for one branch."""
    out, seed = {}, None
    for q in sorted(QS, reverse=True):
        alpha_p = np.arcsin(0.5 * q)
        omega = -c * alpha_p
        P = -omega * np.tan(alpha_p)
        z0 = list(seed) if seed is not None else [
            alpha_p, omega, (1.0 - np.cos(2.0 * alpha_p)) * omega]
        z, T, ok = rn.find_fixed_point(z0, rn.GAM, k, P, rn.RTOL, rn.ATOL, delta=1e-7)
        if not ok:
            seed = None
            continue
        seed = z
        J = rn.jacobian_3d(z, rn.GAM, k, P, rn.RTOL, rn.ATOL, rn.DELTA)
        if J is None:
            continue
        lam = float(np.max(np.abs(rn.sorted_eigs(J))))
        a_h = float(z[0])
        out[q] = dict(q=q, k_hip=k, c=c, alpha_p=float(alpha_p), alpha_h=a_h,
                      s_real=2.0 * np.sin(a_h), T=T, lam_max=lam,
                      margin=1.0 - lam, N_half=-np.log(2.0) / np.log(lam))
    return out


def fit(xs, ys):
    lx, ly = np.log(np.asarray(xs)), np.log(np.asarray(ys))
    n = len(lx)
    b = (n * (lx * ly).sum() - lx.sum() * ly.sum()) / (n * (lx ** 2).sum() - lx.sum() ** 2)
    return b, np.exp((ly.sum() - b * lx.sum()) / n)


rows = []
print("=== hip-spring 枝 ===")
banks = {}
for k in KHIPS:
    b = solve_branch(k)
    banks[('k', k)] = b
    rows += list(b.values())
    r = b.get(0.010)
    if r:
        print("  k_hip=%+.2f  q=0.010 -> s_real=%.6f  (alpha_h/alpha_p=%.6f)  N_half=%.1f"
              % (k, r['s_real'], r['alpha_h'] / r['alpha_p'], r['N_half']))

print("\n=== push-off 係数 枝 (k_hip=-0.16) ===")
for c in CS:
    b = solve_branch(-0.16, c=c)
    banks[('c', c)] = b
    rows += list(b.values())
    r = b.get(0.010)
    if r:
        print("  c=%.2f  q=0.010 -> s_real=%.6f  N_half=%.1f" % (c, r['s_real'], r['N_half']))

out = os.path.join(REPO, "data", "a5_branch_realised.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\n書き出し: %s (%d 行)" % (out, len(rows)))

# ---- どれだけずれるか、そして共通 s での比較 ----
print("\n=== 同じ q が同じ実現歩幅を意味するか(q=0.010)===")
ss = [banks[('k', k)][0.010]['s_real'] for k in KHIPS if 0.010 in banks[('k', k)]]
print("  実現 s の範囲: %.6f – %.6f   相対差 %.3f%%"
      % (min(ss), max(ss), 100 * (max(ss) - min(ss)) / min(ss)))

print("\n=== 共通の実現 s=0.00988 に内挿した半減期(現行は q=0.010 で比較)===")
S_COMMON = 0.00988
print("  %-10s %-12s %-12s %-12s" % ("k_hip", "N_half(q基準)", "N_half(s共通)", "差"))
for k in KHIPS:
    b = banks[('k', k)]
    qs = sorted(b)
    xs = [b[q]['s_real'] for q in qs]
    ys = [b[q]['N_half'] for q in qs]
    if min(xs) <= S_COMMON <= max(xs):
        n_common = float(np.interp(S_COMMON, xs, ys))
        n_q = b[0.010]['N_half'] if 0.010 in b else float('nan')
        print("  %-10.2f %-12.1f %-12.1f %+.1f" % (k, n_q, n_common, n_common - n_q))

print("\n=== 短歩幅スケーリング指数と係数(名目 q 基準 vs 実現 s 基準)===")
print("  %-10s %-16s %-16s" % ("k_hip", "名目 (指数,係数)", "実現 (指数,係数)"))
for k in KHIPS:
    b = banks[('k', k)]
    sel = [b[q] for q in sorted(b) if q <= 0.050 + 1e-9]
    if len(sel) < 5:
        continue
    e1, c1 = fit([r['q'] for r in sel], [r['margin'] for r in sel])
    e2, c2 = fit([r['s_real'] for r in sel], [r['margin'] for r in sel])
    print("  %-10.2f (%.4f, %.4f)  (%.4f, %.4f)" % (k, e1, c1, e2, c2))
