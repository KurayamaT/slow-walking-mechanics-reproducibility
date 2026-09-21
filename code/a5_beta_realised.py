#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A.5 step: finite foot mass at a common REALISED step length.

Supplementary S4 compares four foot masses at "the same s = 0.040", but s there
is the branch label q = 2 sin(alpha_p), not the realised step length. This ports
the finite-beta model from legacy/code/floquet_KUO_v2_beta.m to Python so
the realised heel-strike angle alpha_h can be recorded, validates the port
against the stored MATLAB branches, and re-makes the comparison at a common
realised step length.

Python is the independent cross-check implementation; MATLAB remains primary.
"""
import csv
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)

GAM, KHIP = 0.0, -0.16
RTOL, ATOL, PER, DT = 1e-12, 1e-14, 5.0, 0.005
BETAS = [0.00, 0.05, 0.10, 0.15]


def eom(t, y, beta):
    th, thd, ph, phd = y
    if beta == 0.0:
        return [thd, np.sin(th - GAM), phd,
                np.sin(th - GAM) + np.sin(ph) * (thd ** 2 - np.cos(th - GAM)) + KHIP * ph]
    cph, sph = np.cos(ph), np.sin(ph)
    s1g = np.sin(th - GAM)
    s12 = np.sin(GAM + ph - th)
    M11 = 1 + 2 * beta * (1 - cph)
    M12 = -beta * (1 - cph)
    M21 = 1 - cph
    M22 = -1.0
    rhs1 = -beta * sph * phd * (2 * thd - phd) + (beta * s12 + s1g * (1 + beta))
    rhs2 = -sph * thd ** 2 + s12 - KHIP * ph
    det = M11 * M22 - M12 * M21
    return [thd, (M22 * rhs1 - M12 * rhs2) / det, phd, (M11 * rhs2 - M21 * rhs1) / det]


def _coll(t, y, beta):
    return y[2] - 2 * y[0]
_coll.terminal, _coll.direction = True, 1


def _guard(t, y, beta):
    return np.pi / 2 - abs(y[0])
_guard.terminal = True


def step_map(z, beta, P):
    if abs(z[0]) > np.pi / 3:
        return None, None
    y0 = [z[0], z[1], 2 * z[0], z[2]]
    s1 = solve_ivp(eom, (0.0, DT), y0, args=(beta,), rtol=RTOL, atol=ATOL, dense_output=False)
    if not s1.success:
        return None, None
    s2 = solve_ivp(eom, (DT, PER), s1.y[:, -1], args=(beta,), rtol=RTOL, atol=ATOL,
                   events=[_coll, _guard])
    if len(s2.t_events[0]) == 0:
        return None, None
    T = float(s2.t_events[0][0])
    yc = s2.y_events[0][0]
    c2, sn2 = np.cos(2 * yc[0]), np.sin(2 * yc[0])
    thd_post = (c2 * yc[1] + sn2 * P) / (1.0 + beta * sn2 ** 2)
    return np.array([-yc[0], thd_post, (1 - c2) * thd_post]), T


def fixed_point(z0, beta, P):
    z = np.array(z0, float)
    for _ in range(25):
        S, T = step_map(z, beta, P)
        if S is None:
            return None, None
        res = S - z
        if np.linalg.norm(res) < 1e-12:
            return z, T
        Jg = np.zeros((3, 3))
        for j in range(3):
            zp, zm = z.copy(), z.copy()
            zp[j] += 1e-7; zm[j] -= 1e-7
            Sp, _ = step_map(zp, beta, P)
            Sm, _ = step_map(zm, beta, P)
            if Sp is None or Sm is None:
                return None, None
            Jg[:, j] = (Sp - Sm) / 2e-7
        Jg -= np.eye(3)
        try:
            z = z + 0.8 * np.linalg.solve(Jg, -res)
        except np.linalg.LinAlgError:
            return None, None
    return None, None


def multipliers(z, beta, P):
    J = np.zeros((3, 3))
    for j in range(3):
        zp, zm = np.array(z), np.array(z)
        zp[j] += 1e-6; zm[j] -= 1e-6
        Sp, _ = step_map(zp, beta, P)
        Sm, _ = step_map(zm, beta, P)
        if Sp is None or Sm is None:
            return None
        J[:, j] = (Sp - Sm) / 2e-6
    return float(np.max(np.abs(np.linalg.eigvals(J))))


def gait(q, beta, seed=None, beta_step=0.01):
    """Solve the periodic gait at (q, beta).

    For beta > 0 the geometric guess does not converge, so beta is marched from
    zero in steps of beta_step, exactly as floquet_KUO_v2_beta.m does.
    """
    a = np.arcsin(0.5 * q)
    om = -1.04 * a
    P = -om * np.tan(a)
    z0 = seed if seed is not None else [a, om, (1 - np.cos(2 * a)) * om]
    if beta > 0 and seed is None:
        n = max(1, int(np.ceil(beta / beta_step)))
        z_run = z0
        for b in np.linspace(0.0, beta, n + 1):
            z_run, _ = fixed_point(z_run, float(b), P)
            if z_run is None:
                return None
        z0 = z_run
    z, T = fixed_point(z0, beta, P)
    if z is None:
        return None
    lam = multipliers(z, beta, P)
    if lam is None or not (0 < lam < 1):
        return dict(q=q, beta=beta, z=z, T=T, lam_max=lam, alpha_h=float(z[0]),
                    s_real=2 * np.sin(z[0]), N_half=float('nan'))
    return dict(q=q, beta=beta, z=z, T=T, lam_max=lam, alpha_h=float(z[0]),
                s_real=2 * np.sin(z[0]), N_half=-np.log(2) / np.log(lam))


# ---- 1. validate the port against the stored MATLAB branches ----
print("=== 移植の検証(MATLAB の β 枝との一致)===")
for b in BETAS:
    f = "legacy/results/csv/floquet_beta%.3f_k-0.160.csv" % b
    p = os.path.join(REPO, f)
    if not os.path.exists(p):
        print("  %s が無い" % f); continue
    rows = list(csv.DictReader(open(p)))
    worst, n = 0.0, 0
    for r in rows[:6]:
        a = float(r["alpha"]); q = 2 * np.sin(a)
        g = gait(q, b)
        if g is None or g["lam_max"] is None:
            continue
        d = abs(g["lam_max"] - float(r["lam_max"]))
        worst = max(worst, d); n += 1
    print("  beta=%.2f  %d 点比較  |lam_max| 最大差 = %.2e" % (b, n, worst))

# ---- 2. realised geometry along each beta branch ----
QS = [round(x, 4) for x in np.arange(0.040, 0.0901, 0.004)]
print("\n=== 各 β 枝の実現歩幅 ===")
B = {}
for b in BETAS:
    seed, br = None, {}
    for q in sorted(QS, reverse=True):
        g = gait(q, b, seed)
        if g is None:
            seed = None; continue
        seed = g["z"]; br[q] = g
    B[b] = br
    r = br.get(0.040)
    if r:
        print("  beta=%.2f  q=0.040 -> alpha_h/alpha_p=%.6f  s_real=%.6f  N_half=%.2f"
              % (b, r["alpha_h"] / np.arcsin(0.02), r["s_real"], r["N_half"]))

ok = [b for b in BETAS if 0.040 in B[b]]
if len(ok) == len(BETAS):
    S = round(max(min(r["s_real"] for r in B[b].values()) for b in ok) + 1e-6, 6)
    print("\n=== 共通実現歩幅 s=%.6f での比較(線形内挿、外挿なし)===" % S)
    print("  %-8s %-18s %-18s %-8s" % ("beta", "N_half(q=0.040)", "N_half(s共通)", "差"))
    for b in ok:
        vs = sorted((r["s_real"], r["N_half"]) for r in B[b].values() if r["N_half"] == r["N_half"])
        xs = [v[0] for v in vs]; ys = [v[1] for v in vs]
        nq = B[b][0.040]["N_half"]
        nc = float(np.interp(S, xs, ys))
        print("  %-8.2f %-18.2f %-18.2f %+.2f" % (b, nq, nc, nc - nq))

with open("data/a5_beta_realised.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["beta", "q", "alpha_h", "s_real", "T", "lam_max", "N_half"])
    for b in BETAS:
        for q in sorted(B.get(b, {})):
            r = B[b][q]
            w.writerow([b, q, r["alpha_h"], r["s_real"], r["T"], r["lam_max"], r["N_half"]])
print("\n書き出し: data/a5_beta_realised.csv")
