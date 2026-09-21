#!/usr/bin/env python3
"""
Independent Python implementation of the swing/reset decomposition of the
return-map Jacobian (MATLAB primary: code_matlab/run_jacobian_decomposition.m and
run_freeze_test.m). Cross-checks the two MATLAB CSVs column by column.

  J = B A,  A = d(flow to heel-strike)/dz (numerical),  B = d(reset)/dy (analytic)
  d lambda/dq = l' (dB/dq A + B dA/dq) r / (l' r)   for the simple dominant eigenvalue

Run from code/:  python3 jacobian_decomposition_check.py
Nothing is written except a report to stdout.
"""
import os
import csv
import numpy as np
from numpy.linalg import eig, norm
from scipy.integrate import solve_ivp, trapezoid
import revision_numerics as R

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
C = 1.04
H = 0.001


def flow_to_event(z, gam=R.GAM, k=R.KHIP, rtol=R.RTOL, atol=R.ATOL):
    """Post-impact state -> pre-impact state [theta-, theta_dot-, phi_dot-] at heel-strike."""
    if abs(z[0]) > np.pi / 3.0:
        return None, False
    y0 = [z[0], z[1], 2.0 * z[0], z[2]]
    s1 = solve_ivp(R.eom, (0.0, R.DT_DEPART), y0, args=(gam, k), method="RK45",
                   rtol=1e-12, atol=1e-14)
    s2 = solve_ivp(R.eom, (R.DT_DEPART, R.PER), s1.y[:, -1], args=(gam, k), method="RK45",
                   rtol=rtol, atol=atol, events=(R._ev_collision, R._ev_guard))
    if s2.t_events[0].size == 0:
        return None, False
    yc = s2.y_events[0][0]
    return np.array([yc[0], yc[1], yc[3]]), True


def reset_B(ym, P):
    a, w = ym[0], ym[1]
    c2, s2 = np.cos(2 * a), np.sin(2 * a)
    z2 = c2 * w + P * s2
    dz2_da = -2 * s2 * w + 2 * P * c2
    return np.array([[-1.0, 0.0, 0.0],
                     [dz2_da, c2, 0.0],
                     [2 * s2 * z2 + (1 - c2) * dz2_da, (1 - c2) * c2, 0.0]])


def jac_flow(z, delta=R.DELTA):
    A = np.zeros((3, 3))
    for j in range(3):
        zp = z.copy(); zp[j] += delta
        zm = z.copy(); zm[j] -= delta
        yp, ok1 = flow_to_event(zp); ym_, ok2 = flow_to_event(zm)
        assert ok1 and ok2
        A[:, j] = (yp - ym_) / (2 * delta)
    return A


def dominant(J):
    lam, V = eig(J)
    k = int(np.argmax(np.abs(lam)))
    lamL, W = eig(J.T)
    kl = int(np.argmin(np.abs(lamL - lam[k])))
    return lam[k], V[:, k], W[:, kl]


# ------------------------------------------------------------ branch 0.400 -> 0.010
S = np.round(np.arange(0.400, 0.0099, -H), 6)
Z, P, T, Y, A, B = [], [], [], [], [], []
seed = None
for s in S:
    g = R.solve_gait(float(s), seed=seed)
    assert g is not None, s
    seed = g["z_fp"]
    ym, ok = flow_to_event(g["z_fp"]); assert ok
    Z.append(g["z_fp"]); P.append(g["P"]); T.append(g["T"]); Y.append(ym)
    A.append(jac_flow(g["z_fp"])); B.append(reset_B(ym, g["P"]))
print(f"branch: {len(S)} gaits")

# ------------------------------------------------------------ decomposition
rows = {}
for i in range(1, len(S) - 1):
    J = B[i] @ A[i]
    lam, r, l = dominant(J)
    nrm = np.conj(l) @ r
    dA = (A[i - 1] - A[i + 1]) / (2 * H)
    dB = (B[i - 1] - B[i + 1]) / (2 * H)
    c_sw = (np.conj(l) @ (B[i] @ dA) @ r) / nrm
    c_rs = (np.conj(l) @ (dB @ A[i]) @ r) / nrm
    ep = 1e-6
    a0, w0, P0 = Y[i][0], Y[i][1], P[i]
    da = (Y[i - 1][0] - Y[i + 1][0]) / (2 * H)
    dw = (Y[i - 1][1] - Y[i + 1][1]) / (2 * H)
    dP = (P[i - 1] - P[i + 1]) / (2 * H)
    dBg = (reset_B([a0 + ep, w0, 0], P0) - reset_B([a0 - ep, w0, 0], P0)) / (2 * ep) * da
    dBv = (reset_B([a0, w0 + ep, 0], P0) - reset_B([a0, w0 - ep, 0], P0)) / (2 * ep) * dw
    dBp = (reset_B([a0, w0, 0], P0 + ep) - reset_B([a0, w0, 0], P0 - ep)) / (2 * ep) * dP
    cg = (np.conj(l) @ (dBg @ A[i]) @ r) / nrm
    cv = (np.conj(l) @ (dBv @ A[i]) @ r) / nrm
    cp = (np.conj(l) @ (dBp @ A[i]) @ r) / nrm
    lp = R.sorted_eigs(B[i - 1] @ A[i - 1])[0]; lm = R.sorted_eigs(B[i + 1] @ A[i + 1])[0]
    rows[float(S[i])] = dict(lam_max=abs(lam), dlam_dq_fd=float(np.real((lp - lm) / (2 * H))),
                            c_swing=float(np.real(c_sw)), c_reset=float(np.real(c_rs)),
                            c_reset_geom=float(np.real(cg)), c_reset_vel=float(np.real(cv)),
                            c_reset_push=float(np.real(cp)))

# ------------------------------------------------------------ compare with MATLAB
ML = {float(r["q"]): r for r in csv.DictReader(open(os.path.join(DATA, "jacobian_decomposition.csv")))}
cols = ["lam_max", "dlam_dq_fd", "c_swing", "c_reset", "c_reset_geom", "c_reset_vel", "c_reset_push"]
print("\n=== decomposition: max |python - matlab| over the real regime q <= 0.13 ===")
for c in cols:
    d = max(abs(rows[q][c] - float(ML[q][c])) for q in rows if q <= 0.13 and q in ML)
    print(f"  {c:<14} {d:.2e}")
real = sorted(q for q in rows if q <= 0.13)
I = lambda key: trapezoid([rows[q][key] for q in real], real)
tot = I("c_swing") + I("c_reset")
print("\n=== attribution over q in [%.3f, %.3f] (python) ===" % (real[0], real[-1]))
print(f"  swing/event : {100 * I('c_swing') / tot:6.1f} %")
print(f"  reset       : {100 * I('c_reset') / tot:6.1f} %")
print(f"    geometry  : {100 * I('c_reset_geom') / tot:6.1f} %")
print(f"    velocity  : {100 * I('c_reset_vel') / tot:6.1f} %")
print(f"    push-off  : {100 * I('c_reset_push') / tot:6.1f} %")

# ------------------------------------------------------------ freeze test
MF = {float(r["q"]): r for r in csv.DictReader(open(os.path.join(DATA, "jacobian_freeze_test.csv")))}
idx = {float(s): i for i, s in enumerate(S)}
iref = idx[0.010]
print("\n=== freeze test (A frozen / B frozen at q=0.010): max |python - matlab| ===")
dA_ = dB_ = 0.0
matched = 0
for q in MF:
    i = idx.get(q)
    if i is None:
        continue
    matched += 1
    la = abs(R.sorted_eigs(B[i] @ A[iref])[0]); lb = abs(R.sorted_eigs(B[iref] @ A[i])[0])
    dA_ = max(dA_, abs(la - float(MF[q]["lam_Afrozen_q010"])))
    dB_ = max(dB_, abs(lb - float(MF[q]["lam_Bfrozen_q010"])))
assert matched == len(MF), (matched, len(MF))
print(f"  lam_Afrozen  {dA_:.2e}\n  lam_Bfrozen  {dB_:.2e}")
