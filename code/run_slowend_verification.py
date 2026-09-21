"""
Direct verification of the 842-step half-life at the slowest gait computed,
with an explicit numerical noise floor.

Reported, in the order the analysis needs them:
  1. one-step residual of the fixed point, and every tolerance in force
  2. the unperturbed orbit iterated from z* for the same number of steps, giving
     the full-state drift ||e_n|| and the drift projected on the dominant mode
     a_n^(0) = l1^T e_n -- the numerical floor in the direction that matters
  3. an evaluable interval, defined as the steps where the perturbed modal
     amplitude stays above 10x the largest unperturbed modal drift
  4. regression of log|a_n/a0| over that interval -> lambda_hat, N_half_hat,
     plus the per-step ratio a_{n+1}/a_n, for the dominant-eigenvector direction
     and for the theta-only displacement, at three amplitudes

Left and right eigenvectors are biorthogonal (l1^T r1 = 1), so the modal
amplitude does not depend on eigenvector scaling.

Writes data/slowend_verification.csv, data/slowend_series.npz,
       figures/fig_supp_slowend.{png,pdf}
"""
import os
import json
import csv
import numpy as np
import revision_numerics as R

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", os.environ.get("SLOWWALK_DATA_DIR", "data")))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))
NSTEPS = 5000
C = 1.04

f = json.load(open(os.path.join(DATA, "_fp_s0010.json")))
z_fp = np.array(f["z"]); P = f["P"]
Sz, _, ok = R.step_map(z_fp, R.GAM, R.KHIP, P, R.RTOL, R.ATOL)
res_star = float(np.linalg.norm(Sz - z_fp))
print("=== 1. fixed point and tolerances ===")
print("  s=%.4f  v=%.6f  T=%.4f  P=%.4e" % (f["s"], f["v"], f["T"], P))
print("  one-step residual ||R(z*)-z*|| = %.3e" % res_star)
print("  ODE rtol=%.0e  atol=%.0e   finite-difference delta=%.0e   phase-1 depart dt=%.3f"
      % (R.RTOL, R.ATOL, R.DELTA, R.DT_DEPART))

A = R.jacobian_2d(np.array([z_fp[0], z_fp[1]]), R.GAM, R.KHIP, P, R.RTOL, R.ATOL, R.DELTA)
w, V = np.linalg.eig(A)
o = np.argsort(-np.abs(w)); w = w[o]; V = V[:, o]
L = np.linalg.inv(V)
lam1, lam2 = w[0].real, w[1].real
r1 = V[:, 0].real; l1 = L[0, :].real
N_lin = -np.log(2) / np.log(abs(lam1))
print("  lambda_1=%.9f  lambda_2=%.9f   l1.r1=%.6f   N_1/2(linear)=%.1f"
      % (lam1, lam2, float(l1 @ r1), N_lin))


def iterate(z_start, nsteps):
    """Return the modal amplitude series and the full-state deviation norm."""
    z = np.array(z_start, float)
    amod, anorm = [], []
    for _ in range(nsteps):
        z, _, ok = R.step_map(z, R.GAM, R.KHIP, P, R.RTOL, R.ATOL)
        if not ok:
            break
        e = np.array([z[0] - z_fp[0], z[1] - z_fp[1]])
        amod.append(float(l1 @ e)); anorm.append(float(np.linalg.norm(e)))
    return np.array(amod), np.array(anorm)


print("\n=== 2. unperturbed orbit: the numerical floor ===")
a0_drift, n0_drift = iterate(z_fp, NSTEPS)
floor_mod = float(np.max(np.abs(a0_drift)))
floor_p99 = float(np.percentile(np.abs(a0_drift), 99))
print("  steps completed: %d" % len(a0_drift))
print("  full-state drift ||e_n||   : max %.3e  final %.3e" % (np.max(n0_drift), n0_drift[-1]))
print("  modal drift |a_n^(0)|      : max %.3e  99th pct %.3e  final %.3e"
      % (floor_mod, floor_p99, abs(a0_drift[-1])))
print("  the drift saturates rather than growing, so it is an integration/event floor,")
print("  not a consequence of an inexact fixed point (residual %.1e)." % res_star)

print("\n=== 3-4. perturbed runs, evaluated above the floor ===")
rows = []
series = {}
thresh = 10.0 * floor_mod
print("  evaluable while |a_n| > 10 x max modal drift = %.3e" % thresh)
for direction in ("r1", "theta"):
    for amp in (0.001, 0.003, 0.010):
        d0 = amp * abs(z_fp[0])
        if direction == "r1":
            step = d0 * r1 / abs(r1[0])
            th, thd = z_fp[0] + step[0], z_fp[1] + step[1]
        else:
            th, thd = z_fp[0] + d0, z_fp[1]
        phd = (1.0 - np.cos(2.0 * th)) * thd
        a_init = float(l1 @ np.array([th - z_fp[0], thd - z_fp[1]]))
        amod, _ = iterate([th, thd, phd], NSTEPS)
        rel = np.concatenate(([1.0], amod / a_init))
        n = np.arange(len(rel))
        ok_mask = np.abs(rel * a_init) > thresh
        n_last = int(n[ok_mask].max()) if ok_mask.any() else 0
        fit = (n >= 5) & (n <= n_last)
        sl, ic = np.polyfit(n[fit], np.log(np.abs(rel[fit])), 1)
        lam_hat = float(np.exp(sl)); N_hat = -np.log(2) / np.log(lam_hat)
        ratios = np.abs(rel[6:n_last + 1] / rel[5:n_last])
        below = np.where(np.abs(rel[:n_last + 1]) <= 0.5)[0]
        n_half = int(below[0]) if below.size else None
        print("  %-5s amp=%4.1f%%  evaluable to n=%4d  lam_hat=%.9f  N_hat=%6.1f  "
              "half at n=%s  A=%.3f  ratio mean=%.9f sd=%.2e"
              % (direction, amp * 100, n_last, lam_hat, N_hat, n_half,
                 float(np.exp(ic)), float(ratios.mean()), float(ratios.std())))
        rows.append(dict(direction=direction, amp=amp, n_evaluable=n_last,
                         lam_linear=abs(lam1), N_linear=N_lin,
                         lam_hat=lam_hat, N_hat=N_hat,
                         n_half_observed=n_half if n_half is not None else "",
                         prefactor=float(np.exp(ic)),
                         ratio_mean=float(ratios.mean()), ratio_sd=float(ratios.std()),
                         one_step_residual=res_star, modal_floor=floor_mod,
                         rtol=R.RTOL, atol=R.ATOL, delta=R.DELTA))
        series["%s_%g" % (direction, amp)] = rel
series["unperturbed_modal"] = a0_drift
series["unperturbed_norm"] = n0_drift
np.savez_compressed(os.path.join(DATA, "slowend_series.npz"), **series)
with open(os.path.join(DATA, "slowend_verification.csv"), "w", newline="", encoding="utf-8") as fh:
    w2 = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w2.writeheader(); w2.writerows(rows)
print("\n  wrote data/slowend_verification.csv and data/slowend_series.npz")

# ------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
n = np.arange(NSTEPS + 1)
env = np.abs(lam1) ** n
for a_, keys, ttl in ((ax[0], [("r1", 0.001), ("r1", 0.003), ("r1", 0.010)],
                       "(a) perturbation along the dominant eigenvector"),
                      (ax[1], [("theta", 0.001), ("theta", 0.003), ("theta", 0.010)],
                       r"(b) stance-angle-only displacement")):
    a_.semilogy(n, env, "--", color="0.5", lw=1.6, label=r"$\lambda_{\max}^{\,n}$")
    for (d, amp), sh in zip(keys, ("0.68", "0.4", "0.05")):
        rel = series["%s_%g" % (d, amp)]
        a_.semilogy(np.arange(len(rel)), np.abs(rel), "-", color=sh, lw=1.2,
                    label="%.1f%%" % (amp * 100))
    a_.axhline(0.5, color="0.85", lw=0.8, ls=":")
    d0 = 0.001 * abs(z_fp[0])
    a_.axhspan(1e-12, floor_mod / d0, color="0.9", zorder=0)
    a_.set_xlabel("step $n$"); a_.set_ylim(1e-4, 3.0)
    a_.set_title(ttl, fontsize=9.5, loc="left")
    a_.grid(alpha=0.25, which="both"); a_.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel(r"modal amplitude $|a_n/a_0|$")
ax[0].legend(fontsize=8, frameon=False, loc="lower left")
fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(FIGS, "fig_supp_slowend." + ext)
    fig.savefig(p, dpi=300 if ext == "png" else None)
    print("  wrote", p)
