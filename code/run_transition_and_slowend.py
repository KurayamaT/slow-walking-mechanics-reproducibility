"""
Two verifications requested before submission.

(A) Transition, resolved directly on the reduced two-dimensional return-map
    Jacobian A on the post-impact constraint surface:
        tr A,  det A = lambda_1 lambda_2,  Delta = (tr A)^2 - 4 det A
    A square-root eigenvalue coalescence requires Delta to cross zero roughly
    linearly in v, which makes (Im lambda)^2 linear in v just above the
    transition. That is checked here rather than asserted from Im/sqrt(v-v*).

(B) The 842-step half-life at the slowest gait computed (s = 0.010), which had
    only ever been checked against a nonlinear iteration where N_1/2 was about
    12 and about 2. Two perturbation directions are used:
      r1    the dominant right eigenvector of A -- excites the slow mode alone
      theta the stance-angle-only displacement used for Figure 2b
    Modal amplitude is measured with the corresponding left eigenvector,
        a_n = l1^T (z_n - z*),
    so the slow mode can be followed even when the initial condition also
    excites the fast mode. Convergence with respect to the finite-difference
    width, the ODE tolerance and the Newton residual is reported alongside, so
    the spread across numerical settings can be quoted instead of a fake
    error bar.

Writes data/transition_spectrum.csv, data/slowend_modal.csv,
       figures/fig_supp_transition.{png,pdf}
"""
import os
import json
import numpy as np
import revision_numerics as R

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", os.environ.get("SLOWWALK_DATA_DIR", "data")))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))
C = 1.04


def push_off(s):
    a = np.arcsin(0.5 * s)
    return a, C * a * np.tan(a)


def fp2d(q, seed=None, rtol=R.RTOL, atol=R.ATOL, delta=1e-7):
    """Fixed point of the reduced 2-D map at branch parameter q, returned with T,
    P and the residual. q prescribes the push-off; the realised step length is
    s = 2 sin(alpha_h) with alpha_h = z[0]."""
    a, P = push_off(q)
    z0 = list(seed) if seed is not None else [a, -C * a, (1 - np.cos(2 * a)) * (-C * a)]
    z, T, ok = R.find_fixed_point(z0, R.GAM, R.KHIP, P, rtol, atol, delta=delta)
    if not ok:
        return None
    Sz, _, ok2 = R.step_map(z, R.GAM, R.KHIP, P, rtol, atol)
    res = float(np.linalg.norm(Sz - z)) if ok2 else np.nan
    alpha_h = float(z[0])
    s = 2.0 * np.sin(alpha_h)
    return dict(z=z, T=T, P=P, q=q, alpha_p=a, alpha_h=alpha_h,
                s=s, v=s / T, residual=res)


def spectrum2d(f, rtol=R.RTOL, atol=R.ATOL, delta=R.DELTA):
    """tr, det, discriminant and eigen-decomposition of the reduced 2-D Jacobian."""
    u = np.array([f["z"][0], f["z"][1]])
    A = R.jacobian_2d(u, R.GAM, R.KHIP, f["P"], rtol, atol, delta)
    if A is None:
        return None
    tr = float(np.trace(A)); det = float(np.linalg.det(A))
    disc = tr * tr - 4.0 * det
    w, V = np.linalg.eig(A)
    order = np.argsort(-np.abs(w))
    w = w[order]; V = V[:, order]
    Vinv = np.linalg.inv(V)                      # rows are the left eigenvectors
    return dict(A=A, tr=tr, det=det, disc=disc, lam=w, right=V, left=Vinv)


# ---------------------------------------------------------------- (A) transition
print("=== (A) transition spectrum on the reduced 2-D Jacobian ===")
rows = []
seed = None
for s in np.arange(0.1300, 0.1560, 0.0004):
    f = fp2d(float(s), seed)
    if f is None:
        print("  fixed point failed at s=%.4f" % s); continue
    seed = f["z"]
    sp = spectrum2d(f)
    if sp is None:
        continue
    l1, l2 = sp["lam"][0], sp["lam"][1]
    rows.append(dict(q=f["q"], alpha_p=f["alpha_p"], alpha_h=f["alpha_h"],
                     s=f["s"], v=f["v"], T=f["T"], tr=sp["tr"], det=sp["det"],
                     disc=sp["disc"], l1_re=l1.real, l1_im=l1.imag,
                     l2_re=l2.real, l2_im=l2.imag,
                     absmax=float(max(abs(l1), abs(l2))),
                     sqrt_det=float(np.sqrt(sp["det"])) if sp["det"] > 0 else np.nan))
rows.sort(key=lambda r: r["q"])
cx = [r for r in rows if abs(r["l1_im"]) > 0]
re_ = [r for r in rows if abs(r["l1_im"]) == 0]
v_lo = max(r["v"] for r in re_) if re_ else np.nan
v_hi = min(r["v"] for r in cx) if cx else np.nan
print("  transition bracketed in v = [%.6f, %.6f]  (width %.2e)" % (v_lo, v_hi, v_hi - v_lo))

# is the discriminant linear through zero?
near = [r for r in rows if abs(r["v"] - 0.5 * (v_lo + v_hi)) < 0.004]
vv = np.array([r["v"] for r in near]); dd = np.array([r["disc"] for r in near])
sl, ic = np.polyfit(vv, dd, 1)
pred = sl * vv + ic
r2 = 1.0 - np.sum((dd - pred) ** 2) / np.sum((dd - dd.mean()) ** 2)
v_zero = -ic / sl
print("  discriminant fit  Delta = %.5f*(v) + %.6f   R^2 = %.6f" % (sl, ic, r2))
print("  Delta = 0 at v = %.6f   (bracket midpoint %.6f)" % (v_zero, 0.5 * (v_lo + v_hi)))
# (Im lambda)^2 should be linear in v above the transition, with slope -sl
cxn = [r for r in cx if r["v"] - v_hi < 0.004]
if len(cxn) > 4:
    vc = np.array([r["v"] for r in cxn]); im2 = np.array([r["l1_im"] ** 2 for r in cxn])
    s2, i2 = np.polyfit(vc, im2, 1)
    pr2 = s2 * vc + i2
    r2b = 1.0 - np.sum((im2 - pr2) ** 2) / np.sum((im2 - im2.mean()) ** 2)
    print("  (Im lambda)^2 fit slope = %.5f  R^2 = %.6f   (expect slope = -Delta slope/4 = %.5f)"
          % (s2, r2b, -sl / 4.0))
print("  det A over the neighbourhood: %.6f to %.6f" % (min(r["det"] for r in rows),
                                                        max(r["det"] for r in rows)))

import csv
with open(os.path.join(DATA, "transition_spectrum.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("  wrote data/transition_spectrum.csv")

# ---------------------------------------------------------------- (B) slow end
print("\n=== (B) slowest gait computed: modal verification of N_1/2 ===")
CACHE = os.path.join(DATA, "_fp_s0010.json")
if os.path.exists(CACHE):
    f10 = json.load(open(CACHE))
    f10["z"] = np.array(f10["z"])
else:
    seed = None
    for sc in np.arange(0.400, 0.0099, -0.001):
        g = fp2d(float(sc), seed)
        if g is None:
            raise RuntimeError("continuation failed at s=%.4f" % sc)
        seed = g["z"]
    f10 = g
    json.dump(dict(z=list(map(float, f10["z"])), T=f10["T"], P=f10["P"],
                   s=f10["s"], v=f10["v"], residual=f10["residual"]), open(CACHE, "w"))
print("  s=%.4f v=%.6f T=%.4f P=%.3e  Newton residual=%.2e"
      % (f10["s"], f10["v"], f10["T"], f10["P"], f10["residual"]))

sp10 = spectrum2d(f10)
lam1, lam2 = sp10["lam"][0].real, sp10["lam"][1].real
r1 = sp10["right"][:, 0].real; l1 = sp10["left"][0, :].real
N_lin = -np.log(2) / np.log(abs(lam1))
print("  lambda_1=%.9f  lambda_2=%.9f   N_1/2(linear)=%.1f" % (lam1, lam2, N_lin))
print("  dominant right eigenvector (theta, thetadot) = (%.4f, %.4f)" % (r1[0], r1[1]))

# convergence across numerical settings
print("\n  --- convergence of lambda_1 across numerical settings ---")
settings = []
for d in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
    sp = spectrum2d(f10, delta=d)
    settings.append(("delta=%.0e" % d, abs(sp["lam"][0])))
for rt in (1e-10, 1e-11, 1e-12, 1e-13):
    ff = fp2d(0.010, f10["z"], rtol=rt, atol=1e-14)
    if ff is None:
        settings.append(("rtol=%.0e" % rt, np.nan)); continue
    sp = spectrum2d(ff, rtol=rt, atol=1e-14)
    settings.append(("rtol=%.0e" % rt, abs(sp["lam"][0])))
lams = [x for _, x in settings if np.isfinite(x)]
for name, x in settings:
    print("     %-12s lambda_1=%.9f  N_1/2=%8.1f" % (name, x, -np.log(2) / np.log(x)))
Ns = [-np.log(2) / np.log(x) for x in lams]
print("     spread across settings: lambda_1 %.2e ; N_1/2 %.1f to %.1f"
      % (max(lams) - min(lams), min(Ns), max(Ns)))

# nonlinear iteration, modal amplitude
NSTEPS = 5000
out = []
for direction in ("r1", "theta"):
    for amp in (0.001, 0.003, 0.010):
        d0 = amp * abs(f10["z"][0])
        if direction == "r1":
            step = d0 * r1 / max(abs(r1[0]), 1e-300)      # scale so the theta part is d0
            th, thd = f10["z"][0] + step[0], f10["z"][1] + step[1]
        else:
            th, thd = f10["z"][0] + d0, f10["z"][1]
        phd = (1.0 - np.cos(2.0 * th)) * thd
        z = np.array([th, thd, phd])
        a0 = float(l1 @ np.array([z[0] - f10["z"][0], z[1] - f10["z"][1]]))
        amps = [1.0]
        for _ in range(NSTEPS):
            z, _, ok = R.step_map(z, R.GAM, R.KHIP, f10["P"], R.RTOL, R.ATOL)
            if not ok:
                amps.append(np.nan); break
            an = float(l1 @ np.array([z[0] - f10["z"][0], z[1] - f10["z"][1]]))
            amps.append(an / a0)
        amps = np.array(amps)
        n = np.arange(len(amps))
        good = np.isfinite(amps) & (np.abs(amps) > 0)
        tail = good & (n > len(amps) // 3)
        sl2, ic2 = np.polyfit(n[tail], np.log(np.abs(amps[tail])), 1)
        lam_fit = float(np.exp(sl2)); N_fit = -np.log(2) / np.log(lam_fit)
        below = np.where(np.abs(amps) <= 0.5)[0]
        n_half_obs = int(below[0]) if below.size else None
        print("  %-6s amp=%4.1f%%  lam_fit=%.9f  N_1/2(fit)=%7.1f  first|a|<0.5 at n=%s  A=%.3f"
              % (direction, amp * 100, lam_fit, N_fit, n_half_obs, float(np.exp(ic2))))
        out.append(dict(direction=direction, amp=amp, lam_linear=abs(lam1),
                        N_linear=N_lin, lam_fitted=lam_fit, N_fitted=N_fit,
                        n_half_observed=n_half_obs if n_half_obs is not None else "",
                        prefactor=float(np.exp(ic2)), steps=len(amps) - 1))
with open(os.path.join(DATA, "slowend_modal.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print("  wrote data/slowend_modal.csv")

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
v = np.array([r["v"] for r in rows])
fig, ax = plt.subplots(2, 2, figsize=(9.6, 6.4))
a0 = ax[0, 0]
a0.plot(v, [r["l1_re"] for r in rows], "-", color="0.15", lw=1.6, label=r"$\mathrm{Re}\,\lambda$")
a0.plot(v, [r["l2_re"] for r in rows], "-", color="0.15", lw=1.6)
a0.plot(v, [r["l1_im"] for r in rows], "--", color="0.55", lw=1.5, label=r"$\pm\,\mathrm{Im}\,\lambda$")
a0.plot(v, [r["l2_im"] for r in rows], "--", color="0.55", lw=1.5)
a0.set_ylabel("multiplier"); a0.legend(fontsize=8, frameon=False)
a0.set_title("(a) the two non-zero multipliers coalesce", fontsize=9, loc="left")
a1 = ax[0, 1]
a1.plot(v, [r["tr"] for r in rows], "-", color="0.15", lw=1.6, label=r"$\mathrm{tr}\,A$")
a1.plot(v, [r["det"] for r in rows], "-", color="0.55", lw=1.6, label=r"$\det A$")
a1.legend(fontsize=8, frameon=False)
a1.set_title(r"(b) $\mathrm{tr}\,A$ and $\det A=\lambda_1\lambda_2$", fontsize=9, loc="left")
a2 = ax[1, 0]
a2.plot(v, [r["disc"] for r in rows], "-", color="0.15", lw=1.7)
a2.axhline(0.0, color="0.75", lw=0.9)
a2.axvline(v_zero, color="0.6", ls=":", lw=1.1)
a2.set_xlabel(r"dimensionless speed $v$"); a2.set_ylabel(r"$\Delta$")
a2.set_title(r"(c) discriminant $\Delta=(\mathrm{tr}A)^2-4\det A$ crosses zero", fontsize=9, loc="left")
a3 = ax[1, 1]
a3.plot([r["v"] for r in cx], [r["l1_im"] ** 2 for r in cx], "o", ms=3, color="0.15")
a3.set_xlabel(r"dimensionless speed $v$"); a3.set_ylabel(r"$(\mathrm{Im}\,\lambda)^2$")
a3.set_title(r"(d) $(\mathrm{Im}\,\lambda)^2$ linear in $v$ = square-root coalescence",
             fontsize=9, loc="left")
for a in ax.ravel():
    a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(FIGS, "fig_supp_transition." + ext)
    fig.savefig(p, dpi=300 if ext == "png" else None)
    print("  wrote", p)
