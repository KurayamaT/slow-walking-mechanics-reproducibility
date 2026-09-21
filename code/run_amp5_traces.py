"""
Nonlinear return-map recovery traces at LARGE perturbation amplitude.

The published amplitude check (Figure 2b, Supplementary S3) used 0.1-1.0% of
|theta*|. This run adds 5%, which is the first amplitude big enough to probe the
nonlinearity of the map rather than confirm the linearisation, and it does so at
three gaits including the slowest one computed, where N_1/2 = 842 steps had never
been checked against a nonlinear iteration.

Perturbation: the stance-leg angle alone is displaced at the post-impact fixed
point (not along an eigenvector); k_hip and P are held at their periodic-orbit
values, so the simulated recovery is open-loop. Two variants are run:
  on-constraint   phi_dot rebuilt as (1-cos 2theta) theta_dot  (2-D map)
  off-constraint  phi_dot left at its periodic value           (3-D map)
The off-constraint component is annihilated in one step (lambda_3 = 0), so the
two should agree after the first step; running both shows that they do.

Writes  data/amp_traces.csv  and  figures/fig_amp5_traces.{png,pdf}
"""
import os
import numpy as np
import revision_numerics as R

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", os.environ.get("SLOWWALK_DATA_DIR", "data")))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))

C = 1.04
AMPS = [0.001, 0.003, 0.010, 0.050]          # fractions of |theta*|
GAITS = [                                     # (s, steps to iterate, label)
    (0.402, 40, "plateau"),
    (0.080, 150, "low speed"),
    (0.010, 2000, "slowest computed"),
]


def _solve(s, seed):
    alpha = np.arcsin(0.5 * s)
    P = C * alpha * np.tan(alpha)
    z0 = list(seed) if seed is not None else         [alpha, -C * alpha, (1 - np.cos(2 * alpha)) * (-C * alpha)]
    z_fp, T, ok = R.find_fixed_point(z0, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, delta=1e-7)
    return (z_fp, T, P) if ok else (None, None, P)


def branch_point(s):
    """Seeded continuation down from s=0.40; the geometric guess stalls below
    s ~ 0.054, which is a solver limit, not a physical boundary."""
    z_fp, T, P = _solve(s, None)
    if z_fp is None:
        seed = None
        for sc in np.arange(0.400, s - 1e-9, -0.001):
            z_fp, T, P = _solve(float(sc), seed)
            if z_fp is None:
                raise RuntimeError("continuation failed at s=%.4f" % sc)
            seed = z_fp
        z_fp, T, P = _solve(s, seed)
    if z_fp is None:
        raise RuntimeError("fixed point failed at s=%.3f" % s)
    J = R.jacobian_3d(z_fp, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, R.DELTA)
    lam = sorted(np.linalg.eigvals(J), key=lambda L: -abs(L))
    lam_max = abs(lam[0])
    return z_fp, T, P, lam_max, s / T


def trace(z_fp, P, amp, nsteps, on_constraint):
    """Return normalised |theta_n - theta*| / |theta_0 - theta*| per step."""
    d0 = amp * abs(z_fp[0])
    th = z_fp[0] + d0
    thd = z_fp[1]
    if on_constraint:
        phd = (1.0 - np.cos(2.0 * th)) * thd
    else:
        phd = z_fp[2]
    z = np.array([th, thd, phd])
    out = [1.0]
    for _ in range(nsteps):
        z_new, _, ok = R.step_map(z, R.GAM, R.KHIP, P, R.RTOL, R.ATOL)
        if not ok:
            out.append(np.nan)
            break
        z = z_new
        out.append(abs(z[0] - z_fp[0]) / d0)
    return np.array(out)


rows = []
results = {}
for s, nsteps, label in GAITS:
    z_fp, T, P, lam_max, v = branch_point(s)
    N_half = -np.log(2) / np.log(lam_max)
    print("s=%.3f v=%.5f |lam_max|=%.9f N_half=%.1f  theta*=%.6f rad" %
          (s, v, lam_max, N_half, z_fp[0]))
    per_gait = dict(s=s, v=v, lam_max=lam_max, N_half=N_half,
                    theta=z_fp[0], label=label, traces={})
    for amp in AMPS:
        for oc in (True, False):
            tr = trace(z_fp, P, amp, nsteps, oc)
            fell = np.isnan(tr[-1])
            per_gait["traces"][(amp, oc)] = tr
            # asymptotic decay rate from a log-linear fit over the tail, plus the
            # transient prefactor A in  tr_n ~ A * lam^n.  The first crossing of
            # 0.5 is NOT N_1/2: it also carries the initial transient.
            meas = lam_fit = A = float("nan")
            if not fell and len(tr) > 12:
                tail = slice(max(4, (len(tr) * 2) // 3), len(tr))
                n_t = np.arange(len(tr))[tail]
                y = np.log(np.abs(tr[tail]))
                good = np.isfinite(y)
                if good.sum() > 4:
                    sl, ic = np.polyfit(n_t[good], y[good], 1)
                    lam_fit = float(np.exp(sl)); A = float(np.exp(ic))
                    meas = -np.log(2) / np.log(lam_fit) if 0 < lam_fit < 1 else float("nan")
            print("    amp=%5.1f%%  %-14s final=%9.3e  lam_fit=%.9f  N_half_fit=%8.1f  A=%6.2f%s"
                  % (amp * 100, "on-constraint" if oc else "off-constraint",
                     tr[-1] if not fell else float("nan"),
                     lam_fit, meas, A, "   FELL step %d" % (len(tr) - 1) if fell else ""))
            rows.append(dict(s=s, v=v, amp=amp, on_constraint=int(oc),
                             lam_max=lam_max, N_half_linear=N_half,
                             lam_fitted=lam_fit, N_half_fitted=meas, prefactor_A=A,
                             final_ratio=(tr[-1] if not fell else ""),
                             steps=len(tr) - 1, fell=int(fell)))
    results[s] = per_gait

os.makedirs(DATA, exist_ok=True)
import csv
with open(os.path.join(DATA, "amp_traces.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nwrote", os.path.join(DATA, "amp_traces.csv"))

# ----------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7))
greys = {0.001: "0.72", 0.003: "0.55", 0.010: "0.32", 0.050: "0.0"}
for ax, (s, nsteps, label) in zip(axes, GAITS):
    g = results[s]
    n = np.arange(nsteps + 1)
    env = g["lam_max"] ** n
    ax.semilogy(n, env, "--", color="0.45", lw=1.6, zorder=1,
                label=r"$|\lambda_{\max}|^{n}$")
    for amp in AMPS:
        tr = g["traces"][(amp, True)]
        fell = np.isnan(tr[-1])
        ax.semilogy(np.arange(len(tr)), np.abs(tr), "-", color=greys[amp], lw=1.3,
                    zorder=3, label="%.1f%%%s" % (amp * 100, " (fell)" if fell else ""))
        if fell:
            ax.plot(len(tr) - 1, np.abs(tr[-2]), "x", color=greys[amp], ms=8, mew=2, zorder=4)
    ax.axhline(1.0, color="0.88", lw=0.8, zorder=0)
    ax.axhline(0.5, color="0.88", lw=0.8, ls=":", zorder=0)
    ax.set_title("%s\n$v$=%.4f, $|\\lambda_{\\max}|$=%.4f, $N_{1/2}$=%.0f"
                 % (label, g["v"], g["lam_max"], g["N_half"]), fontsize=9)
    ax.set_xlabel("step $n$")
    top = max(2.0, 2.0 * max(np.nanmax(np.abs(g["traces"][(a, True)]))
                             for a in AMPS
                             if np.isfinite(np.nanmax(np.abs(g["traces"][(a, True)])))))
    ax.set_ylim(1e-3, top)
    ax.grid(True, which="both", alpha=0.25)
axes[0].set_ylabel(r"normalised $|\theta_n-\theta^*|$")
axes[0].legend(fontsize=8, loc="lower left", title="amplitude", title_fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    out = os.path.join(FIGS, "fig_amp5_traces." + ext)
    fig.savefig(out, dpi=300 if ext == "png" else None)
    print("wrote", out)
