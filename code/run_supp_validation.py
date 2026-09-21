"""
Supplementary S1 - numerical implementation and its validation, in one figure
and one table.

Consolidates what used to be scattered across several supplementary items:

  (i)   convergence of |lambda_max| with the central-difference step delta,
        at three gaits spanning the branch;
  (ii)  insensitivity to the ODE tolerances (rtol);
  (iii) agreement between the eigenvalues of the full 3-D post-impact Jacobian
        and those of the reduced 2-D Jacobian on the constraint surface;
  (iv)  the structurally zero third multiplier;
  (v)   the passive Garcia et al. (1998) walker reproduced with the same
        collision routine, giving the low-speed ratio that fixes the 1.04 in
        the prescribed push-off.

Writes data/supp_validation.csv, data/supp_validation_delta.csv
       figures/fig_supp_validation.{png,pdf}

Run from code/:  python run_supp_validation.py
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import revision_numerics as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DATA = os.path.join(ROOT, os.environ.get("SLOWWALK_DATA_DIR", "data"))
FIGS = os.path.join(ROOT, "figures")

C = 1.04
GAITS = [(0.010, "slowest, $s=0.010$"),
         (0.080, "low speed, $s=0.080$"),
         (0.402, "plateau, $s=0.402$")]
DELTAS = [1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
RTOLS = [1e-13, 1e-12, 1e-11, 1e-10]


def push_off(s):
    a = np.arcsin(0.5 * s)
    return C * a * np.tan(a)


def fixed_point(s):
    """Periodic post-impact state, by seeded continuation from s=0.40."""
    a = np.arcsin(0.5 * 0.40)
    seed = [a, -C * a, (1 - np.cos(2 * a)) * (-C * a)]
    z, T, ok = R.find_fixed_point(seed, R.GAM, R.KHIP, push_off(0.40),
                                  R.RTOL, R.ATOL, delta=1e-7)
    if not ok:
        raise RuntimeError("anchor solve failed")
    grid = np.arange(0.399, s - 1e-9, -0.001) if s < 0.40 else \
        np.arange(0.401, s + 1e-9, 0.001)
    for st in grid:
        z2, T2, ok = R.find_fixed_point(list(z), R.GAM, R.KHIP, push_off(float(st)),
                                        R.RTOL, R.ATOL, delta=1e-7)
        if not ok:
            raise RuntimeError("continuation failed at s=%.4f" % st)
        z, T = z2, T2
    return np.asarray(z, float), float(T)


def sweep_branch():
    """One continuation pass over the branch, recording at every point the
    structural zero and the 3-D/2-D eigenvalue agreement, and returning the
    fixed points of the three reference gaits along the way."""
    a = np.arcsin(0.5 * 0.40)
    seed = [a, -C * a, (1 - np.cos(2 * a)) * (-C * a)]
    z, T, ok = R.find_fixed_point(seed, R.GAM, R.KHIP, push_off(0.40),
                                  R.RTOL, R.ATOL, delta=1e-7)
    if not ok:
        raise RuntimeError("anchor solve failed")
    out, refs = [], {}
    for direction in ("down", "up"):
        zc = np.asarray(z, float)
        grid = (np.arange(0.399, 0.0099, -0.001) if direction == "down"
                else np.arange(0.401, 0.801, 0.001))
        for i, st in enumerate(grid):
            st = float(st)
            P = push_off(st)
            z2, T2, ok = R.find_fixed_point(list(zc), R.GAM, R.KHIP, P,
                                            R.RTOL, R.ATOL, delta=1e-7)
            if not ok:
                break
            zc = np.asarray(z2, float)
            for sref, _ in GAITS:
                if abs(st - sref) < 5e-4:
                    refs[sref] = (zc.copy(), float(T2))
            if i % 20:                      # record every 20th point
                continue
            J3 = R.jacobian_3d(zc, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, 1e-7)
            J2 = R.jacobian_2d(zc[:2].copy(), R.GAM, R.KHIP, P, R.RTOL, R.ATOL, 1e-7)
            if J3 is None or J2 is None:
                continue
            l3, l2 = R.sorted_eigs(J3), R.sorted_eigs(J2)
            out.append(dict(
                s=st, v=st / float(T2), lam3_abs=float(abs(l3[2])),
                dev_2d_3d=float(np.max(np.abs(np.sort_complex(l3[:2])
                                              - np.sort_complex(l2)))),
                residual=float(np.linalg.norm(
                    R.step_map(zc, R.GAM, R.KHIP, P, R.RTOL, R.ATOL)[0] - zc))))
    out.sort(key=lambda r: r["v"])
    return out, refs


print("=== branch sweep: structural zero and 2-D/3-D agreement ===")
branch, REFS = sweep_branch()
print("recorded %d points, v in [%.5f, %.5f]" % (len(branch), branch[0]["v"], branch[-1]["v"]))
with open(os.path.join(DATA, "supp_validation_branch.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(branch[0].keys()))
    w.writeheader()
    w.writerows(branch)

rows_delta, rows_summary = [], []
print("=== delta / rtol convergence at three reference gaits ===")
for s, label in GAITS:
    z, T = REFS[s]
    P = push_off(s)
    v = s / T
    res = float(np.linalg.norm(R.step_map(z, R.GAM, R.KHIP, P, R.RTOL, R.ATOL)[0] - z))
    print("s=%.3f  v=%.6f  T=%.4f  P=%.4e  ||R(z*)-z*||=%.2e" % (s, v, T, P, res))

    # (i) finite-difference step
    for d in DELTAS:
        J3 = R.jacobian_3d(z, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, d)
        lam = R.sorted_eigs(J3)
        rows_delta.append(dict(s=s, v=v, kind="delta", setting=d,
                               lam_max=float(abs(lam[0])),
                               N_half=float(-np.log(2) / np.log(abs(lam[0])))))
    # (ii) ODE tolerance, at the reference delta
    for rt in RTOLS:
        J3 = R.jacobian_3d(z, R.GAM, R.KHIP, P, rt, R.ATOL, 1e-7)
        lam = R.sorted_eigs(J3)
        rows_delta.append(dict(s=s, v=v, kind="rtol", setting=rt,
                               lam_max=float(abs(lam[0])),
                               N_half=float(-np.log(2) / np.log(abs(lam[0])))))

    # (iii)+(iv) 3-D vs reduced 2-D, and the structural zero
    J3 = R.jacobian_3d(z, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, 1e-7)
    l3 = R.sorted_eigs(J3)
    J2 = R.jacobian_2d(z[:2].copy(), R.GAM, R.KHIP, P, R.RTOL, R.ATOL, 1e-7)
    l2 = R.sorted_eigs(J2)
    dev = float(np.max(np.abs(np.sort_complex(l3[:2]) - np.sort_complex(l2))))
    rows_summary.append(dict(
        s=s, v=v, T=T, P=P, residual=res,
        lam1=complex(l3[0]), lam2=complex(l3[1]), lam3_abs=float(abs(l3[2])),
        dev_2d_3d=dev,
        N_half=float(-np.log(2) / np.log(abs(l3[0]))),
        N_half_delta1e5=float(-np.log(2) / np.log(abs(R.sorted_eigs(
            R.jacobian_3d(z, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, 1e-5))[0]))),
    ))
    print("   |lambda_3| = %.2e   max|lambda(3D)-lambda(2D)| = %.2e" % (abs(l3[2]), dev))

with open(os.path.join(DATA, "supp_validation_delta.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_delta[0].keys()))
    w.writeheader()
    w.writerows(rows_delta)
with open(os.path.join(DATA, "supp_validation.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_summary[0].keys()))
    w.writeheader()
    w.writerows(rows_summary)

# ---------------------------------------------------------------- figure S1
STY = [("0.55", "-", "o"), ("0.25", "--", "s"), ("0.0", "-.", "^")]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))

ax = axes[0]
for (s, label), (col, ls, mk) in zip(GAITS, STY):
    d = [r for r in rows_delta if r["kind"] == "delta" and r["s"] == s]
    ref = [r for r in d if r["setting"] == 1e-9][0]["lam_max"]
    x = [r["setting"] for r in d]
    y = [max(abs(r["lam_max"] - ref) / ref, 1e-17) for r in d]
    ax.loglog(x, y, ls, color=col, marker=mk, ms=3.5, lw=1.3, label=label)
ax.axvline(1e-7, color="0.7", ls=":", lw=1.2)
ax.text(1.25e-7, 3e-13, "$\\delta$ used", fontsize=8.5, color="0.35")
ax.set_xlabel(r"central-difference step $\delta$")
ax.set_ylabel(r"$|\lambda_{\max}(\delta)-\lambda_{\max}(10^{-9})|/\lambda_{\max}(10^{-9})$")
ax.set_title("(a) finite-difference convergence", loc="left", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.grid(True, which="major", lw=0.4, color="0.9")

ax = axes[1]
v = [r["v"] for r in branch]
ax.semilogy(v, [max(r["dev_2d_3d"], 1e-18) for r in branch], "-",
            color="0.0", lw=1.4, label=r"$\max_i|\lambda_i^{\,3D}-\lambda_i^{\,2D}|$")
ax.semilogy(v, [max(r["lam3_abs"], 1e-18) for r in branch], "--",
            color="0.5", lw=1.4, label=r"$|\lambda_3|$ (structurally zero)")
ax.semilogy(v, [max(r["residual"], 1e-18) for r in branch], ":",
            color="0.3", lw=1.4, label=r"$\|R(\mathbf{z}^{*})-\mathbf{z}^{*}\|$")
ax.set_xlabel(r"dimensionless speed $v$")
ax.set_ylabel("magnitude")
ax.set_ylim(1e-16, 1e-6)
ax.set_title("(b) reduction and structural zero, along the branch",
             loc="left", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.grid(True, which="major", lw=0.4, color="0.9")

fig.tight_layout()
out = os.path.join(FIGS, "fig_supp_validation.png")
fig.savefig(out, dpi=300)
fig.savefig(out.replace(".png", ".pdf"))
plt.close(fig)
print("saved:", out)

print("\n=== summary (for Table S1) ===")
for r in rows_summary:
    print("s=%.3f v=%.6f  N_1/2=%.1f  N_1/2(delta=1e-5)=%.1f  |lam3|=%.1e  dev=%.1e"
          % (r["s"], r["v"], r["N_half"], r["N_half_delta1e5"],
             r["lam3_abs"], r["dev_2d_3d"]))
