"""
Push-off-coefficient robustness, consistent with the corrected (extended, no-fold)
family. For each c in {0.80, 1.00, 1.04, 1.20}, the prescribed push-off is
P(s) = c * alpha * tan(alpha); the periodic branch is traced by seeded
continuation over s in [0.01, 0.80] (same method as run_extend_family.py).

Shows that for every c the branch extends smoothly to low speed with no fold encountered over
the computed range, and N_1/2 rises steeply toward the slow end — i.e. the low-speed
weakening is not an artefact of the specific coefficient 1.04. No claim is made
about the limit v -> 0: there is no analytic asymptotic derivation, so only the
trend over the computed range is reported.

Output: newest/figures/sens_pushoff_c.png (+ .tiff, .pdf)  (N_1/2 vs v, log axis, one curve per c)
        newest/data/supp_csweep.csv                        (the traced branches)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import revision_numerics as R

CS = [0.80, 1.00, 1.04, 1.20]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "figures", "sens_pushoff_c.png")


def solve_at(q, c, seed):
    """q is the branch parameter; the realised step length is 2 sin(alpha_h)."""
    alpha_p = np.arcsin(0.5 * q)
    P = c * alpha_p * np.tan(alpha_p)
    z0 = list(seed) if seed is not None else [alpha_p, -c * alpha_p, (1 - np.cos(2 * alpha_p)) * (-c * alpha_p)]
    z_fp, T, ok = R.find_fixed_point(z0, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, delta=1e-7)
    if not ok:
        return None
    J = R.jacobian_3d(z_fp, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, R.DELTA)
    if J is None:
        return None
    lam_max = float(np.max(np.abs(R.sorted_eigs(J))))
    alpha_h = float(z_fp[0])
    s = 2.0 * np.sin(alpha_h)
    return dict(q=q, alpha_h=alpha_h, s=s, v=s / T, lam_max=lam_max, z=list(z_fp))


REF = solve_at(0.40, 1.04, None)   # converges from the geometric guess; seed for all c


def anchor_for(c):
    """Reach the s=0.40 fixed point at coefficient c by continuation in c from 1.04
    (the geometric initial guess does not converge directly for c != 1.04)."""
    if abs(c - 1.04) < 1e-9:
        return REF
    n = int(round(abs(c - 1.04) / 0.01))
    seed = REF["z"]
    r = REF
    for ct in np.linspace(1.04, c, n + 1)[1:]:
        r = solve_at(0.40, float(ct), seed)
        if r is None:
            raise RuntimeError(f"c-continuation failed near c={ct:.3f}")
        seed = r["z"]
    return r


def trace(c):
    anchor = anchor_for(c)
    rows = [anchor]
    sd = anchor["z"]
    for s in np.arange(0.399, 0.0099, -0.001):
        r = solve_at(float(s), c, sd)
        if r is None:
            break
        sd = r["z"]
        rows.append(r)
    su = anchor["z"]
    for s in np.arange(0.401, 0.801, 0.001):
        r = solve_at(float(s), c, su)
        if r is None:
            break
        su = r["z"]
        rows.append(r)
    rows.sort(key=lambda r: r["q"])
    v = np.array([r["v"] for r in rows])
    qq = np.array([r["q"] for r in rows])
    ah = np.array([r["alpha_h"] for r in rows])
    sr = np.array([r["s"] for r in rows])
    lm = np.array([r["lam_max"] for r in rows])
    nh = -np.log(2.0) / np.log(np.clip(lm, 1e-12, 0.9999999))
    return v, lm, nh, qq, ah, sr


plt.rcParams.update({"font.size": 11})

# --- compute all traces once (expensive), then plot both mono + color ---
traces = []
print(f"{'c':>5} {'n':>4} {'v_lo':>7} {'Nh_lo':>7} {'v~0.016':>8} {'Nh':>6} {'v_hi':>7} {'Nh_hi':>6}")
for c in CS:
    v, lm, nh, qq, ah, sr = trace(c)
    traces.append((c, v, nh, qq, ah, sr))
    i_lo = int(np.argmin(v)); i_mid = int(np.argmin(np.abs(v - 0.016))); i_hi = int(np.argmax(v))
    print(f"{c:5.2f} {len(v):4d} {v[i_lo]:7.4f} {nh[i_lo]:7.1f} {v[i_mid]:8.4f} {nh[i_mid]:6.1f} {v[i_hi]:7.4f} {nh[i_hi]:6.1f}")

# the traced branches, so Supplementary S4 can be replotted without recomputing
import csv as _csv
with open(os.path.join(ROOT, os.environ.get("SLOWWALK_DATA_DIR", "data"), "supp_csweep.csv"), "w", newline="") as _fh:
    _w = _csv.writer(_fh)
    _w.writerow(["c", "q", "alpha_h", "s", "v", "lam_max", "N_half"])
    for c, v, nh, qq, ah, sr in traces:
        lm = np.exp(-np.log(2.0) / nh)
        for vi, li, ni, qi, ai, si in zip(v, lm, nh, qq, ah, sr):
            # q at three decimals is the grid key; the realised s must keep its
            # digits or every branch would collapse onto the same label.
            _w.writerow(["%.2f" % c, "%.3f" % qi, "%.10g" % ai, "%.10g" % si,
                         "%.10g" % vi, "%.10f" % li, "%.4f" % ni])
print("saved: data/supp_csweep.csv")


def make(palette, suffix):
    """palette: list of (color, linestyle, dashes, lw) in CS order."""
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for (c, v, nh), (col, ls, dash, lw) in zip(traces, palette):
        lab = f"$c={c:.2f}$" + ("  (main)" if abs(c - 1.04) < 1e-9 else "")
        ax.plot(v, nh, ls, color=col, lw=lw, dashes=dash, label=lab,
                zorder=4 if abs(c - 1.04) < 1e-9 else 3)
    ax.axvline(0.041, color="0.6", ls=":", lw=1.0, zorder=1)
    ax.annotate(r"$N_{1/2}$ rises steeply toward the slow end for every $c$,"
                + "\nwith no fold encountered along the branch traced",
                xy=(0.006, 200), xytext=(0.055, 250), fontsize=9, color="0.3", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="0.55"))
    ax.set_yscale("log")
    ax.set_xlabel(r"Dimensionless speed  $v$")
    ax.set_ylabel(r"Perturbation half-life  $N_{1/2}$ (steps)")
    ax.set_xlim(0.0, 0.235)
    ax.set_ylim(1, 1100)
    ax.set_title(r"Push-off coefficient $c$ in $P(s)=c\,\alpha\tan\alpha$: low-speed weakening is robust",
                 loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9.5, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = OUT.replace(".png", suffix + ".png")
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace(".png", ".tiff"), dpi=300)
    # vector copy for submission: Elsevier asks 1000 dpi for bitmapped line
    # drawings, which vector output makes moot.
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print("saved:", out)


# grayscale (unchanged)
MONO = [("0.0", "-", (None, None), 1.6), ("0.30", "--", (5, 2), 1.7),
        ("0.0", "-", (None, None), 2.6), ("0.45", ":", (1, 1), 1.8)]
# colorblind-safe (Okabe-Ito): c=0.8 blue, 1.0 green, 1.04 vermillion (bold, main), 1.2 purple
COLOR = [("#0072B2", "-", (None, None), 1.8), ("#009E73", "--", (5, 2), 1.9),
         ("#D55E00", "-", (None, None), 2.6), ("#CC79A7", ":", (1, 1), 2.0)]
make(MONO, "")
make(COLOR, "_color")
