"""
Supplementary S4 - robustness of the low-speed weakening, consolidated into one
four-panel figure so that the hip-spring, foot-mass and push-off sweeps are not
three separate supplementary items.

  (a) each k_hip branch's slowest computed gait in the (v, N_1/2) plane, against
      the main k_hip = -0.16 branch as reference
  (b) the transition speed v* against k_hip
  (c) half-life at s = 0.040, the lowest step length common to every foot mass
      tested, against beta; the three beta values at which no orbitally stable
      gait was found are marked individually, not shaded as a region
  (d) half-life against speed for the four push-off coefficients

Reads only precomputed summaries and traces - no solver work:
  data/sens_khip_summary.csv        (MATLAB main_khip_sweep.m)
  data/sens_beta_summary.csv        (MATLAB main_beta_sweep.m)
  data/supp_csweep.csv              (run_csweep_extended.py)
  data/master_fixed_khip_extended.csv

Deliberately NOT plotted: every curve of every coefficient. The summaries carry
the claim the main text makes - a steep rise toward the slow end for each
coefficient - without presenting nine or four full families as evidence.

Writes figures/fig_supp_robustness.{png,pdf}
Run from code/:  python run_supp_robustness.py
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DATA = os.path.join(ROOT, os.environ.get("SLOWWALK_DATA_DIR", "data"))
FIGS = os.path.join(ROOT, "figures")


def read(name):
    with open(os.path.join(DATA, name)) as fh:
        return list(csv.DictReader(fh))


def f(row, key):
    v = row[key].strip()
    return float("nan") if v in ("", "NaN", "nan") else float(v)


khip = read("sens_khip_summary.csv")
beta = read("sens_beta_summary.csv")
main = read("master_fixed_khip_extended.csv")
cs_path = os.path.join(DATA, "supp_csweep.csv")
csw = read("supp_csweep.csv") if os.path.exists(cs_path) else None
ksw = read("supp_khip_sweep.csv")

plt.rcParams.update({"font.size": 9.5})
fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0))

# ---- (a) k_hip: half-life at the common slowest step length ----
# Every branch is now traced by the same seeded continuation to s = 0.010, so the
# comparison is at one step length rather than at wherever each solver stalled.
ax = axes[0, 0]
ks = np.array(sorted({f(r, "k_hip") for r in ksw}))
n_at_smin, v_at_smin = [], []
for k in ks:
    rows = [r for r in ksw if abs(f(r, "k_hip") - k) < 1e-9 and abs(f(r, "s") - 0.010) < 1e-9]
    n_at_smin.append(f(rows[0], "N_half"))
    v_at_smin.append(f(rows[0], "v"))
n_at_smin = np.array(n_at_smin)
ax.semilogy(ks, n_at_smin, "o-", color="0.0", ms=5, lw=1.4,
            label=r"$N_{1/2}$ at $s=0.010$, each branch")
ax.axvline(-0.16, color="0.6", ls=":", lw=1.1)
i16 = int(np.argmin(np.abs(ks + 0.16)))
ax.annotate(r"main branch, $N_{1/2}=842.2$", (ks[i16], n_at_smin[i16]),
            textcoords="offset points", xytext=(12, 14), fontsize=8, color="0.3")
ax.set_xlabel(r"$k_{\mathrm{hip}}$")
ax.set_ylabel(r"$N_{1/2}$ at $s=0.010$ (steps)")
ax.set_ylim(300, 1600)
ax.set_title(r"(a) hip-spring coefficient", loc="left", fontsize=10)
ax.legend(fontsize=7.6, frameon=False, loc="upper right")
ax.grid(True, lw=0.4, color="0.92")

# ---- (b) transition speed against k_hip ----
ax = axes[0, 1]
# The transition is bracketed by adjacent grid points, not resolved to a value:
# take the midpoint of the bracket and show its width.
mid, halfw = [], []
for k in ks:
    br = [r for r in ksw if abs(f(r, "k_hip") - k) < 1e-9]
    cx = [f(r, "v") for r in br if f(r, "lam1_im") != 0.0]
    re_ = [f(r, "v") for r in br if f(r, "lam1_im") == 0.0]
    lo = max([v for v in re_ if v < min(cx)]) if cx and re_ else float("nan")
    hi = min(cx) if cx else float("nan")
    mid.append(0.5 * (lo + hi)); halfw.append(0.5 * (hi - lo))
mid = np.array(mid); halfw = np.array(halfw)
ax.errorbar(ks, mid, yerr=halfw, fmt="s-", color="0.0", ms=5, lw=1.4,
            elinewidth=1.0, capsize=2.5)
ax.axvline(-0.16, color="0.6", ls=":", lw=1.1)
ax.annotate("main branch", (-0.16, np.nanmax(mid)), textcoords="offset points",
            xytext=(6, -4), fontsize=8, color="0.35")
ax.set_xlabel(r"$k_{\mathrm{hip}}$")
ax.set_ylabel(r"transition speed $v^{*}$")
ax.set_title(r"(b) where the mode switch occurs", loc="left", fontsize=10)
ax.grid(True, lw=0.4, color="0.92")

# ---- (c) foot mass ----
ax = axes[1, 0]
bb = np.array([f(r, "beta") for r in beta])
bn = np.array([f(r, "Nhalf_max_stable") for r in beta])
ok = ~np.isnan(bn)
ax.plot(bb[ok], bn[ok], "^-", color="0.0", ms=6, lw=1.4,
        label="$N_{1/2}$ at $s=0.040$ (lowest common step length)")
# beta >= 0.20 was sampled at three values only. Plotting them would put a marker
# at a vertical position with no meaning, so they are stated instead.
ax.text(0.29, 150, "No orbitally stable gait was found\non the tested grid at\n"
        r"$\beta=0.20,\ 0.25,\ 0.28$.",
        fontsize=8, color="0.3", ha="right", va="center")
ax.set_xlim(0, 0.30)
ax.set_ylim(0, 260)
ax.set_xlabel(r"foot mass $\beta$")
ax.set_ylabel(r"$N_{1/2}$ (steps)")
ax.set_title(r"(c) small finite foot mass", loc="left", fontsize=10)
ax.legend(fontsize=7.8, frameon=False, loc="upper left")
ax.grid(True, lw=0.4, color="0.92")

# ---- (d) push-off coefficient ----
ax = axes[1, 1]
if csw is None:
    ax.text(0.5, 0.5, "supp_csweep.csv not yet written\n(run run_csweep_extended.py)",
            ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.4")
else:
    STY = {0.80: ("0.0", "-", 1.3), 1.00: ("0.35", "--", 1.4),
           1.04: ("0.0", "-", 2.4), 1.20: ("0.5", ":", 1.7)}
    for c in (0.80, 1.00, 1.04, 1.20):
        rs = [r for r in csw if abs(float(r["c"]) - c) < 1e-9]
        v = np.array([float(r["v"]) for r in rs])
        n = np.array([float(r["N_half"]) for r in rs])
        o = np.argsort(v)
        col, ls, lw = STY[c]
        ax.semilogy(v[o], n[o], ls, color=col, lw=lw,
                    label="$c=%.2f$%s" % (c, "  (main)" if c == 1.04 else ""))
    ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.set_xlim(0, 0.235)
ax.set_ylim(1, 1500)
ax.set_xlabel(r"dimensionless speed $v$")
ax.set_ylabel(r"$N_{1/2}$ (steps)")
ax.set_title(r"(d) push-off coefficient in $P(s)=c\,\alpha\tan\alpha$",
             loc="left", fontsize=10)
ax.grid(True, lw=0.4, color="0.92")

fig.tight_layout()
out = os.path.join(FIGS, "fig_supp_robustness.png")
fig.savefig(out, dpi=300)
fig.savefig(out.replace(".png", ".pdf"))
plt.close(fig)
print("saved:", out, "(panel d from CSV)" if csw else "(panel d PENDING)")
