"""
Fig 3 — the dominant non-zero Floquet multiplier changes TYPE at the transition
speed v* (two positive real multipliers below v*, a complex-conjugate pair above).

Renders TWO coexisting versions:
  fig_eigtype.png        grayscale (unchanged)
  fig_eigtype_color.png  colorblind-safe (Okabe-Ito): blue Re, orange +/-Im, black |lambda_max|

Reads: newest/data/master_fixed_khip_extended.csv
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV = os.path.join(ROOT, "data", "master_fixed_khip_extended.csv")
OUT = os.path.join(ROOT, "figures", "fig_eigtype.png")

d = np.genfromtxt(CSV, delimiter=",", names=True, dtype=None, encoding="utf-8")
keep = (d["v"] >= d["v"].min() - 1e-12) & (d["v"] <= d["v"].max() + 1e-12)
d = d[keep]
d = d[np.argsort(d["v"])]
v = d["v"]
l1r, l1i = d["lam1_re"], d["lam1_im"]
l2r, l2i = d["lam2_re"], d["lam2_im"]
lammax = d["lam_max"]

is_complex = (l1i != 0) | (l2i != 0)
vstar = float(v[is_complex].min()) if is_complex.any() else float("nan")
print("v* =", vstar)

plt.rcParams.update({"font.size": 11})


def make(cRe, cIm, cMax, suffix):
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(vstar, color="0.55", ls=":", lw=1.2, zorder=1)
    # real parts of the two non-zero multipliers
    ax.plot(v, l1r, "-", color=cRe, lw=1.6, zorder=3, label=r"$\mathrm{Re}\,\lambda_{1,2}$")
    ax.plot(v, l2r, "-", color=cRe, lw=1.6, zorder=3)
    # imaginary parts +/- (dashed; zero below v*)
    ax.plot(v, l1i, "--", color=cIm, lw=1.5, dashes=(5, 3), zorder=2, label=r"$\pm\,\mathrm{Im}\,\lambda$")
    ax.plot(v, l2i, "--", color=cIm, lw=1.5, dashes=(5, 3), zorder=2)
    # dominant magnitude (bold)
    ax.plot(v, lammax, "-", color=cMax, lw=2.4, zorder=5, label=r"$|\lambda_{\max}|$")

    ax.annotate(r"transition speed $v^*\approx%.3f$" % vstar, xy=(vstar, -0.34),
                xytext=(vstar + 0.018, -0.5), fontsize=9, color="0.25",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="0.5"))
    ax.text(vstar * 0.5, 1.08, "below $v^*$:\ntwo positive real\n(monotonic recovery)",
            fontsize=8.3, ha="center", va="top", color="0.3")
    ax.text((vstar + 0.231) / 2.0, 1.08, "above $v^*$:\ncomplex-conjugate pair\n(oscillatory recovery)",
            fontsize=8.3, ha="center", va="top", color="0.3")
    ax.set_xlabel(r"Dimensionless speed  $v$")
    ax.set_ylabel(r"Floquet multiplier")
    ax.set_xlim(0.0, 0.235)
    ax.set_ylim(-0.65, 1.18)
    ax.legend(fontsize=8.8, frameon=False, loc="center right")
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


make("0.45", "0.62", "#111111", "")            # grayscale (unchanged)
make("#0072B2", "#E69F00", "#000000", "_color")  # colorblind-safe
