"""
Figure 2 (combined a,b), from the EXTENDED family: no fold encountered along the
branch traced; |lambda_max| rises toward 1 and N_1/2 rises steeply toward the slow
end of the computed range.
  (a) |lambda_max| (solid, left) and N_1/2 on a LOG axis (dashed, right) vs v.
  (b) NONLINEAR return-map recovery at a low-speed gait (real, slow monotonic) and
      a plateau gait (complex, oscillatory ring-down), read from
      data/fig2b_traces.csv, which run_fig2b_traces.py generates by iterating the
      map. Panel (b) previously plotted the closed-form rho^n cos(n phi) instead,
      which contradicted Methods 3.3.

Renders TWO coexisting versions:
  figure2_combined.png        grayscale (unchanged)
  figure2_combined_color.png  colorblind-safe (Okabe-Ito): blue |lambda_max|/low-speed, vermillion N_1/2/plateau
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import revision_numerics as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EXT = os.path.join(ROOT, "data", "master_fixed_khip_extended.csv")
OUT = os.path.join(ROOT, "figures", "figure2_combined.png")
TRACES = os.path.join(ROOT, "data", "fig2b_traces.csv")


def panel_a(ax1, P):
    d = np.genfromtxt(EXT, delimiter=",", names=True, dtype=None, encoding="utf-8")
    keep = (d["v"] >= d["v"].min() - 1e-12) & (d["v"] <= d["v"].max() + 1e-12)
    d = d[keep]
    d = d[np.argsort(d["v"])]
    v, lam = d["v"], d["lam_max"]
    half = -np.log(2.0) / np.log(np.clip(lam, 1e-12, 0.9999999))
    is_cx = (d["lam1_im"] != 0) | (d["lam2_im"] != 0)
    vstar = float(v[is_cx].min())

    ax1.plot(v, lam, "-", color=P["lam"], lw=2.2)
    ax1.set_xlabel(r"Dimensionless speed  $v$")
    ax1.set_ylabel(r"Spectral radius  $|\lambda_{\max}|$", color=P["lam"])
    ax1.set_ylim(0.5, 1.02)
    ax1.set_xlim(0, 0.235)
    ax1.axvline(vstar, color="0.6", ls=":", lw=1.0)
    ax1.tick_params(axis="y", labelcolor=P["lam"])

    ax2 = ax1.twinx()
    ax2.plot(v, half, "--", color=P["half"], lw=2.0, dashes=(5, 3))
    ax2.set_yscale("log")
    ax2.set_ylim(1, 1100)
    ax2.set_ylabel(r"Perturbation half-life  $N_{1/2}$ (steps, log)", color=P["half"])
    ax2.tick_params(axis="y", labelcolor=P["half"])

    ax1.annotate(r"$|\lambda_{\max}|\to 1$, $N_{1/2}$ grows steeply"
                 + "\ntoward the slow end\n(no fold encountered along the branch traced)",
                 xy=(0.006, 0.985), xytext=(0.045, 0.90), fontsize=8, color="0.25",
                 ha="left", arrowprops=dict(arrowstyle="->", lw=0.8, color="0.5"))
    ax1.annotate("stable-recovery plateau", xy=(0.15, 0.74), fontsize=8, ha="center", color="0.3")
    ax1.set_title("(a)  Recovery weakens toward the slow end of the computed branch",
                  loc="left", fontsize=11, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)


def dominant_mode(s):
    g = R.solve_gait(s)
    lam = sorted(g["lam"], key=lambda L: -abs(L))[0]
    return g["v"], complex(lam)


def panel_b(axB, P):
    """Nonlinear return-map traces, read from data/fig2b_traces.csv."""
    cLow, cPlat = P["low"], P["plat"]
    d = np.genfromtxt(TRACES, delimiter=",", names=True, dtype=None, encoding="utf-8")
    sel = {g: d[d["gait"] == g] for g in ("low speed", "plateau")}

    lam = {}
    for g, sub in sel.items():
        sub.sort(order="n")
        lam[g] = float(sub["envelope"][1])          # envelope[1] = |lambda_max|
    nhw = np.log(2) / -np.log(lam["low speed"])
    nhp = np.log(2) / -np.log(lam["plateau"])
    N = int(sel["low speed"]["n"].max())

    axB.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    axB.axhline(0.5, color="0.8", lw=0.9, zorder=0)

    nf = np.linspace(0, N, 600)
    env_p = lam["plateau"] ** nf
    axB.plot(nf, env_p, ":", color="0.6", lw=1.0, zorder=1)
    axB.plot(nf, -env_p, ":", color="0.6", lw=1.0, zorder=1)

    w, p = sel["low speed"], sel["plateau"]
    axB.plot(w["n"], w["dev_signed"], "-o", color=cLow, ms=4.2, lw=1.8, zorder=4,
             label=f"low speed  $v=0.023$   ($|\\lambda|={lam['low speed']:.2f}$, real)")
    axB.plot(p["n"], p["dev_signed"], "--s", color=cPlat, ms=3.8, lw=1.6,
             dashes=(5, 3), mfc=cPlat, zorder=3,
             label=f"plateau  $v=0.116$   ($|\\lambda|={lam['plateau']:.2f}$, complex)")

    axB.plot([nhw, nhw], [0, 0.5], ":", color=cLow, lw=1.2, zorder=2)
    axB.plot([nhp, nhp], [0, lam["plateau"] ** nhp], ":", color=cPlat, lw=1.2, zorder=2)
    axB.annotate(f"$N_{{1/2}}\\approx{nhw:.0f}$ steps", xy=(nhw, 0.5),
                 xytext=(nhw + 2.0, 0.70), fontsize=9.5, color=cLow,
                 arrowprops=dict(arrowstyle="-|>", lw=0.9, color=cLow))
    axB.annotate(f"envelope $N_{{1/2}}\\approx{nhp:.0f}$", xy=(nhp, lam["plateau"] ** nhp),
                 xytext=(nhp + 2.6, 0.20), fontsize=9.5, color=cPlat,
                 arrowprops=dict(arrowstyle="-|>", lw=0.9, color=cPlat))
    axB.text(N - 0.3, 0.52, "half of initial envelope", ha="right", va="bottom",
             fontsize=8, color="0.5")
    axB.text(0.30 * N, -0.30, "plateau recovery oscillates as it decays",
             fontsize=8.5, color="0.35", ha="left", va="center", style="italic")
    axB.set_xlabel("Step number after perturbation  $n$")
    axB.set_ylabel("Normalized stance-angle deviation")
    axB.set_xlim(0, N)
    axB.set_ylim(-0.5, 1.05)
    axB.set_title("(b)  Perturbation recovery is much slower at low speed",
                  loc="left", fontsize=11, fontweight="bold")
    axB.legend(fontsize=8.8, frameon=False, loc="upper right")
    axB.spines[["top", "right"]].set_visible(False)


def main(P, suffix):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    panel_a(axA, P)
    panel_b(axB, P)
    fig.tight_layout(w_pad=4.0)
    out = OUT.replace(".png", suffix + ".png")
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace(".png", ".tiff"), dpi=300)
    # vector copy for submission: Elsevier asks 1000 dpi for bitmapped line
    # drawings, which vector output makes moot.
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 11})
    main(dict(lam="#111111", half="#7a7a7a", low="black", plat="0.45"), "")            # grayscale
    main(dict(lam="#0072B2", half="#D55E00", low="#0072B2", plat="#D55E00"), "_color")  # colorblind-safe
