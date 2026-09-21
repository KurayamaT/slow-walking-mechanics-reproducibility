"""
Low-speed scaling of the recovery margin.

Descriptive fits, not statistical tests: the continuation points are a dense
deterministic sequence, not independent samples, so R^2 reports goodness of fit
and nothing more. The exponent drifts with the fitting range, which is the point
worth recording -- the quadratic behaviour is local to the slow end, not a law
for the whole branch.

Writes data/lowspeed_scaling.csv and the main Figure 4,
figures/figure4_scaling{,_color}.{png,pdf,tiff}.

This figure was Supplementary Figure S5 until the low-step-length scaling became the
third headline result; Supplementary S5 and Table S3 remain as the fit-range
sensitivity, and now point at Figure 4.
"""
import os
import csv
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", os.environ.get("SLOWWALK_DATA_DIR", "data")))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))

rows = list(csv.DictReader(open(os.path.join(DATA, "master_fixed_khip_extended.csv"),
                                encoding="utf-8-sig")))
for r in rows:
    r["s"] = float(r["s"]); r["v"] = float(r["v"]); r["lam"] = float(r["lam_max"])
rows = [r for r in rows if r["lam"] < 1.0]
rows.sort(key=lambda r: r["s"])
s = np.array([r["s"] for r in rows])
lam = np.array([r["lam"] for r in rows])
marg = 1.0 - lam
nhalf = -np.log(2.0) / np.log(lam)

print("exponent depends on the fitting range -- the quadratic form is local to the slow end")
out = []
for smax in (0.03, 0.05, 0.10, 0.15, 0.30, 0.80):
    m = s <= smax
    if m.sum() < 5:
        continue
    sl, ic = np.polyfit(np.log(s[m]), np.log(marg[m]), 1)
    pred = sl * np.log(s[m]) + ic
    r2 = 1.0 - np.sum((np.log(marg[m]) - pred) ** 2) / np.sum((np.log(marg[m]) - np.log(marg[m]).mean()) ** 2)
    print("  s <= %.2f  n=%4d  exponent=%.4f  coefficient=%.4f  R2=%.6f"
          % (smax, m.sum(), sl, math.exp(ic), r2))
    out.append(dict(s_max=smax, n=int(m.sum()), exponent=sl,
                    coefficient=math.exp(ic), r_squared=r2))
with open(os.path.join(DATA, "lowspeed_scaling.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

ratio = marg / s ** 2
print("\n(1-rho)/s^2 at the slow end: %.4f at s=%.3f, %.4f at s=%.3f"
      % (ratio[0], s[0], ratio[10], s[10]))
print("N_half * s^2               : %.5f at s=%.3f" % ((nhalf * s ** 2)[0], s[0]))
print("collision deficit (1-cos2a)/s^2 is exactly 0.5, so the ratio of coefficients is %.2f"
      % (ratio[0] / 0.5))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fit actually drawn in panel (a): the s <= 0.05 descriptive fit
m05 = s <= 0.05
sl05, ic05 = np.polyfit(np.log(s[m05]), np.log(marg[m05]), 1)
C05 = math.exp(ic05)


def draw(P, suffix):
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9))

    # (a) whole branch, log-log: margin, the s<=0.05 fit, and the collision deficit
    ax[0].loglog(s, marg, "-", color=P["data"], lw=1.7, label=r"$1-|\lambda_{\max}|$")
    ax[0].loglog(s[m05], C05 * s[m05] ** sl05, "--", color=P["fit"], lw=1.6,
                 label=r"fit for $s\leq 0.05$: $1-|\lambda_{\max}|=%.2f\,s^{%.3f}$" % (C05, sl05))
    ax[0].loglog(s, s ** 2 / 2, ":", color=P["coll"], lw=1.5,
                 label=r"collision deficit $s^{2}/2$")
    ax[0].set_xlabel("step length $s$"); ax[0].set_ylabel(r"$1-|\lambda_{\max}|$")
    ax[0].legend(fontsize=8, frameon=False, loc="lower right")
    ax[0].grid(alpha=0.25, which="both")
    ax[0].set_title("(a) both quadratic at low $s$, with different coefficients",
                    fontsize=9.5, loc="left")

    # (b) slow end only, linear: the coefficient itself against the exact 0.5
    lo = s <= 0.05
    ax[1].plot(s[lo], ratio[lo], "-", color=P["data"], lw=1.8,
               label=r"$(1-|\lambda_{\max}|)/s^{2}$")
    ax[1].axhline(0.5, color=P["coll"], lw=1.3, ls=":")
    ax[1].text(0.0125, 1.1, r"$(1-\cos 2\alpha)/s^{2}=0.5$ exactly",
               fontsize=8.5, color=P["coll"])
    ax[1].annotate(r"$\approx 8.23$ at the slow end", (s[0], ratio[0]),
                   textcoords="offset points", xytext=(16, -26), fontsize=8.5,
                   color=P["data"],
                   arrowprops=dict(arrowstyle="-", lw=0.8, color=P["data"]))
    ax[1].set_xlim(0.01, 0.05); ax[1].set_ylim(0, 10)
    ax[1].set_xlabel("step length $s$"); ax[1].set_ylabel(r"$(1-|\lambda_{\max}|)/s^{2}$")
    ax[1].legend(fontsize=8, frameon=False, loc="upper right")
    ax[1].grid(alpha=0.25)
    ax[1].set_title("(b) the coefficient against the exact collision value",
                    fontsize=9.5, loc="left")

    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    base = os.path.join(FIGS, "figure4_scaling" + suffix)
    for ext in ("png", "pdf", "tiff"):
        fig.savefig(base + "." + ext, dpi=300 if ext in ("png", "tiff") else None)
    plt.close(fig)
    print("wrote", base + ".{png,pdf,tiff}")


draw(dict(data="#111111", fit="#7a7a7a", coll="0.55"), "")
draw(dict(data="#0072B2", fit="#D55E00", coll="#009E73"), "_color")
