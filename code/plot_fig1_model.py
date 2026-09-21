"""
Figure 1: the actively powered simplest walking model.

Redrawn because the previous version mixed two instants in one picture. It showed
a swing-phase configuration - trailing foot in the air - while carrying the
heel-strike reset equation above it and a push-off arrow at the stance foot, and
alpha, which the equation uses, appeared nowhere in the drawing.

The two are now separated:
  main panel  a representative swing-phase configuration: continuous dynamics,
              theta, phi, the hip spring, the hip mass, gravity, walking direction.
              No reset equation and no push-off, because neither acts here.
  inset       the heel-strike instant: both feet on the ground, the impulsive
              push-off P along the trailing leg, the half inter-leg angle alpha,
              and the reset itself.

There is no colour version; the schematic is grayscale and build_pretypeset.py maps
both the mono and colour Figure 1 to this file.

Writes figures/figure1_model.{png,pdf,tiff}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))

BLACK, GREY, LIGHT = "#111111", "#555555", "#999999"


def spring(ax, p0, p1, coils=6, amp=0.035, lw=1.3, color=BLACK):
    """Zig-zag between two points, for the hip spring."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.hypot(*d)
    u = d / L
    n = np.array([-u[1], u[0]])
    ts = np.linspace(0, 1, 4 * coils + 1)
    pts = [p0 + t * d + n * amp * (0 if i in (0, len(ts) - 1) else (1 if i % 2 else -1))
           for i, t in enumerate(ts)]
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=lw, solid_joinstyle="miter",
            zorder=4)


def ground(ax, x0, x1, y=0.0, n=26, h=0.045, lw=1.0):
    ax.plot([x0, x1], [y, y], "-", color=BLACK, lw=1.6, zorder=3)
    for x in np.linspace(x0, x1 - (x1 - x0) / n, n):
        ax.plot([x, x - h], [y, y - h], "-", color=GREY, lw=lw, zorder=2)


fig = plt.figure(figsize=(7.6, 4.5))
ax = fig.add_axes([0.02, 0.04, 0.64, 0.92])
ax.set_aspect("equal"); ax.axis("off")

# ---------------- main panel: a swing-phase configuration ----------------
F1 = np.array([0.00, 0.00])          # stance foot, on the ground
TH = 0.165                            # stance-leg angle from the vertical (hip ahead)
H = F1 + np.array([np.sin(TH), np.cos(TH)])
PHI = 0.60                            # inter-leg angle at the hip
TH_SW = TH + PHI                      # swing leg is AHEAD of the stance leg
F2 = H + np.array([np.sin(TH_SW), -np.cos(TH_SW)])

ground(ax, -0.75, 1.35)
ax.plot([H[0], H[0]], [H[1] + 0.13, -0.02], ":", color=LIGHT, lw=1.1, zorder=1)

ax.plot([F1[0], H[0]], [F1[1], H[1]], "-", color=BLACK, lw=3.0,
        solid_capstyle="round", zorder=5)                      # stance leg
ax.plot([H[0], F2[0]], [H[1], F2[1]], "--", color=GREY, lw=2.6,
        dashes=(5, 3), zorder=5)                               # swing leg
ax.plot(*F1, "o", color="white", mec=BLACK, mew=1.8, ms=9, zorder=6)
ax.plot(*F2, "o", color="white", mec=GREY, mew=1.8, ms=9, zorder=6)
ax.plot(*H, "o", color=BLACK, ms=16, zorder=7)
ax.text(H[0] - 0.30, H[1] + 0.02, r"$M$", fontsize=14, style="italic")

# hip spring, between the two legs just below the hip
p_st, p_sw = H + (F1 - H) * 0.26, H + (F2 - H) * 0.26
spring(ax, p_st, p_sw, coils=5, amp=0.028)
mid = 0.5 * (p_st + p_sw)
ax.annotate(r"$k_{\mathrm{hip}}$", xy=mid, xytext=(mid[0] + 0.34, mid[1] + 0.30),
            fontsize=13, color=BLACK,
            arrowprops=dict(arrowstyle="-", lw=0.9, color=GREY))

a_st = np.degrees(np.arctan2(*(F1 - H)[::-1])) % 360      # hip -> stance foot
a_sw = np.degrees(np.arctan2(*(F2 - H)[::-1])) % 360      # hip -> swing foot
ax.add_patch(Arc(H, 0.86, 0.86, angle=0, theta1=a_st, theta2=270,
                 color=BLACK, lw=1.1, zorder=4))
ax.text(H[0] - 0.20, H[1] - 0.48, r"$\theta$", fontsize=14, style="italic")
ax.add_patch(Arc(H, 1.30, 1.30, angle=0, theta1=a_st, theta2=a_sw,
                 color=GREY, lw=1.1, zorder=4))
ax.text(H[0] + 0.30, H[1] - 0.70, r"$\varphi$", fontsize=14, style="italic", color=GREY)

ax.annotate("", xy=(-0.55, 0.32), xytext=(-0.55, 0.86),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color=BLACK))
ax.text(-0.70, 0.55, r"$g$", fontsize=14, style="italic")
ax.annotate("", xy=(1.28, -0.26), xytext=(0.84, -0.26),
            arrowprops=dict(arrowstyle="-|>", lw=1.4, color=GREY))
ax.text(0.80, -0.40, "walking direction", fontsize=9.5, color=GREY, style="italic")

ax.set_xlim(-0.80, 1.42); ax.set_ylim(-0.50, 1.30)
ax.set_title("Swing phase: continuous dynamics", fontsize=11, loc="left", x=0.02)

# ---------------- inset: the heel-strike instant ----------------
ax2 = fig.add_axes([0.655, 0.20, 0.335, 0.72])
ax2.set_aspect("equal"); ax2.axis("off")
for sp in ("left", "right", "top", "bottom"):
    pass
ax2.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax2.transAxes, fill=False,
                            edgecolor=LIGHT, lw=1.0, zorder=0))

A = 0.42                                   # half inter-leg angle, drawn large
Hi = np.array([0.0, np.cos(A)])
Ft = Hi + np.array([-np.sin(A), -np.cos(A)])   # trailing foot
Fl = Hi + np.array([+np.sin(A), -np.cos(A)])   # leading foot
ground(ax2, -0.95, 0.95, n=16, h=0.05)
ax2.plot([Ft[0], Hi[0]], [Ft[1], Hi[1]], "-", color=BLACK, lw=2.6, zorder=5)
ax2.plot([Hi[0], Fl[0]], [Hi[1], Fl[1]], "-", color=GREY, lw=2.6, zorder=5)
ax2.plot(*Hi, "o", color=BLACK, ms=11, zorder=7)
for F, c in ((Ft, BLACK), (Fl, GREY)):
    ax2.plot(*F, "o", color="white", mec=c, mew=1.6, ms=7, zorder=6)
ax2.plot([Hi[0], Hi[0]], [Hi[1] + 0.10, 0], ":", color=LIGHT, lw=1.0)

ax2.add_patch(Arc(Hi, 0.50, 0.50, angle=0, theta1=270 - np.degrees(A),
                  theta2=270 + np.degrees(A), color=BLACK, lw=1.0))
ax2.text(Hi[0] - 0.055, Hi[1] - 0.40, r"$2\alpha$", fontsize=12, style="italic")

u = (Hi - Ft) / np.hypot(*(Hi - Ft))
n = np.array([-u[1], u[0]])
ax2.annotate("", xy=Ft + u * 0.42, xytext=Ft + u * 0.04,
             arrowprops=dict(arrowstyle="-|>", lw=2.2, color=BLACK))
lab = Ft + u * 0.24 - n * 0.20
ax2.text(lab[0], lab[1], r"$P$", fontsize=13, style="italic", ha="center", va="center")

ax2.text(0.0, -0.42, r"$\dot{\theta}^{+}=\dot{\theta}^{-}\cos 2\alpha-P\sin 2\alpha$",
         fontsize=10.5, ha="center")
ax2.text(0.0, -0.60, r"$s=2\sin\alpha$", fontsize=10, ha="center", color=GREY)
ax2.set_xlim(-1.0, 1.0); ax2.set_ylim(-0.72, 1.20)
ax2.set_title("Heel strike: impulsive reset", fontsize=10, loc="left", x=0.03)

base = os.path.join(FIGS, "figure1_model")
for ext in ("png", "pdf", "tiff"):
    fig.savefig(base + "." + ext, dpi=300 if ext in ("png", "tiff") else None,
                facecolor="white")
plt.close(fig)
print("wrote", base + ".{png,pdf,tiff}")
