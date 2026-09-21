#!/usr/bin/env python3
"""
revision_numerics.py — numerical analyses requested at major revision for
"Perturbation half-life at low speed in an actively powered simplest walking
model" (fixed k_hip = -0.16, level ground, prescribed push-off branch).

This is a faithful Python port of code/floquet/floquet_KUO_v2.m (the MATLAB
numerical core that produced results/csv/floquet_b0_k-0.160.csv and
new article/data/master_fixed_khip.csv).  It is used ONLY to compute the
revision items; it does not modify any existing code/data.

Analyses
  VALIDATE : reproduce |lambda_max|, eig type, T from master CSV (port check)
  C3       : reduced-2D vs full-3D return-map multiplier agreement; lambda3~0
  C4a      : epsilon-convergence of the central-difference Jacobian multipliers
  C4b      : Newton-convergence / branch-termination classification over s
  C6       : cos(2 alpha) vs |lambda_max| correlation (post-processing of CSV)

Model (dimensionless, leg length L=1, g=1):
  state on post-impact beginning-of-step section  z = [theta, theta_dot, phi_dot]
  full ODE state                                  y = [theta, theta_dot, phi, phi_dot]
  swing EOM (gam = ground slope, k = hip-spring coefficient):
     theta''  = sin(theta - gam)
     phi''    = sin(theta - gam) + sin(phi)*(theta_dot^2 - cos(theta - gam)) + k*phi
  heel-strike (phi = 2 theta) + impulsive push-off P:
     theta+      = -theta-
     theta_dot+  = cos(2a) theta_dot- + P sin(2a)        (a = theta- at impact, so a < 0;
                                                         equivalently -P sin(2 alpha_h))
     phi_dot+    = (1 - cos(2a)) theta_dot+
  prescribed push-off on this branch:  P(q) = 1.04 * alpha_p * tan(alpha_p),
     with alpha_p = arcsin(q/2); the REALISED heel-strike half-angle is
     alpha_h = theta+ and the realised step length is s = 2 sin(alpha_h),
     alpha = asin(s/2).
"""
import os
import sys
import numpy as np
from numpy.linalg import eig, solve as lsolve, cond
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
NEWART = os.path.dirname(HERE)
MASTER = os.path.join(NEWART, os.environ.get("SLOWWALK_DATA_DIR", "data"), "master_fixed_khip.csv")

GAM = 0.0          # level ground
KHIP = -0.16       # main analysis branch
PER = 5.0          # max step duration
DT_DEPART = 0.005  # phase-1 departure time
RTOL = 1e-12       # tight ODE tolerance (Jacobian / fixed point)
ATOL = 1e-14
DELTA = 1e-7       # central-difference step for Jacobian


# ----------------------------------------------------------------------- EOM
def eom(t, y, gam, k):
    th, thd, ph, phd = y
    return [thd,
            np.sin(th - gam),
            phd,
            np.sin(th - gam) + np.sin(ph) * (thd * thd - np.cos(th - gam)) + k * ph]


def _ev_collision(t, y, gam, k):
    return y[2] - 2.0 * y[0]          # phi - 2 theta = 0
_ev_collision.terminal = True
_ev_collision.direction = 1.0


def _ev_guard(t, y, gam, k):
    return np.pi / 2.0 - abs(y[0])    # divergence guard
_ev_guard.terminal = True
_ev_guard.direction = 0.0


# ------------------------------------------------------- 3-D step map R(z)
def step_map(z, gam=GAM, k=KHIP, P=0.0, rtol=RTOL, atol=ATOL):
    """Two-phase integration of one step. z=[theta,theta_dot,phi_dot]."""
    if abs(z[0]) > np.pi / 3.0:
        return None, None, False
    y0 = [z[0], z[1], 2.0 * z[0], z[2]]
    # Phase 1: depart the collision surface (no events)
    s1 = solve_ivp(eom, (0.0, DT_DEPART), y0, args=(gam, k),
                   method="RK45", rtol=1e-12, atol=1e-14, dense_output=False)
    if not s1.success:
        return None, None, False
    y_dep = s1.y[:, -1]
    # Phase 2: integrate to heel-strike
    s2 = solve_ivp(eom, (DT_DEPART, PER), y_dep, args=(gam, k),
                   method="RK45", rtol=rtol, atol=atol,
                   events=(_ev_collision, _ev_guard))
    if s2.t_events[0].size == 0:
        return None, None, False
    te = s2.t_events[0][0]
    yc = s2.y_events[0][0]
    c2 = np.cos(2.0 * yc[0])
    s2P = np.sin(2.0 * yc[0]) * P
    z_new = np.array([-yc[0],
                      c2 * yc[1] + s2P,
                      c2 * (1.0 - c2) * yc[1] + (1.0 - c2) * s2P])
    return z_new, te, True


# ----------------------------------------------- reduced 2-D map R2([th,thd])
def step_map_2d(u, gam=GAM, k=KHIP, P=0.0, rtol=RTOL, atol=ATOL):
    """2-D map on the post-impact constraint surface.
       u=[theta,theta_dot]; phi_dot reconstructed as (1-cos2theta)*theta_dot."""
    th, thd = u
    phd = (1.0 - np.cos(2.0 * th)) * thd
    z_new, te, ok = step_map([th, thd, phd], gam, k, P, rtol, atol)
    if not ok:
        return None, None, False
    return np.array([z_new[0], z_new[1]]), te, True


# -------------------------------------------------------- Newton fixed point
def find_fixed_point(z0, gam=GAM, k=KHIP, P=0.0, rtol=RTOL, atol=ATOL,
                     tol=1e-12, max_iter=20, delta=1e-7):
    z = np.array(z0, float)
    _, _, ok = step_map(z, gam, k, P, rtol, atol)
    if not ok:
        return None, None, False
    T = None
    for _ in range(max_iter):
        Sz, T, ok = step_map(z, gam, k, P, rtol, atol)
        if not ok:
            return None, None, False
        res = Sz - z
        if np.linalg.norm(res) < tol:
            return z, T, True
        Jg = np.zeros((3, 3))
        for j in range(3):
            zp = z.copy(); zp[j] += delta
            zm = z.copy(); zm[j] -= delta
            Sp, _, ok1 = step_map(zp, gam, k, P, rtol, atol)
            Sm, _, ok2 = step_map(zm, gam, k, P, rtol, atol)
            if not (ok1 and ok2):
                return None, None, False
            Jg[:, j] = (Sp - Sm) / (2.0 * delta)
        Jg = Jg - np.eye(3)
        if 1.0 / cond(Jg) < 1e-14:
            return None, None, False
        z = z + 0.8 * lsolve(Jg, -res)
    Sz, T, ok = step_map(z, gam, k, P, rtol, atol)
    if ok and np.linalg.norm(Sz - z) < 1e-9:
        return z, T, True
    return None, None, False


def jacobian_3d(z_fp, gam=GAM, k=KHIP, P=0.0, rtol=RTOL, atol=ATOL, delta=DELTA):
    J = np.zeros((3, 3))
    for j in range(3):
        zp = z_fp.copy(); zp[j] += delta
        zm = z_fp.copy(); zm[j] -= delta
        Sp, _, ok1 = step_map(zp, gam, k, P, rtol, atol)
        Sm, _, ok2 = step_map(zm, gam, k, P, rtol, atol)
        if not (ok1 and ok2):
            return None
        J[:, j] = (Sp - Sm) / (2.0 * delta)
    return J


def jacobian_2d(u_fp, gam=GAM, k=KHIP, P=0.0, rtol=RTOL, atol=ATOL, delta=DELTA):
    J = np.zeros((2, 2))
    for j in range(2):
        up = u_fp.copy(); up[j] += delta
        um = u_fp.copy(); um[j] -= delta
        Sp, _, ok1 = step_map_2d(up, gam, k, P, rtol, atol)
        Sm, _, ok2 = step_map_2d(um, gam, k, P, rtol, atol)
        if not (ok1 and ok2):
            return None
        J[:, j] = (Sp - Sm) / (2.0 * delta)
    return J


def sorted_eigs(J):
    lam = eig(J)[0]
    return lam[np.argsort(-np.abs(lam))]


def solve_gait_cont(s_target, s_start=0.080, ds=0.002, **kw):
    """Reach s_target by continuation from s_start (robust past the low-s fold)."""
    g = solve_gait(s_start, **kw)
    if g is None:
        return None
    s = s_start
    direction = -1.0 if s_target < s_start else 1.0
    while (direction < 0 and s > s_target + 1e-9) or \
          (direction > 0 and s < s_target - 1e-9):
        s = max(s_target, s - ds) if direction < 0 else min(s_target, s + ds)
        gnew = solve_gait(s, seed=g["z_fp"], **kw)
        if gnew is None:           # fold reached before target
            return g
        g = gnew
    return g


def solve_gait(q, gam=GAM, k=KHIP, rtol=RTOL, atol=ATOL, delta=DELTA, seed=None):
    """Full pipeline for one branch parameter q. Returns dict or None.

    q prescribes the push-off; it is NOT the step length. The fixed point
    realises a heel-strike half inter-leg angle alpha_h = z_fp[0], and the
    realised step length and speed are s = 2 sin(alpha_h) and v = s / T, which
    are what the manuscript reports. alpha_h / alpha_p is about 0.988 on the
    reference branch. The key "alpha" was deliberately removed so that a caller
    written against the old convention raises KeyError instead of silently
    mixing the two bases.

    seed: optional initial z guess (e.g. for continuation past a turning point).
    """
    alpha_p = np.arcsin(0.5 * q)
    omega = -1.04 * alpha_p
    P = -omega * np.tan(alpha_p)           # = 1.04*alpha_p*tan(alpha_p)
    if seed is None:
        z0 = [alpha_p, omega, (1.0 - np.cos(2.0 * alpha_p)) * omega]
    else:
        z0 = list(seed)
    z_fp, T, ok = find_fixed_point(z0, gam, k, P, rtol, atol, delta=1e-7)
    if not ok:
        return None
    J = jacobian_3d(z_fp, gam, k, P, rtol, atol, delta)
    if J is None:
        return None
    lam = sorted_eigs(J)
    lam_max = float(np.max(np.abs(lam)))
    alpha_h = float(z_fp[0])
    s = 2.0 * np.sin(alpha_h)
    return dict(q=q, alpha_p=alpha_p, P=P, alpha_h=alpha_h, s=s, v=s / T, T=T,
                z_fp=z_fp, lam=lam, lam_max=lam_max, J=J)


# --------------------------------------------------------------- utilities
def load_master():
    import csv
    rows = []
    with open(MASTER, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({key: r[key] for key in r})
    return rows


def fmt_lam(lam):
    out = []
    for L in lam:
        if abs(L.imag) < 1e-7:
            out.append(f"{L.real:+.4f}")
        else:
            out.append(f"{L.real:+.4f}{L.imag:+.4f}i")
    return ", ".join(out)


# =====================================================================
# ANALYSES
# =====================================================================
def run_validate():
    """Reproduce master CSV |lambda_max|, eig type, T at sampled s values."""
    master = load_master()
    s_all = np.array([float(r["s"]) for r in master])
    print(f"{'s':>7} {'v_csv':>8} {'|lam|_csv':>10} {'|lam|_py':>10} "
          f"{'dlam':>9} {'T_csv':>7} {'T_py':>7} {'type_csv':>9} {'type_py':>9}")
    test_s = [0.020, 0.040, 0.054, 0.080, 0.142, 0.200, 0.300, 0.402,
              0.500, 0.600, 0.700, 0.780]
    maxdev = 0.0
    for s in test_s:
        i = int(np.argmin(np.abs(s_all - s)))
        r = master[i]
        s_csv = float(r["s"])
        g = solve_gait(s_csv)
        if g is None:
            print(f"{s_csv:7.3f}   solve_gait FAILED")
            continue
        lam_csv = float(r["lam_max"])
        type_csv = r["eig_type"]
        type_py = "complex" if abs(g["lam"][0].imag) > 1e-6 else "real"
        dev = abs(g["lam_max"] - lam_csv)
        maxdev = max(maxdev, dev)
        print(f"{s_csv:7.3f} {float(r['v']):8.4f} {lam_csv:10.4f} "
              f"{g['lam_max']:10.4f} {dev:9.2e} {float(r['T']):7.3f} "
              f"{g['T']:7.3f} {type_csv:>9} {type_py:>9}")
    print(f"\nmax |lambda_max| deviation (py vs CSV) over sample = {maxdev:.2e}")


def run_c3():
    """Reduced-2D vs full-3D multiplier agreement; structural zero lambda3."""
    reps = [("slowest stable", 0.054), ("low-speed", 0.080),
            ("transition", 0.142), ("plateau", 0.402),
            ("fast", 0.600)]
    print(f"{'gait':>15} {'v':>7} | {'3D lam1,lam2 (nonzero)':>34} "
          f"{'|lam3| (struct.0)':>17} | {'2D lam1,lam2':>34} {'max|d|':>9}")
    worst = 0.0
    worst_l3 = 0.0
    for name, s in reps:
        g = solve_gait_cont(s) if name == "slowest stable" else solve_gait(s)
        if g is None:
            print(f"{name:>15} solve failed"); continue
        lam3d = g["lam"]
        l3 = float(np.abs(lam3d[2]))
        worst_l3 = max(worst_l3, l3)
        # 2D Jacobian at the same fixed point (theta, theta_dot)
        u_fp = np.array([g["z_fp"][0], g["z_fp"][1]])
        J2 = jacobian_2d(u_fp, P=g["P"])
        lam2d = sorted_eigs(J2)
        # compare the two non-zero 3D multipliers to the 2D multipliers
        a = np.sort_complex(lam3d[:2])
        b = np.sort_complex(lam2d)
        d = float(np.max(np.abs(a - b)))
        worst = max(worst, d)
        print(f"{name:>15} {g['v']:7.4f} | {fmt_lam(lam3d[:2]):>34} "
              f"{l3:17.2e} | {fmt_lam(lam2d):>34} {d:9.2e}")
    print(f"\nmax |2D - 3D(nonzero)| multiplier discrepancy = {worst:.2e}")
    print(f"max structural-zero |lambda3| over reps        = {worst_l3:.2e}")


def run_c4a():
    """Epsilon-convergence of the central-difference Jacobian multipliers."""
    reps = [("slowest stable", 0.054), ("low-speed", 0.080),
            ("transition", 0.142), ("plateau", 0.402)]
    deltas = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    for name, s in reps:
        # converge fixed point once (delta-independent), then vary Jacobian delta
        alpha = np.arcsin(0.5 * s)
        P = -(-1.04 * alpha) * np.tan(alpha)
        if name == "slowest stable":
            gg = solve_gait_cont(s)
            z_fp, T, ok = (gg["z_fp"], gg["T"], True) if gg else (None, None, False)
        else:
            z0 = [alpha, -1.04 * alpha, (1 - np.cos(2 * alpha)) * (-1.04 * alpha)]
            z_fp, T, ok = find_fixed_point(z0, P=P)
        if not ok:
            print(f"{name}: fixed point failed"); continue
        print(f"\n{name}  (s={s}, v={s/T:.4f})")
        print(f"  {'delta':>8} {'|lam_max|':>12} {'lam1':>26}")
        vals = []
        for dl in deltas:
            J = jacobian_3d(z_fp, P=P, delta=dl)
            lam = sorted_eigs(J)
            vals.append(np.max(np.abs(lam)))
            print(f"  {dl:8.0e} {np.max(np.abs(lam)):12.6f} {fmt_lam([lam[0]]):>26}")
        vals = np.array(vals)
        print(f"  spread over delta in [1e-9,1e-5] = {vals.max()-vals.min():.2e} "
              f"(rel {100*(vals.max()-vals.min())/vals.mean():.2e}%)")


def run_c4b(n=240):
    """Classify Newton convergence / stability / branch termination over s."""
    s_grid = np.linspace(0.01, 0.80, n)
    cls = []   # 'stable', 'unstable', 'fail'
    rows = []
    for s in s_grid:
        g = solve_gait(s)
        if g is None:
            cls.append("fail"); rows.append((s, np.nan, np.nan)); continue
        c = "stable" if g["lam_max"] < 1.0 else "unstable"
        cls.append(c)
        rows.append((s, g["v"], g["lam_max"]))
    cls = np.array(cls)
    stab = np.array([r for r, c in zip(rows, cls) if c == "stable"])
    print(f"grid: {n} step lengths in s=[0.01,0.80]")
    for c in ["stable", "unstable", "fail"]:
        print(f"  {c:>9}: {(cls==c).sum():4d}")
    if stab.size:
        v = stab[:, 1]; s = stab[:, 0]
        print(f"\nstable branch: s in [{s.min():.4f}, {s.max():.4f}], "
              f"v in [{v.min():.4f}, {v.max():.4f}]")
        print(f"  slowest stable: s={s[np.argmin(v)]:.4f} v={v.min():.4f} "
              f"|lam|={stab[np.argmin(v),2]:.4f}")
        print(f"  fastest stable: s={s[np.argmax(v)]:.4f} v={v.max():.4f} "
              f"|lam|={stab[np.argmax(v),2]:.4f}")
    # describe boundaries: what happens just below smin and just above smax
    print("\nboundary scan (transitions in classification, s ascending):")
    prev = None
    for (s, v, lm), c in zip(rows, cls):
        if c != prev:
            print(f"  s={s:.4f}  v={'nan' if v!=v else f'{v:.4f}'}  -> {c}")
            prev = c


def run_c6():
    """cos(2 alpha) vs |lambda_max| correlation from master CSV."""
    master = load_master()
    stab = [r for r in master if r["stable"] in ("1", "true", "True")]
    v = np.array([float(r["v"]) for r in stab])
    lam = np.array([float(r["lam_max"]) for r in stab])
    cos2a = np.array([float(r["cos2alpha"]) for r in stab])
    etype = np.array([r["eig_type"] for r in stab])
    # transition speed v*: scanning v ascending, first real->complex switch
    order = np.argsort(v)
    vs, ts = v[order], etype[order]
    vstar = np.nan
    for i in range(1, len(vs)):
        if ts[i] == "complex" and ts[i - 1] == "real":
            vstar = 0.5 * (vs[i] + vs[i - 1]); break

    def pear(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    def spear(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return pear(ra.astype(float), rb.astype(float))

    print(f"stable gaits: {len(stab)},  v* (eig-type switch) = {vstar:.4f}")
    for label, mask in [("whole stable branch", np.ones_like(v, bool)),
                        ("below v* (real-dominant)", v < vstar),
                        ("above v* (complex-dominant)", v >= vstar)]:
        if mask.sum() < 3:
            continue
        print(f"\n{label}  (n={mask.sum()}, "
              f"v in [{v[mask].min():.4f},{v[mask].max():.4f}])")
        print(f"  Pearson  r(cos2a, |lam|) = {pear(cos2a[mask], lam[mask]):+.4f}")
        print(f"  Spearman r(cos2a, |lam|) = {spear(cos2a[mask], lam[mask]):+.4f}")
        print(f"  cos2a range = [{cos2a[mask].min():.4f},{cos2a[mask].max():.4f}]; "
              f"|lam| range = [{lam[mask].min():.4f},{lam[mask].max():.4f}]")
    # divergence above v*: cos2a keeps falling while |lam| stays on plateau
    above = v >= vstar
    if above.sum() > 3:
        dl = lam[above].max() - lam[above].min()
        dc = cos2a[above].max() - cos2a[above].min()
        print(f"\nabove v*: |lam| varies by {dl:.4f} while cos2a varies by "
              f"{dc:.4f}  -> they diverge (cos2a not the sole determinant)")


# =====================================================================
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    np.set_printoptions(precision=6, suppress=True)
    print(f"# revision_numerics : {cmd}  (gam={GAM}, k_hip={KHIP})")
    {"validate": run_validate, "c3": run_c3, "c4a": run_c4a,
     "c4b": run_c4b, "c6": run_c6}[cmd]()
