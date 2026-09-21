"""
Hip-spring-coefficient robustness, recomputed by seeded continuation so that it is
comparable with the main branch.

The existing sens_khip_summary.csv comes from main_khip_sweep.m, which solves each
gait from a fixed geometric guess. That guess stalls well above s = 0.01 and at a
different place on every branch, so its "slowest gait computed" is a property of
the solver, not of the branch: at k_hip = -0.16 it stops at v = 0.0156 with
N_1/2 = 28.07, while the same branch continued by seeding reaches v = 0.0029 with
N_1/2 = 842.2. Plotting those stall points against the continued main branch, as
Figure S4a did, compares two different things.

Here every k_hip branch is traced the same way as the main branch: seeded
continuation over s in [0.01, 0.80] at 0.001 spacing, anchored at s = 0.40 and
reached by continuation in k_hip from the main value where the geometric guess
does not converge.

Output: data/supp_khip_sweep.csv  (branch data, one row per (k_hip, s))
        printed summary comparable with Table 1
"""
import os
import numpy as np
import revision_numerics as R

KHIPS = [-0.30, -0.20, -0.16, -0.12, -0.08, -0.04, 0.00, 0.05, 0.10]
C = 1.04
S_ANCHOR = 0.40
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", os.environ.get("SLOWWALK_DATA_DIR", "data")))


def push_off(s):
    a = np.arcsin(0.5 * s)
    return C * a * np.tan(a)


def solve_at(s, k, seed):
    P = push_off(s)
    a = np.arcsin(0.5 * s)
    z0 = list(seed) if seed is not None else [a, -C * a, (1 - np.cos(2 * a)) * (-C * a)]
    z, T, ok = R.find_fixed_point(z0, R.GAM, k, P, R.RTOL, R.ATOL, delta=1e-7)
    return (z, T, P) if ok else (None, None, P)


def anchor_for(k):
    """Fixed point at s = S_ANCHOR for this k_hip, by continuation in k_hip."""
    z, T, P = solve_at(S_ANCHOR, k, None)
    if z is not None:
        return z
    seed = None
    for kc in np.linspace(R.KHIP, k, max(2, int(abs(k - R.KHIP) / 0.01) + 1)):
        z, T, P = solve_at(S_ANCHOR, float(kc), seed)
        if z is None:
            raise RuntimeError("k_hip continuation failed at k=%.3f" % kc)
        seed = z
    return z


def trace_branch(k):
    """Seeded continuation down from s = S_ANCHOR to 0.01, then up to 0.80."""
    out = {}
    z = anchor_for(k)
    for direction in (-1, +1):
        seed = z
        grid = (np.arange(S_ANCHOR, 0.010 - 1e-9, -0.001) if direction < 0
                else np.arange(S_ANCHOR, 0.800 + 1e-9, +0.001))
        for s in grid:
            s = round(float(s), 3)
            zz, T, P = solve_at(s, k, seed)
            if zz is None:
                break
            seed = zz
            J = R.jacobian_3d(zz, R.GAM, k, P, R.RTOL, R.ATOL, R.DELTA)
            lam = sorted(np.linalg.eigvals(J), key=lambda L: -abs(L))
            rho = abs(lam[0])
            alpha_h = float(zz[0])          # this gait's fixed point, not the anchor's
            s_real = 2.0 * np.sin(alpha_h)
            out[s] = (s_real / T, rho, (-np.log(2) / np.log(rho)) if 0 < rho < 1 else np.nan,
                      float(np.imag(lam[0])), alpha_h, s_real)
    return out


rows, summary = [], []
for k in KHIPS:
    br = trace_branch(k)
    ss = sorted(br)
    for s in ss:
        v, rho, nh, im, a_h, s_real = br[s]
        rows.append((k, s, a_h, s_real, v, rho, nh, im))
    s_lo = ss[0]
    v_lo, rho_lo, nh_lo, _, _, _ = br[s_lo]
    cx = [br[s][0] for s in ss if br[s][3] != 0.0]
    vstar = min(cx) if cx else float("nan")
    summary.append((k, len(ss), s_lo, v_lo, nh_lo, vstar))
    print("  k_hip=%+.3f  n=%3d  s_min=%.3f  v_min=%.4f  N_1/2=%8.1f  v*=%.4f"
          % (k, len(ss), s_lo, v_lo, nh_lo, vstar), flush=True)

with open(os.path.join(DATA, "supp_khip_sweep.csv"), "w") as f:
    f.write("k_hip,q,alpha_h,s,v,lam_max,N_half,lam1_im\n")
    for k, q, a_h, s_real, v, rho, nh, im in rows:
        # q at three decimals is the grid key; the realised s must not be
        # rounded there — all nine branches would print as 0.010.
        f.write("%.3f,%.3f,%.10g,%.10g,%.10g,%.10f,%.4f,%.10f\n"
                % (k, q, a_h, s_real, v, rho, nh, im))
print("saved: data/supp_khip_sweep.csv")
