"""
Investigate the low-speed end of the fixed-k_hip family (c = 1.04):
  1. Continue the STABLE branch finely from s=0.40 downward; record (s, v,
     |lambda_max|); find the lowest s with a converged stable fixed point and
     whether |lambda_max| -> 1 there (a genuine fold/limit) or the branch is
     simply cut off by non-convergence.
  2. Cross-check against the earlier MATLAB master near s=0.054.
  3. Forward-iterate the return map from the lowest gait to confirm it is a
     genuine, stable periodic orbit (not a numerical artifact).
  4. Near s=0.054, look for a second fixed point.
"""
import numpy as np
import revision_numerics as R

C = 1.04


def solve_at(s, seed):
    alpha = np.arcsin(0.5 * s)
    P = C * alpha * np.tan(alpha)
    z0 = list(seed) if seed is not None else [alpha, -C * alpha, (1 - np.cos(2 * alpha)) * (-C * alpha)]
    z_fp, T, ok = R.find_fixed_point(z0, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, delta=1e-7)
    if not ok:
        return None
    J = R.jacobian_3d(z_fp, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, R.DELTA)
    if J is None:
        return None
    lam = R.sorted_eigs(J)
    return dict(z=list(z_fp), v=s / T, T=T, P=P, lam_max=float(np.max(np.abs(lam))), lam=lam)


# ---- 1. fine downward continuation of the stable branch ------------------ #
print("=== downward continuation (s : 0.40 -> 0.01), watching |lambda_max| ===")
seed = solve_at(0.40, None)["z"]
last = None
milestones = [0.0538, 0.054, 0.050, 0.045, 0.040, 0.030, 0.020, 0.015, 0.010]
mset = sorted(milestones, reverse=True)
mi = 0
for s in np.arange(0.400, 0.0099, -0.001):
    r = solve_at(float(s), seed)
    if r is None:
        print(f"  --> branch ENDS near s={s+0.001:.3f} (last v={last['v']:.4f}, "
              f"|lmax|={last['lam_max']:.4f}); no converged fixed point below.")
        break
    seed = r["z"]
    last = r
    last["s"] = float(s)
    while mi < len(mset) and s <= mset[mi] + 1e-9:
        print(f"  s={s:.4f}  v={r['v']:.4f}  |lmax|={r['lam_max']:.5f}")
        mi += 1
else:
    print(f"  reached s={last['s']:.3f} (v={last['v']:.4f}, |lmax|={last['lam_max']:.4f}) "
          f"still converged & stable.")

# ---- 2. cross-check master near s=0.054 ---------------------------------- #
print("\n=== MATLAB master vs Python near the paper's slowest gait ===")
rows = R.load_master()
mrow = min(rows, key=lambda r: float(r["v"]))
print(f"  master slowest: s={float(mrow['s']):.4f} v={float(mrow['v']):.4f} "
      f"|lmax|={float(mrow['lam_max']):.4f}  (eig_type={mrow['eig_type']})")
pc = solve_at(float(mrow["s"]), seed=None)
if pc:
    print(f"  python  same s: s={float(mrow['s']):.4f} v={pc['v']:.4f} |lmax|={pc['lam_max']:.4f}")

# ---- 3. forward-iterate the lowest gait (true periodic orbit?) ----------- #
print("\n=== forward iteration of the lowest converged gait ===")
zlo = np.array(last["z"], float)
P_lo = last["P"]
z = zlo.copy()
devs = []
for n in range(20):
    z_next, _, ok = R.step_map(z, R.GAM, R.KHIP, P_lo, R.RTOL, R.ATOL)
    if not ok:
        raise RuntimeError(f"step map failed at iteration {n + 1}")
    z = np.array(z_next, float)
    devs.append(float(np.linalg.norm(z - zlo)))
print(f"  s={last['s']:.4f} v={last['v']:.4f}: max |z_n - z*| over 20 steps = {max(devs):.2e} "
      f"(stays put => genuine stable orbit)" if max(devs) < 1e-3 else
      f"  s={last['s']:.4f}: DRIFTS, max dev={max(devs):.2e} (NOT a clean orbit)")

# ---- 4. search for a second (unstable) fixed point near s=0.054 ---------- #
print("\n=== search for an unstable partner near s=0.054 ===")
for s in (0.054, 0.050, 0.045):
    alpha = np.arcsin(0.5 * s)
    P = C * alpha * np.tan(alpha)
    found = {}
    for fac in (0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5):
        z0 = [alpha, -fac * alpha, (1 - np.cos(2 * alpha)) * (-fac * alpha)]
        zf, T, ok = R.find_fixed_point(z0, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, delta=1e-7)
        if ok:
            J = R.jacobian_3d(zf, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, R.DELTA)
            lm = float(np.max(np.abs(R.sorted_eigs(J)))) if J is not None else float("nan")
            found[round(s / T, 4)] = round(lm, 4)
    print(f"  s={s:.3f}: distinct (v:|lmax|) solutions found = {found}")
