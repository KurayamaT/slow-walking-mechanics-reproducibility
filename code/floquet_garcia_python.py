"""
Passive Garcia (1998) simplest walking model — Floquet sweep across slope (speed).
Python port of code/floquet/floquet_Garcia.m (beta=0, point feet, hip mass).

Purpose: resolve whether the PASSIVE model's stability margin (|lambda_max| of the
stable long-step branch) collapses toward 1 at LOW speed (like the active Kuo model)
or stays bounded (~0.5-0.6). This decides the passive-vs-active novelty question.

State on Poincare section: z = [theta1, theta1dot].
  theta1   = stance leg angle from vertical
  theta2   = exterior angle between legs (= pi - phi)
Collision: theta1dot+ = theta1dot- * cos(2*theta1)  (beta=0),  theta1 -> -theta1.
"""
import numpy as np
from scipy.integrate import solve_ivp

G = 1.0
L = 1.0
PER = 40.0
BETA = 0.0


def eom_garcia(t, y, beta, gam):
    th1, th2, th1d, th2d = y
    c2 = np.cos(th2); s2 = np.sin(th2)
    s1g = np.sin(th1 - gam)
    s12g = np.sin(th1 + th2 - gam)
    M = np.array([[1.0 + 2*beta*(1+c2), beta*(1+c2)],
                  [1.0 + c2,            1.0       ]])
    rhs = np.array([beta*s2*th2d*(2*th1d+th2d) + (G/L)*(beta*s12g + s1g*(1+beta)),
                    -s2*th1d**2 + (G/L)*s12g])
    qdd = np.linalg.solve(M, rhs)
    return [th1d, th2d, qdd[0], qdd[1]]


def _hs_event(t, y, beta, gam):
    return y[1] + 2*y[0] - np.pi
_hs_event.terminal = True
_hs_event.direction = -1


def stride_map(z, gam, beta=BETA):
    th1, th1d = float(z[0]), float(z[1])
    if abs(th1) > np.pi/3 or th1 <= 0:
        return None, None
    th2 = np.pi - 2*th1
    th2d = -th1d * (1 - np.cos(2*th1))
    y0 = [th1, th2, th1d, th2d]
    dt_depart = 0.005
    s1 = solve_ivp(eom_garcia, [0, dt_depart], y0, args=(beta, gam),
                   rtol=1e-12, atol=1e-14)
    yd = s1.y[:, -1]
    s2 = solve_ivp(eom_garcia, [dt_depart, PER], yd, args=(beta, gam),
                   rtol=1e-12, atol=1e-14, events=_hs_event)
    if len(s2.t_events[0]) == 0:
        return None, None
    te = s2.t_events[0][0]
    yc = s2.y_events[0][0]
    if yc[0] > 0:
        return None, None
    c2t = np.cos(2*yc[0]); s2t = np.sin(2*yc[0])
    th1d_new = yc[2] * c2t / (1 + beta * s2t**2)
    th1_new = -yc[0]
    return np.array([th1_new, th1d_new]), te


def find_fp(z0, gam, beta=BETA):
    z = np.array(z0, dtype=float)
    Sz, T = stride_map(z, gam, beta)
    if Sz is None:
        return None, False
    for _ in range(25):
        Sz, T = stride_map(z, gam, beta)
        if Sz is None:
            return None, False
        res = Sz - z
        if np.linalg.norm(res) < 1e-11:
            return z, True
        J = np.zeros((2, 2)); d = 1e-7
        for j in range(2):
            zp = z.copy(); zp[j] += d
            zm = z.copy(); zm[j] -= d
            Sp, _ = stride_map(zp, gam, beta)
            Sm, _ = stride_map(zm, gam, beta)
            if Sp is None or Sm is None:
                return None, False
            J[:, j] = (Sp - Sm) / (2*d)
        Jg = J - np.eye(2)
        if abs(np.linalg.det(Jg)) < 1e-14:
            return None, False
        z = z + 0.8 * np.linalg.solve(Jg, -res)
    Sz, T = stride_map(z, gam, beta)
    return (z, True) if (Sz is not None and np.linalg.norm(Sz - z) < 1e-8) else (None, False)


def floquet_at(z_fp, gam, beta=BETA):
    d = 1e-7; J = np.zeros((2, 2))
    for j in range(2):
        zp = z_fp.copy(); zp[j] += d
        zm = z_fp.copy(); zm[j] -= d
        Sp, _ = stride_map(zp, gam, beta)
        Sm, _ = stride_map(zm, gam, beta)
        if Sp is None or Sm is None:
            return None, None
        J[:, j] = (Sp - Sm) / (2*d)
    lam = np.linalg.eigvals(J)
    lam = lam[np.argsort(-np.abs(lam))]
    _, T = stride_map(z_fp, gam, beta)
    return lam, T


def sweep_branch(z_ref, gam_ref, gam_grid):
    """Continuation from gam_ref outward over gam_grid (sorted)."""
    out = {}
    i_ref = int(np.argmin(np.abs(gam_grid - gam_ref)))
    z_fp, ok = find_fp(z_ref, gam_grid[i_ref])
    if not ok:
        print("  reference failed"); return out
    # forward (increasing) then backward (decreasing) continuation
    for direction in (+1, -1):
        z_guess = z_fp.copy()
        idxs = range(i_ref, len(gam_grid)) if direction == +1 else range(i_ref, -1, -1)
        nfail = 0
        for i in idxs:
            zf, ok = find_fp(z_guess, gam_grid[i])
            if not ok:
                gc = gam_grid[i] ** (1/3)               # theta1* ~ gamma^(1/3) fallback
                zf, ok = find_fp([0.963*gc, -0.961*gc], gam_grid[i])
            if not ok:
                nfail += 1
                if nfail > 10:
                    break
                continue
            nfail = 0
            lam, T = floquet_at(zf, gam_grid[i])
            if lam is None:
                continue
            s_len = 2*L*np.sin(zf[0])
            out[i] = dict(gam=gam_grid[i], theta1=zf[0], s=s_len, T=T, v=s_len/T,
                          lam=lam, absmax=float(np.max(np.abs(lam))))
            z_guess = zf
    return out


if __name__ == "__main__":
    gam_grid = np.linspace(0.0005, 0.019, 75)
    gam_ref = 0.009
    z_long = [0.20031090049483, -0.19983247291764]   # stable long-step
    print("Sweeping PASSIVE Garcia long-step (stable) branch, beta=0 ...")
    res = sweep_branch(z_long, gam_ref, gam_grid)
    rows = [res[i] for i in sorted(res)]
    print(f"  converged {len(rows)}/{len(gam_grid)} orbits\n")
    print(f"{'gamma':>8} {'s':>8} {'v':>9} {'T':>7} {'|lam_max|':>10}  {'lam1':>22} {'lam2':>12}")
    print("-"*92)
    for r in rows:
        l1, l2 = r['lam'][0], r['lam'][1]
        l1s = f"{l1.real:.4f}{l1.imag:+.4f}i" if abs(l1.imag) > 1e-6 else f"{l1.real:.5f}"
        l2s = f"{l2.real:.4f}{l2.imag:+.4f}i" if abs(l2.imag) > 1e-6 else f"{l2.real:.5f}"
        print(f"{r['gam']:8.4f} {r['s']:8.4f} {r['v']:9.5f} {r['T']:7.3f} {r['absmax']:10.5f}  {l1s:>22} {l2s:>12}")

    # focused low-speed summary
    print("\n=== LOW-SPEED behaviour of PASSIVE stable branch ===")
    rows_sorted_v = sorted(rows, key=lambda r: r['v'])
    for r in rows_sorted_v[:8]:
        print(f"  v={r['v']:.4f}  |lam_max|={r['absmax']:.4f}  (gamma={r['gam']:.4f}, s={r['s']:.4f})")
    vmin_row = rows_sorted_v[0]
    print(f"\n  slowest passive orbit: v={vmin_row['v']:.4f}, |lam_max|={vmin_row['absmax']:.4f}")
    print(f"  range of |lam_max| over sweep: [{min(r['absmax'] for r in rows):.4f}, {max(r['absmax'] for r in rows):.4f}]")
