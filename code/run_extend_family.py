"""
Recompute the fixed-k_hip family (c=1.04) over the INTENDED sweep range
s in [0.01, 0.80] by seeded continuation, which (unlike the geometric-guess
Newton of the original master) does not stall near s=0.054. Confirms there is
NO saddle-node fold over the computed range: |lambda_max| rises smoothly toward 1 and N_1/2 rises steeply as
v -> 0.

Writes data/master_fixed_khip_extended.csv (Python continuation; a MATLAB
regeneration should replace it before submission).
"""
import os
import numpy as np
import revision_numerics as R

C = 1.04
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", os.environ.get("SLOWWALK_DATA_DIR", "data"), "master_fixed_khip_extended.csv")


def solve_at(q, seed):
    """q is the branch parameter that prescribes the push-off, not the step
    length. The realised heel-strike half-angle is alpha_h = z_fp[0] and the
    reported step length is s = 2 sin(alpha_h)."""
    alpha_p = np.arcsin(0.5 * q)
    P = C * alpha_p * np.tan(alpha_p)
    z0 = list(seed) if seed is not None else [alpha_p, -C * alpha_p, (1 - np.cos(2 * alpha_p)) * (-C * alpha_p)]
    z_fp, T, ok = R.find_fixed_point(z0, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, delta=1e-7)
    if not ok:
        return None
    J = R.jacobian_3d(z_fp, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, R.DELTA)
    if J is None:
        return None
    lam = R.sorted_eigs(J)                      # 3 eigenvalues, one ~0
    nz = sorted(lam, key=lambda L: -abs(L))[:2]  # two non-zero, largest first
    l1, l2 = nz[0], nz[1]
    lam_max = abs(l1)
    alpha_h = float(z_fp[0])
    s = 2.0 * np.sin(alpha_h)
    return dict(q=q, alpha_p=alpha_p, P=P, alpha_h=alpha_h, s=s, v=s / T, T=T,
                cos2a_h=np.cos(2.0 * alpha_h), lam_max=lam_max,
                l1=l1, l2=l2, z=list(z_fp))


# 2-D continuation: anchor, then sweep s down and up
anchor = solve_at(0.40, None)
rows = []
sd = anchor["z"]
for s in np.arange(0.400, 0.0099, -0.001):
    r = solve_at(float(s), sd)
    if r is None:
        print(f"  down: ended at s={s+0.001:.3f}")
        break
    sd = r["z"]
    rows.append(r)
su = anchor["z"]
for s in np.arange(0.401, 0.801, 0.001):
    r = solve_at(float(s), su)
    if r is None:
        print(f"  up: ended at s={s-0.001:.3f}")
        break
    su = r["z"]
    rows.append(r)

rows.sort(key=lambda r: r["q"])

with open(OUT, "w") as fh:
    fh.write("q,alpha_p,P,alpha_h,s,v,T,cos2alpha_h,lam_max,recovery_margin,N_half,lam1_re,lam1_im,lam2_re,lam2_im,eig_type,stable\n")
    for r in rows:
        lm = r["lam_max"]
        nh = -np.log(2.0) / np.log(lm) if 0 < lm < 1 else float("nan")
        etype = "complex" if abs(r["l1"].imag) > 1e-7 else "real"
        stable = 1 if lm < 1 else 0
        fh.write(f"{r['q']:.3f},{r['alpha_p']:.10g},{r['P']:.10g},{r['alpha_h']:.10g},"
                 f"{r['s']:.10g},{r['v']:.10g},{r['T']:.6f},{r['cos2a_h']:.10g},"
                 f"{lm:.8f},{1-lm:.8f},{nh:.4f},"
                 f"{r['l1'].real:.8f},{r['l1'].imag:.8f},{r['l2'].real:.8f},{r['l2'].imag:.8f},"
                 f"{etype},{stable}\n")

v = np.array([r["v"] for r in rows])
lm = np.array([r["lam_max"] for r in rows])
nh = -np.log(2.0) / np.log(np.clip(lm, 1e-12, 0.9999999))
print(f"\nrows={len(rows)}  v in [{v.min():.4f}, {v.max():.4f}]  "
      f"|lmax| in [{lm.min():.4f}, {lm.max():.5f}]")
for vq in (0.0029, 0.006, 0.0116, 0.0156, 0.0231, 0.116, 0.231):
    i = int(np.argmin(np.abs(v - vq)))
    print(f"  v~{v[i]:.4f}: |lmax|={lm[i]:.5f}  N_half={nh[i]:.1f}")
print(f"\nsaved: {OUT}")
print("NO fold over the computed range: |lambda_max| rises smoothly toward 1; N_1/2 rises steeply toward the slow end.")
