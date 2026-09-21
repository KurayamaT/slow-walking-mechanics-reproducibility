"""
Nonlinear return-map recovery traces for Figure 2b.

Figure 2b previously plotted the closed-form modal signal rho^n cos(n phi) implied
by the dominant multiplier, while Methods (3.3) and Results (4.3) stated that a
nonlinear return-map iteration was shown. This script produces the iteration the
text describes, so that the figure and the text agree.

Two gaits, matching 4.3 and Table 1:
    s = 0.080   low speed, dominant positive real multiplier
    s = 0.402   plateau,   dominant complex-conjugate pair
The slowest gait (s = 0.010) is deliberately not here: Figure 2b contrasts the two
recovery modes, and the 5,000-step verification of the 842-step half-life belongs
to Supplementary S3.

PERTURBATION DIRECTION. The displacement is applied along the dominant right
eigenvector where the dominant multiplier is real, and along the real part of the
dominant right eigenvector where the dominant pair is complex - a complex
eigenvector is not a usable real initial state, and neither member of the pair is
"the" dominant one. In both cases the direction is scaled so its stance-angle
component is one, so that "0.3% of |theta*|" names a stance-angle displacement and
not the norm of an arbitrarily scaled eigenvector. It is not applied along the
stance-leg angle alone. Displacing theta alone is
dominated by the conditioning of the eigenbasis rather than by the recovery rate:
the two eigenvectors are nearly parallel (Supplementary S3), so the stance-angle
component reverses sign at the first step and grows to about 6x its initial size
at s = 0.080 before decaying, and the transient factor itself depends on the
amplitude. That behaviour is real and is reported in S3, but it is not what
Figure 2b is about, and plotting it would contradict the figure's own caption.
Along the dominant eigenvector the per-step ratio equals |lambda_max| to four
decimals, the decay halves in exactly N_1/2 steps, and the recovery mode is qualitatively
unchanged over 0.1-1.0% - so the figure shows the recovery modes it claims to.

k_hip and P(s) are held at their periodic-gait values, so the simulated recovery is
open-loop. What is recorded is the signed stance-angle deviation normalised by its
own initial value, which keeps the oscillation of the complex mode visible.

Writes data/fig2b_traces.csv and prints the consistency checks.
"""
import os
import numpy as np
import revision_numerics as R

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", os.environ.get("SLOWWALK_DATA_DIR", "data")))

C = 1.04
AMP = 0.003                            # 0.3% of |theta*|, the amplitude Figure 2b shows
AMPS_CHECK = [0.001, 0.003, 0.010]     # shape must not depend on which of these is used
GAITS = [(0.402, 28, "plateau"), (0.080, 40, "low speed")]


def solve(s):
    """Fixed point, period, push-off and spectrum at step length s.

    The geometric guess converges for s >= 0.054 (3.2), which covers both gaits
    used here, so no continuation is needed.
    """
    alpha = np.arcsin(0.5 * s)
    P = C * alpha * np.tan(alpha)
    z0 = [alpha, -C * alpha, (1 - np.cos(2 * alpha)) * (-C * alpha)]
    z_fp, T, ok = R.find_fixed_point(z0, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, delta=1e-7)
    if not ok:
        raise RuntimeError("fixed point failed at s=%.3f" % s)
    J = R.jacobian_3d(z_fp, R.GAM, R.KHIP, P, R.RTOL, R.ATOL, R.DELTA)
    w, V = np.linalg.eig(J)
    o = np.argsort(-np.abs(w))
    return z_fp, T, P, w[o], V[:, o]


def dominant_direction(V):
    """Real perturbation direction from the dominant right eigenvector.

    When the dominant multiplier is real this is the eigenvector itself. When the
    dominant pair is complex the eigenvector is not a usable real initial state and
    neither member of the pair is "the" dominant one, so the real part is used.

    PHASE. numpy returns a complex eigenvector up to an arbitrary phase, and
    Re(exp(i psi) r) is a different real vector for each psi - all inside the same
    two-dimensional invariant subspace, so the |lambda|^n envelope is the same, but
    the oscillation phase of the trace is not. Taking the real part of whatever
    numpy happened to return would therefore make Figure 2b depend on the LAPACK
    build. The phase is normalised first, so that the stance-angle component is
    real and positive; the initial direction is then unique. This does nothing at
    a real-multiplier gait, where the eigenvector is already real and the scaling
    below absorbs the sign. On the reference build LAPACK returned r[0] real and
    negative at both gaits, so the normalisation is exactly the sign flip the
    scaling already absorbed and the deposited traces are unchanged by it.

    The vector is scaled so its stance-angle component is exactly 1, which makes
    the perturbation amplitude below a stance-angle displacement rather than a
    vector norm - the norm would depend on the arbitrary scaling of the eigenvector.
    """
    r = V[:, 0]
    r = r * np.exp(-1j * np.angle(r[0]))
    d = np.real(r)
    d = d / d[0]
    return d


def signed_trace(z_fp, P, d, amp, nsteps):
    """Signed stance-angle deviation, normalised by its own initial value.

    d has unit stance-angle component, so the initial stance-angle displacement is
    exactly amp * |theta*|.
    """
    d0 = amp * abs(z_fp[0])
    z = np.asarray(z_fp, dtype=float) + d0 * d
    dev = []
    for _ in range(nsteps + 1):
        dev.append(z[0] - z_fp[0])
        z, _, ok = R.step_map(z, R.GAM, R.KHIP, P, R.RTOL, R.ATOL)
        if not ok:
            break
    dev = np.array(dev)
    return dev / dev[0]


rows, summary = [], []
for s, nsteps, label in GAITS:
    z_fp, T, P, w, V = solve(s)
    d = dominant_direction(V)
    lam_max = abs(w[0])
    v = s / T
    N_half = -np.log(2) / np.log(lam_max)
    tr = signed_trace(z_fp, P, d, AMP, nsteps)

    # per-step ratio over the first few steps, before the complex mode changes sign
    ratios = np.abs(tr[1:5] / tr[0:4])
    fitted = float(np.median(ratios))

    # shape must not depend on amplitude over the range 3.3 reports
    max_dev = max(float(np.nanmax(np.abs(signed_trace(z_fp, P, d, a, nsteps) - tr)))
                  for a in AMPS_CHECK)

    # value at n = N_1/2 (real-dominant gaits should be at one half)
    at_half = float(tr[int(round(N_half))]) if int(round(N_half)) < len(tr) else float("nan")

    summary.append((label, s, v, lam_max, N_half, fitted, max_dev, at_half, w))
    for n, dv in enumerate(tr):
        rows.append((label, s, n, dv, lam_max ** n))

with open(os.path.join(DATA, "fig2b_traces.csv"), "w") as f:
    f.write("gait,s,n,dev_signed,envelope\n")
    for label, s, n, dv, e in rows:
        f.write("%s,%.3f,%d,%.10e,%.10e\n" % (label, s, n, dv, e))

print("wrote data/fig2b_traces.csv   amplitude = %.1f%% of |theta*|, "
      "as a stance-angle displacement along the dominant direction" % (100 * AMP))
for label, s, v, lam_max, N_half, fitted, max_dev, at_half, w in summary:
    print("  %-10s s=%.3f v=%.4f |lam_max|=%.6f N_1/2=%.2f" % (label, s, v, lam_max, N_half))
    print("             per-step ratio %.6f vs |lam_max| %.6f (err %.1e); "
          "dev at n=N_1/2 = %+.4f; shape spread over 0.1-1.0%% = %.2e"
          % (fitted, lam_max, abs(fitted - lam_max), at_half, max_dev))
    print("             lam = %s" % ", ".join("%.6f%+.6fi" % (L.real, L.imag) for L in w))
