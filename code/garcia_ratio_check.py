"""
Origin of the coefficient 1.04 in the prescribed push-off P(s)=1.04*alpha*tan(alpha).

It equals MINUS the post-collision stance-velocity-to-step-angle ratio: the PASSIVE
simplest walker (Garcia et al. 1998, ref [8]) approaches theta1dot+/alpha = -1.04 in
the low-speed (small-slope gamma) limit, and the coefficient is +1.04. Getting this
sign wrong here is how it reached the manuscript, where it has been corrected. The active push-off is
prescribed via omega = theta1dot+ = -1.04*alpha, so the powered level-ground gait
reproduces this natural ratio.

Uses the passive-walker port floquet_garcia_python.py (same model as Supplementary S6).
Run: python3 garcia_ratio_check.py
"""
import floquet_garcia_python as G

print("Passive simplest walker [8], period-1 (stable long-step) gait:")
print(f"{'gamma':>9} {'alpha(=theta1*)':>16} {'theta1dot+':>12} {'ratio thd/alpha':>16}")
z = [0.20031090049483, -0.19983247291764]   # stable long-step seed
for gam in (0.018, 0.009, 0.003, 0.001, 0.0003, 0.0001):
    zf, ok = G.find_fp(z, gam)
    if ok:
        a, thd = float(zf[0]), float(zf[1])
        print(f"{gam:9.4f} {a:16.5f} {thd:12.5f} {thd / a:16.4f}")
        z = zf

print("\n-> the ratio approaches about -1.04 as speed -> 0,")
print("   matching the prescribed active push-off omega = -1.04*alpha")
print("   (P = 1.04*alpha*tan(alpha)).")
