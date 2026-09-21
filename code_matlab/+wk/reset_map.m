function [z_new, B] = reset_map(ym, P)
%RESET_MAP  Impulsive push-off + heel-strike collision + leg exchange, and its Jacobian.
%   ym = [theta-; theta_dot-; phi_dot-] is the pre-impact state on the collision surface;
%   z_new = [theta+; theta_dot+; phi_dot+] is the post-impact state. P is held fixed
%   (it is prescribed from the nominal step length, not from the perturbed state).
%   B = d z_new / d ym, analytic. The third column is zero: the pre-impact swing-leg
%   velocity is discarded at impact, which is the structural zero multiplier.
a = ym(1); w = ym(2);
c2 = cos(2 * a); s2 = sin(2 * a);
z2 = c2 * w + P * s2;
z_new = [-a; z2; (1 - c2) * z2];
dz2_da = -2 * s2 * w + 2 * P * c2;
dz2_dw = c2;
B = [ -1,                                   0,                 0;
      dz2_da,                               dz2_dw,            0;
      2 * s2 * z2 + (1 - c2) * dz2_da,      (1 - c2) * dz2_dw, 0 ];
end
