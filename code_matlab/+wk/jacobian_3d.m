function J = jacobian_3d(z_fp, gam, k, P, rtol, atol, delta)
%JACOBIAN_3D  Central-difference Jacobian of the full 3-D return map at z_fp.
c = wk.constants();
if nargin < 2 || isempty(gam),   gam   = c.GAM;   end
if nargin < 3 || isempty(k),     k     = c.KHIP;  end
if nargin < 4 || isempty(P),     P     = 0.0;     end
if nargin < 5 || isempty(rtol),  rtol  = c.RTOL;  end
if nargin < 6 || isempty(atol),  atol  = c.ATOL;  end
if nargin < 7 || isempty(delta), delta = c.DELTA; end
z_fp = z_fp(:);
J = zeros(3, 3);
for j = 1:3
    zp = z_fp; zp(j) = zp(j) + delta;
    zm = z_fp; zm(j) = zm(j) - delta;
    [Sp, ~, okp] = wk.step_map(zp, gam, k, P, rtol, atol);
    [Sm, ~, okm] = wk.step_map(zm, gam, k, P, rtol, atol);
    if ~(okp && okm), J = []; return; end
    J(:, j) = (Sp - Sm) / (2.0 * delta);
end
end
