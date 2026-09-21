function J = jacobian_2d(u_fp, gam, k, P, rtol, atol, delta)
%JACOBIAN_2D  Central-difference Jacobian of the reduced 2-D map at u_fp = [theta; theta_dot].
c = wk.constants();
if nargin < 2 || isempty(gam),   gam   = c.GAM;   end
if nargin < 3 || isempty(k),     k     = c.KHIP;  end
if nargin < 4 || isempty(P),     P     = 0.0;     end
if nargin < 5 || isempty(rtol),  rtol  = c.RTOL;  end
if nargin < 6 || isempty(atol),  atol  = c.ATOL;  end
if nargin < 7 || isempty(delta), delta = c.DELTA; end
u_fp = u_fp(:);
J = zeros(2, 2);
for j = 1:2
    up = u_fp; up(j) = up(j) + delta;
    um = u_fp; um(j) = um(j) - delta;
    [Sp, ~, okp] = wk.step_map_2d(up, gam, k, P, rtol, atol);
    [Sm, ~, okm] = wk.step_map_2d(um, gam, k, P, rtol, atol);
    if ~(okp && okm), J = []; return; end
    J(:, j) = (Sp - Sm) / (2.0 * delta);
end
end
