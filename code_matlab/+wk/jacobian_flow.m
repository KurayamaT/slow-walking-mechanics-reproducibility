function A = jacobian_flow(z_fp, gam, k, rtol, atol, delta)
%JACOBIAN_FLOW  Central-difference Jacobian of the swing-phase map (post-impact state to
%   pre-impact state at heel-strike), A = d ym / d z at the fixed point. Together with
%   the analytic reset Jacobian B, the return-map Jacobian is J = B * A.
c = wk.constants();
if nargin < 2 || isempty(gam),   gam   = c.GAM;   end
if nargin < 3 || isempty(k),     k     = c.KHIP;  end
if nargin < 4 || isempty(rtol),  rtol  = c.RTOL;  end
if nargin < 5 || isempty(atol),  atol  = c.ATOL;  end
if nargin < 6 || isempty(delta), delta = c.DELTA; end
z_fp = z_fp(:);
A = zeros(3, 3);
for j = 1:3
    zp = z_fp; zp(j) = zp(j) + delta;
    zm = z_fp; zm(j) = zm(j) - delta;
    [yp, ~, okp] = wk.flow_to_event(zp, gam, k, rtol, atol);
    [ymn, ~, okm] = wk.flow_to_event(zm, gam, k, rtol, atol);
    if ~(okp && okm), A = []; return; end
    A(:, j) = (yp - ymn) / (2.0 * delta);
end
end
