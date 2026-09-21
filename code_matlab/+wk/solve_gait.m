function g = solve_gait(q, gam, k, rtol, atol, delta, seed)
%SOLVE_GAIT  Periodic gait and Floquet multipliers at branch parameter q on the
%   prescribed-push-off branch P(q) = 1.04 alpha_p tan(alpha_p), alpha_p =
%   asin(q/2).
%
%   q is the CONTINUATION parameter that prescribes the push-off. It is not the
%   step length: the fixed point realises a heel-strike half inter-leg angle
%   alpha_h = z_fp(1), and the realised step length and speed are
%
%       s = 2 sin(alpha_h),   v = s / T,
%
%   which are what the manuscript reports. alpha_h/alpha_p is about 0.988 on the
%   reference branch, so the two differ by roughly 1.2%; keeping them apart is
%   the whole point of this signature. The field `alpha` was deliberately
%   removed so that a caller written against the old convention fails loudly
%   instead of silently mixing bases.
%
%   Returns a struct, or [] if the fixed point or the Jacobian cannot be formed.
%   seed: optional initial post-impact state for continuation.
c = wk.constants();
if nargin < 2 || isempty(gam),   gam   = c.GAM;   end
if nargin < 3 || isempty(k),     k     = c.KHIP;  end
if nargin < 4 || isempty(rtol),  rtol  = c.RTOL;  end
if nargin < 5 || isempty(atol),  atol  = c.ATOL;  end
if nargin < 6 || isempty(delta), delta = c.DELTA; end
if nargin < 7, seed = []; end

alpha_p = asin(0.5 * q);
omega = -1.04 * alpha_p;
P = -omega * tan(alpha_p);                     % = 1.04 alpha_p tan(alpha_p)
if isempty(seed)
    z0 = [alpha_p; omega; (1.0 - cos(2.0 * alpha_p)) * omega];
else
    z0 = seed(:);
end
[z_fp, T, ok] = wk.find_fixed_point(z0, gam, k, P, rtol, atol, 1e-12, 20, 1e-7);
if ~ok, g = []; return; end
J = wk.jacobian_3d(z_fp, gam, k, P, rtol, atol, delta);
if isempty(J), g = []; return; end
lam = wk.sorted_eigs(J);
alpha_h = z_fp(1);
s = 2.0 * sin(alpha_h);
g = struct('q', q, 'alpha_p', alpha_p, 'P', P, 'alpha_h', alpha_h, ...
           's', s, 'v', s / T, 'T', T, ...
           'z_fp', z_fp, 'lam', lam, 'lam_max', max(abs(lam)), 'J', J);
end
