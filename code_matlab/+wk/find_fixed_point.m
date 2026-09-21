function [z, T, ok] = find_fixed_point(z0, gam, k, P, rtol, atol, tol, max_iter, delta)
%FIND_FIXED_POINT  Damped Newton iteration for the periodic post-impact state.
%   Central-difference Jacobian with step delta, damping 0.8, condition guard,
%   and a final acceptance test at 1e-9 (as in the reference implementation).
c = wk.constants();
if nargin < 2 || isempty(gam),      gam      = c.GAM;  end
if nargin < 3 || isempty(k),        k        = c.KHIP; end
if nargin < 4 || isempty(P),        P        = 0.0;    end
if nargin < 5 || isempty(rtol),     rtol     = c.RTOL; end
if nargin < 6 || isempty(atol),     atol     = c.ATOL; end
if nargin < 7 || isempty(tol),      tol      = 1e-12;  end
if nargin < 8 || isempty(max_iter), max_iter = 20;     end
if nargin < 9 || isempty(delta),    delta    = 1e-7;   end

z = double(z0(:)); T = []; ok = false;
[~, ~, ok0] = wk.step_map(z, gam, k, P, rtol, atol);
if ~ok0, z = []; return; end
for it = 1:max_iter
    [Sz, T, ok1] = wk.step_map(z, gam, k, P, rtol, atol);
    if ~ok1, z = []; T = []; return; end
    res = Sz - z;
    if norm(res) < tol, ok = true; return; end
    Jg = zeros(3, 3);
    for j = 1:3
        zp = z; zp(j) = zp(j) + delta;
        zm = z; zm(j) = zm(j) - delta;
        [Sp, ~, okp] = wk.step_map(zp, gam, k, P, rtol, atol);
        [Sm, ~, okm] = wk.step_map(zm, gam, k, P, rtol, atol);
        if ~(okp && okm), z = []; T = []; return; end
        Jg(:, j) = (Sp - Sm) / (2.0 * delta);
    end
    Jg = Jg - eye(3);
    if 1.0 / cond(Jg) < 1e-14, z = []; T = []; return; end
    z = z + 0.8 * (Jg \ (-res));
end
[Sz, T, ok1] = wk.step_map(z, gam, k, P, rtol, atol);
if ok1 && norm(Sz - z) < 1e-9
    ok = true; return;
end
z = []; T = []; ok = false;
end
