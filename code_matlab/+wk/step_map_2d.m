function [u_new, te, ok] = step_map_2d(u, gam, k, P, rtol, atol)
%STEP_MAP_2D  Reduced map on the post-impact constraint surface, u = [theta; theta_dot].
%   phi_dot is reconstructed as (1 - cos 2 theta) theta_dot.
c = wk.constants();
if nargin < 2 || isempty(gam),  gam  = c.GAM;  end
if nargin < 3 || isempty(k),    k    = c.KHIP; end
if nargin < 4 || isempty(P),    P    = 0.0;    end
if nargin < 5 || isempty(rtol), rtol = c.RTOL; end
if nargin < 6 || isempty(atol), atol = c.ATOL; end
th = u(1); thd = u(2);
phd = (1.0 - cos(2.0 * th)) * thd;
[z_new, te, ok] = wk.step_map([th; thd; phd], gam, k, P, rtol, atol);
if ~ok, u_new = []; return; end
u_new = [z_new(1); z_new(2)];
end
