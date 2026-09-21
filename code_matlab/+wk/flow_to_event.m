function [ym, te, ok] = flow_to_event(z, gam, k, rtol, atol)
%FLOW_TO_EVENT  Swing phase only: post-impact state z to the pre-impact state at heel-strike.
%   ym = [theta-; theta_dot-; phi_dot-] on the collision surface (phi- = 2 theta- is implied).
%   This is the first factor of the return map, R = reset o flow_to_event; it carries
%   the swing dynamics AND the dependence of the event time on the initial state.
c = wk.constants();
if nargin < 2 || isempty(gam),  gam  = c.GAM;  end
if nargin < 3 || isempty(k),    k    = c.KHIP; end
if nargin < 4 || isempty(rtol), rtol = c.RTOL; end
if nargin < 5 || isempty(atol), atol = c.ATOL; end
ym = []; te = []; ok = false;
z = z(:);
if abs(z(1)) > pi / 3.0, return; end
y0 = [z(1); z(2); 2.0 * z(1); z(3)];
f = @(t, y) wk.eom(t, y, gam, k);
opts1 = odeset('RelTol', 1e-12, 'AbsTol', 1e-14);
[~, y1] = ode45(f, [0.0, c.DT_DEPART], y0, opts1);
opts2 = odeset('RelTol', rtol, 'AbsTol', atol, 'Events', @wk.events);
[~, ~, te_all, ye, ie] = ode45(f, [c.DT_DEPART, c.PER], y1(end, :).', opts2);
idx = find(ie == 1, 1);
if isempty(idx), return; end
te = te_all(idx); yc = ye(idx, :);
ym = [yc(1); yc(2); yc(4)];
ok = true;
end
