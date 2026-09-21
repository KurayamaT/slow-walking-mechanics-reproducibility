function [z_new, te, ok] = step_map(z, gam, k, P, rtol, atol)
%STEP_MAP  One step of the post-impact return map R(z), z = [theta; theta_dot; phi_dot].
%   Two-phase integration: a short event-free departure from the collision
%   surface, then integration to heel-strike, then the impulsive reset with
%   push-off P. Returns ok = false if the walker falls or no heel-strike occurs.
c = wk.constants();
if nargin < 2 || isempty(gam),  gam  = c.GAM;  end
if nargin < 3 || isempty(k),    k    = c.KHIP; end
if nargin < 4 || isempty(P),    P    = 0.0;    end
if nargin < 5 || isempty(rtol), rtol = c.RTOL; end
if nargin < 6 || isempty(atol), atol = c.ATOL; end

z_new = []; te = []; ok = false;
z = z(:);
if abs(z(1)) > pi / 3.0, return; end
y0 = [z(1); z(2); 2.0 * z(1); z(3)];
f = @(t, y) wk.eom(t, y, gam, k);

% Phase 1: depart the collision surface (no events; tolerances fixed as in the reference code)
opts1 = odeset('RelTol', 1e-12, 'AbsTol', 1e-14);
[~, y1] = ode45(f, [0.0, c.DT_DEPART], y0, opts1);
y_dep = y1(end, :).';

% Phase 2: integrate to heel-strike
opts2 = odeset('RelTol', rtol, 'AbsTol', atol, 'Events', @wk.events);
[~, ~, te_all, ye, ie] = ode45(f, [c.DT_DEPART, c.PER], y_dep, opts2);
idx = find(ie == 1, 1);
if isempty(idx), return; end
te = te_all(idx);
yc = ye(idx, :);
c2  = cos(2.0 * yc(1));
s2P = sin(2.0 * yc(1)) * P;
z_new = [-yc(1);
         c2 * yc(2) + s2P;
         c2 * (1.0 - c2) * yc(2) + (1.0 - c2) * s2P];
ok = true;
end
