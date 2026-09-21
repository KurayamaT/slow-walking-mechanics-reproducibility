%GARCIA_RATIO_CHECK  Origin of the coefficient 1.04 in the prescribed push-off
%   P(s) = 1.04 alpha tan(alpha).
%
%   It equals MINUS the post-collision stance-velocity-to-step-angle ratio: the
%   PASSIVE simplest walker (Garcia et al., 1998) approaches theta1dot+/alpha =
%   -1.04 in the low-speed (small-slope gamma) limit, so the coefficient is +1.04.
%   The active push-off is prescribed via omega = theta1dot+ = -1.04 alpha, so the
%   powered level-ground gait reproduces this natural ratio.
%
%   Port of code/garcia_ratio_check.py + code/floquet_garcia_python.py (which are
%   themselves ports of others/code/floquet/floquet_Garcia.m). The passive model
%   is self-contained in the local functions below: beta = 0, point feet, hip mass.
%     state on the Poincare section  z = [theta1; theta1dot]
%     theta1  stance leg angle from vertical;  theta2  exterior angle between legs
%     collision (beta = 0):  theta1dot+ = theta1dot- cos(2 theta1),  theta1 -> -theta1
GAMS = [0.018 0.009 0.003 0.001 0.0003 0.0001];
fprintf('Passive simplest walker, period-1 (stable long-step) gait:\n');
fprintf('%9s %16s %12s %16s\n', 'gamma', 'alpha(=theta1*)', 'theta1dot+', 'ratio thd/alpha');
z = [0.20031090049483; -0.19983247291764];      % stable long-step seed
for gam = GAMS
    [zf, ok] = find_fp(z, gam);
    if ok
        a = zf(1); thd = zf(2);
        fprintf('%9.4f %16.5f %12.5f %16.4f\n', gam, a, thd, thd / a);
        z = zf;
    end
end
fprintf('\n-> the ratio approaches about -1.04 as speed -> 0,\n');
fprintf('   matching the prescribed active push-off omega = -1.04*alpha\n');
fprintf('   (P = 1.04*alpha*tan(alpha)).\n');

% ---------------------------------------------------------------- passive model
function dy = eom_garcia(~, y, beta, gam)
    th1 = y(1); th2 = y(2); th1d = y(3); th2d = y(4);
    c2 = cos(th2); s2 = sin(th2);
    s1g = sin(th1 - gam); s12g = sin(th1 + th2 - gam);
    M = [1 + 2*beta*(1 + c2), beta*(1 + c2);
         1 + c2,              1];
    rhs = [beta*s2*th2d*(2*th1d + th2d) + (beta*s12g + s1g*(1 + beta));
           -s2*th1d^2 + s12g];
    qdd = M \ rhs;
    dy = [th1d; th2d; qdd(1); qdd(2)];
end

function [value, isterminal, direction] = hs_event(~, y)
    value = y(2) + 2*y(1) - pi;   % heel strike
    isterminal = 1; direction = -1;
end

function [z_new, te] = stride_map(z, gam, beta)
    z_new = []; te = [];
    th1 = z(1); th1d = z(2);
    if abs(th1) > pi/3 || th1 <= 0, return; end
    th2 = pi - 2*th1;
    th2d = -th1d * (1 - cos(2*th1));
    y0 = [th1; th2; th1d; th2d];
    f = @(t, y) eom_garcia(t, y, beta, gam);
    dt = 0.005;
    [~, y1] = ode45(f, [0 dt], y0, odeset('RelTol', 1e-12, 'AbsTol', 1e-14));
    yd = y1(end, :).';
    [~, ~, tev, ye, ~] = ode45(f, [dt 40.0], yd, ...
        odeset('RelTol', 1e-12, 'AbsTol', 1e-14, 'Events', @hs_event));
    if isempty(tev), return; end
    te = tev(1); yc = ye(1, :);
    if yc(1) > 0, te = []; return; end
    c2t = cos(2*yc(1)); s2t = sin(2*yc(1));
    z_new = [-yc(1); yc(3) * c2t / (1 + beta * s2t^2)];
end

function [z, ok] = find_fp(z0, gam)
    beta = 0.0;
    z = double(z0(:)); ok = false;
    [Sz, ~] = stride_map(z, gam, beta);
    if isempty(Sz), z = []; return; end
    for it = 1:25
        [Sz, ~] = stride_map(z, gam, beta);
        if isempty(Sz), z = []; return; end
        res = Sz - z;
        if norm(res) < 1e-11, ok = true; return; end
        J = zeros(2, 2); d = 1e-7;
        for j = 1:2
            zp = z; zp(j) = zp(j) + d;
            zm = z; zm(j) = zm(j) - d;
            [Sp, ~] = stride_map(zp, gam, beta);
            [Sm, ~] = stride_map(zm, gam, beta);
            if isempty(Sp) || isempty(Sm), z = []; return; end
            J(:, j) = (Sp - Sm) / (2*d);
        end
        Jg = J - eye(2);
        if abs(det(Jg)) < 1e-14, z = []; return; end
        z = z + 0.8 * (Jg \ (-res));
    end
    [Sz, ~] = stride_map(z, gam, beta);
    if ~isempty(Sz) && norm(Sz - z) < 1e-8, ok = true; else, z = []; end
end
