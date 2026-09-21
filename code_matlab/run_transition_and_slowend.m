%RUN_TRANSITION_AND_SLOWEND  Two verifications on the fixed-k_hip branch (c = 1.04).
%   (A) The real-to-complex transition resolved directly on the reduced 2-D
%       return-map Jacobian A over s in [0.130, 0.156] at ds = 4e-4:
%       tr A, det A, the discriminant Delta = (tr A)^2 - 4 det A, and a check
%       that Delta crosses zero linearly in v (square-root coalescence).
%   (B) The half-life at the slowest gait computed (s = 0.010), verified by
%       iterating the nonlinear return map for 5000 steps along two
%       perturbation directions and measuring the modal amplitude
%       a_n = l1^T (z_n - z*) with the dominant left eigenvector.
%   Port of code/run_transition_and_slowend.py. Writes
%       data_matlab/transition_spectrum.csv, data_matlab/slowend_modal.csv,
%       data_matlab/_fp_s0010.json (cache of the s = 0.010 fixed point),
%       figures_matlab/fig_supp_transition.{png,pdf,tiff}
%   with the same schemas and number formats as the Python outputs.
C = 1.04;
c = wk.constants();
here = fileparts(mfilename('fullpath'));
DATA = fullfile(here, '..', 'data_matlab');
FIGS = fullfile(here, '..', 'figures_matlab');
if ~exist(DATA, 'dir'), mkdir(DATA); end
if ~exist(FIGS, 'dir'), mkdir(FIGS); end
t_all = tic;

% ---------------------------------------------------------------- (A) transition
fprintf('=== (A) transition spectrum on the reduced 2-D Jacobian ===\n');
rows = {};
seed = [];
s_grid = np_arange(0.1300, 0.1560, 0.0004);          % same floats as numpy.arange
for s = s_grid
    f = fp2d(s, seed, c.RTOL, c.ATOL, 1e-7, C, c);   % s here is the grid value of q
    if isempty(f), fprintf('  fixed point failed at q=%.4f\n', s); continue; end
    seed = f.z;
    sp = spectrum2d(f, c.RTOL, c.ATOL, c.DELTA, c);
    if isempty(sp), continue; end
    l1 = sp.lam(1); l2 = sp.lam(2);
    if sp.det > 0, sd = sqrt(sp.det); else, sd = NaN; end
    rows{end+1} = struct('q', f.q, 'alpha_p', f.alpha_p, 'alpha_h', f.alpha_h, ...
        's', f.s, 'v', f.v, 'T', f.T, 'tr', sp.tr, 'det', sp.det, ...
        'disc', sp.disc, 'l1_re', real(l1), 'l1_im', imag(l1), ...
        'l2_re', real(l2), 'l2_im', imag(l2), ...
        'absmax', max(abs(l1), abs(l2)), 'sqrt_det', sd); %#ok<SAGROW>
end
[~, order] = sort(cellfun(@(r) r.q, rows));
rows = rows(order);
l1im = cellfun(@(r) r.l1_im, rows);
vv_all = cellfun(@(r) r.v, rows);
cx = rows(abs(l1im) > 0);
re_ = rows(abs(l1im) == 0);
if ~isempty(re_), v_lo = max(cellfun(@(r) r.v, re_)); else, v_lo = NaN; end
if ~isempty(cx),  v_hi = min(cellfun(@(r) r.v, cx));  else, v_hi = NaN; end
fprintf('  transition bracketed in v = [%.6f, %.6f]  (width %.2e)\n', v_lo, v_hi, v_hi - v_lo);

% is the discriminant linear through zero?
near = rows(abs(vv_all - 0.5 * (v_lo + v_hi)) < 0.004);
vv = cellfun(@(r) r.v, near); dd = cellfun(@(r) r.disc, near);
p = polyfit(vv, dd, 1); sl = p(1); ic = p(2);
pred = sl * vv + ic;
r2 = 1.0 - sum((dd - pred).^2) / sum((dd - mean(dd)).^2);
v_zero = -ic / sl;
fprintf('  discriminant fit  Delta = %.5f*(v) + %.6f   R^2 = %.6f\n', sl, ic, r2);
fprintf('  Delta = 0 at v = %.6f   (bracket midpoint %.6f)\n', v_zero, 0.5 * (v_lo + v_hi));
% (Im lambda)^2 should be linear in v above the transition, with slope -sl
cxn = cx(cellfun(@(r) r.v, cx) - v_hi < 0.004);
if numel(cxn) > 4
    vc = cellfun(@(r) r.v, cxn); im2 = cellfun(@(r) r.l1_im^2, cxn);
    p2 = polyfit(vc, im2, 1); s2 = p2(1); i2 = p2(2);
    pr2 = s2 * vc + i2;
    r2b = 1.0 - sum((im2 - pr2).^2) / sum((im2 - mean(im2)).^2);
    fprintf('  (Im lambda)^2 fit slope = %.5f  R^2 = %.6f   (expect slope = -Delta slope/4 = %.5f)\n', ...
            s2, r2b, -sl / 4.0);
end
dets = cellfun(@(r) r.det, rows);
fprintf('  det A over the neighbourhood: %.6f to %.6f\n', min(dets), max(dets));

fh = fopen(fullfile(DATA, 'transition_spectrum.csv'), 'w');
fprintf(fh, 'q,alpha_p,alpha_h,s,v,T,tr,det,disc,l1_re,l1_im,l2_re,l2_im,absmax,sqrt_det\n');
for i = 1:numel(rows)
    r = rows{i};
    fprintf(fh, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', ...
        pyrepr(r.q), pyrepr(r.alpha_p), pyrepr(r.alpha_h), pyrepr(r.s), pyrepr(r.v), ...
        pyrepr(r.T), pyrepr(r.tr), pyrepr(r.det), pyrepr(r.disc), pyrepr(r.l1_re), ...
        pyrepr(r.l1_im), pyrepr(r.l2_re), pyrepr(r.l2_im), pyrepr(r.absmax), pyrepr(r.sqrt_det));
end
fclose(fh);
fprintf('  wrote data_matlab/transition_spectrum.csv\n');

% ---------------------------------------------------------------- (B) slow end
fprintf('\n=== (B) slowest gait computed: modal verification of N_1/2 ===\n');
CACHE = fullfile(DATA, '_fp_s0010.json');
if exist(CACHE, 'file')
    f10 = jsondecode(fileread(CACHE));
    f10.z = f10.z(:);
else
    seed = [];
    for sc = np_arange(0.400, 0.0099, -0.001)
        g = fp2d(sc, seed, c.RTOL, c.ATOL, 1e-7, C, c);
        if isempty(g), error('continuation failed at s=%.4f', sc); end
        seed = g.z;
    end
    f10 = g;
    fh = fopen(CACHE, 'w');
    fprintf(fh, '{"z": [%s, %s, %s], "T": %s, "P": %s, "s": %s, "v": %s, "residual": %s}', ...
        pyrepr(f10.z(1)), pyrepr(f10.z(2)), pyrepr(f10.z(3)), pyrepr(f10.T), pyrepr(f10.P), ...
        pyrepr(f10.s), pyrepr(f10.v), pyrepr(f10.residual));
    fclose(fh);
end
fprintf('  s=%.4f v=%.6f T=%.4f P=%.3e  Newton residual=%.2e\n', ...
        f10.s, f10.v, f10.T, f10.P, f10.residual);

sp10 = spectrum2d(f10, c.RTOL, c.ATOL, c.DELTA, c);
lam1 = real(sp10.lam(1)); lam2 = real(sp10.lam(2));
r1 = real(sp10.right(:, 1)); l1 = real(sp10.left(1, :));
% eigenvector sign is arbitrary (LAPACK); fix r1(1) > 0 so the perturbation is
% applied in the same direction as the reference run (r1 = (0.6845, -0.7290))
if r1(1) < 0, r1 = -r1; l1 = -l1; end
N_lin = -log(2) / log(abs(lam1));
fprintf('  lambda_1=%.9f  lambda_2=%.9f   N_1/2(linear)=%.1f\n', lam1, lam2, N_lin);
fprintf('  dominant right eigenvector (theta, thetadot) = (%.4f, %.4f)\n', r1(1), r1(2));

% convergence across numerical settings
fprintf('\n  --- convergence of lambda_1 across numerical settings ---\n');
names = {}; vals = [];
for d = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    sp = spectrum2d(f10, c.RTOL, c.ATOL, d, c);
    names{end+1} = sprintf('delta=%.0e', d); vals(end+1) = abs(sp.lam(1)); %#ok<SAGROW>
end
for rt = [1e-10, 1e-11, 1e-12, 1e-13]
    ff = fp2d(0.010, f10.z, rt, 1e-14, 1e-7, C, c);
    names{end+1} = sprintf('rtol=%.0e', rt); %#ok<SAGROW>
    if isempty(ff), vals(end+1) = NaN; continue; end %#ok<SAGROW>
    sp = spectrum2d(ff, rt, 1e-14, c.DELTA, c);
    vals(end+1) = abs(sp.lam(1)); %#ok<SAGROW>
end
for i = 1:numel(names)
    fprintf('     %-12s lambda_1=%.9f  N_1/2=%8.1f\n', names{i}, vals(i), -log(2) / log(vals(i)));
end
lams = vals(isfinite(vals));
Ns = -log(2) ./ log(lams);
fprintf('     spread across settings: lambda_1 %.2e ; N_1/2 %.1f to %.1f\n', ...
        max(lams) - min(lams), min(Ns), max(Ns));

% nonlinear iteration, modal amplitude
NSTEPS = 5000;
out = {};
dirs = {'r1', 'theta'};
for di = 1:numel(dirs)
    direction = dirs{di};
    for amp = [0.001, 0.003, 0.010]
        d0 = amp * abs(f10.z(1));
        if strcmp(direction, 'r1')
            step = d0 * r1 / max(abs(r1(1)), 1e-300);     % scale so the theta part is d0
            th = f10.z(1) + step(1); thd = f10.z(2) + step(2);
        else
            th = f10.z(1) + d0; thd = f10.z(2);
        end
        phd = (1.0 - cos(2.0 * th)) * thd;
        z = [th; thd; phd];
        a0 = l1 * [z(1) - f10.z(1); z(2) - f10.z(2)];
        amps = zeros(1, NSTEPS + 1); amps(1) = 1.0; nA = 1;
        for it = 1:NSTEPS
            [z, ~, ok] = wk.step_map(z, c.GAM, c.KHIP, f10.P, c.RTOL, c.ATOL);
            if ~ok, nA = nA + 1; amps(nA) = NaN; break; end
            an = l1 * [z(1) - f10.z(1); z(2) - f10.z(2)];
            nA = nA + 1; amps(nA) = an / a0;
        end
        amps = amps(1:nA);
        n = 0:(numel(amps) - 1);
        good = isfinite(amps) & (abs(amps) > 0);
        tail = good & (n > floor(numel(amps) / 3));
        pf = polyfit(n(tail), log(abs(amps(tail))), 1); sl2 = pf(1); ic2 = pf(2);
        lam_fit = exp(sl2); N_fit = -log(2) / log(lam_fit);
        below = find(abs(amps) <= 0.5, 1);
        if isempty(below), n_half_obs = []; nh_str = 'None'; nh_csv = '';
        else, n_half_obs = below - 1; nh_str = sprintf('%d', n_half_obs); nh_csv = nh_str; end
        fprintf('  %-6s amp=%4.1f%%  lam_fit=%.9f  N_1/2(fit)=%7.1f  first|a|<0.5 at n=%s  A=%.3f\n', ...
                direction, amp * 100, lam_fit, N_fit, nh_str, exp(ic2));
        out{end+1} = struct('direction', direction, 'amp', amp, 'lam_linear', abs(lam1), ...
            'N_linear', N_lin, 'lam_fitted', lam_fit, 'N_fitted', N_fit, ...
            'n_half_observed', nh_csv, 'prefactor', exp(ic2), 'steps', numel(amps) - 1); %#ok<SAGROW>
    end
end
fh = fopen(fullfile(DATA, 'slowend_modal.csv'), 'w');
fprintf(fh, 'direction,amp,lam_linear,N_linear,lam_fitted,N_fitted,n_half_observed,prefactor,steps\n');
for i = 1:numel(out)
    o = out{i};
    fprintf(fh, '%s,%s,%s,%s,%s,%s,%s,%s,%d\n', o.direction, pyrepr(o.amp), ...
        pyrepr(o.lam_linear), pyrepr(o.N_linear), pyrepr(o.lam_fitted), pyrepr(o.N_fitted), ...
        o.n_half_observed, pyrepr(o.prefactor), o.steps);
end
fclose(fh);
fprintf('  wrote data_matlab/slowend_modal.csv\n');

% ---------------------------------------------------------------- figure
v = cellfun(@(r) r.v, rows);
g15 = [0.15 0.15 0.15]; g55 = [0.55 0.55 0.55];
fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [1 1 9.6 6.4], 'Color', 'w');
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
% Text uses the TeX interpreter with Helvetica so that it resembles matplotlib's
% sans-serif mathtext (upright operators, italic variables).
a0 = nexttile(tl); hold(a0, 'on');
plot(a0, v, cellfun(@(r) r.l1_re, rows), '-', 'Color', g15, 'LineWidth', 1.6, 'DisplayName', 'Re \lambda');
plot(a0, v, cellfun(@(r) r.l2_re, rows), '-', 'Color', g15, 'LineWidth', 1.6, 'HandleVisibility', 'off');
plot(a0, v, cellfun(@(r) r.l1_im, rows), '--', 'Color', g55, 'LineWidth', 1.5, 'DisplayName', '\pm Im \lambda');
plot(a0, v, cellfun(@(r) r.l2_im, rows), '--', 'Color', g55, 'LineWidth', 1.5, 'HandleVisibility', 'off');
ylabel(a0, 'multiplier');
legend(a0, 'FontSize', 8, 'Box', 'off', 'Location', 'east');
title(a0, '(a) the two non-zero multipliers coalesce');
a1 = nexttile(tl); hold(a1, 'on');
plot(a1, v, cellfun(@(r) r.tr, rows), '-', 'Color', g15, 'LineWidth', 1.6, 'DisplayName', 'tr {\itJ}_r');
plot(a1, v, cellfun(@(r) r.det, rows), '-', 'Color', g55, 'LineWidth', 1.6, 'DisplayName', 'det {\itJ}_r');
legend(a1, 'FontSize', 8, 'Box', 'off', 'Location', 'east');
title(a1, '(b) tr {\itJ}_r and det {\itJ}_r = \lambda_1\lambda_2');
a2 = nexttile(tl); hold(a2, 'on');
plot(a2, v, cellfun(@(r) r.disc, rows), '-', 'Color', g15, 'LineWidth', 1.7);
yline(a2, 0.0, '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 0.9);
xline(a2, v_zero, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.1);
xlabel(a2, 'dimensionless speed {\itv}'); ylabel(a2, '\Delta');
title(a2, '(c) discriminant \Delta = (tr {\itJ}_r)^2 - 4 det {\itJ}_r crosses zero');
a3 = nexttile(tl); hold(a3, 'on');
plot(a3, cellfun(@(r) r.v, cx), cellfun(@(r) r.l1_im^2, cx), 'o', 'MarkerSize', 3, ...
     'MarkerFaceColor', g15, 'MarkerEdgeColor', g15, 'LineStyle', 'none');
xlabel(a3, 'dimensionless speed {\itv}'); ylabel(a3, '(Im \lambda)^2');
title(a3, '(d) (Im \lambda)^2 linear in {\itv} = square-root coalescence');
for a = [a0 a1 a2 a3]
    a.Title.FontSize = 9; a.Title.FontWeight = 'normal';
    a.TitleHorizontalAlignment = 'left';
    a.FontName = 'Helvetica'; a.FontSize = 10;
    a.Box = 'off'; a.TickDir = 'out'; a.Layer = 'bottom';
    a.XGrid = 'on'; a.YGrid = 'on'; a.GridColor = [0.69 0.69 0.69]; a.GridAlpha = 0.25;
    a.LineWidth = 0.8;
    % matplotlib's default 5 % data margins in x
    xl = [Inf -Inf];
    for h = a.Children'
        if isprop(h, 'XData') && numel(h.XData) > 1
            xl = [min(xl(1), min(h.XData)) max(xl(2), max(h.XData))];
        end
    end
    a.XLim = xl + 0.05 * diff(xl) * [-1 1];
end
set(fig, 'PaperUnits', 'inches', 'PaperSize', [9.6 6.4], 'PaperPosition', [0 0 9.6 6.4]);
base = fullfile(FIGS, 'fig_supp_transition');
print(fig, [base '.png'], '-dpng', '-r300');
print(fig, [base '.pdf'], '-dpdf', '-r300');
print(fig, [base '.tiff'], '-dtiff', '-r300');
fprintf('  wrote %s.{png,pdf,tiff}\n', base);
close(fig);
fprintf('total %.1f s\n', toc(t_all));

% ================================================================ local functions
function a = np_arange(start, stop, step)
%NP_ARANGE  Reproduce numpy.arange floats: delta = (start+step)-start, a_i = start + i*delta.
    n = ceil((stop - start) / step);
    delta = (start + step) - start;
    a = start + (0:n-1) * delta;
end

function f = fp2d(q, seed, rtol, atol, delta, C, c)
%FP2D  Fixed point of the map at branch parameter q, with T, P and the one-step
%   residual. q prescribes the push-off; the realised step length is
%   s = 2 sin(alpha_h) with alpha_h = z(1).
    a = asin(0.5 * q); P = C * a * tan(a);
    if isempty(seed)
        z0 = [a; -C * a; (1 - cos(2 * a)) * (-C * a)];
    else
        z0 = seed(:);
    end
    [z, T, ok] = wk.find_fixed_point(z0, c.GAM, c.KHIP, P, rtol, atol, 1e-12, 20, delta);
    if ~ok, f = []; return; end
    [Sz, ~, ok2] = wk.step_map(z, c.GAM, c.KHIP, P, rtol, atol);
    if ok2, res = norm(Sz - z); else, res = NaN; end
    alpha_h = z(1); s = 2.0 * sin(alpha_h);
    f = struct('z', z, 'T', T, 'P', P, 'q', q, 'alpha_p', a, 'alpha_h', alpha_h, ...
               's', s, 'v', s / T, 'residual', res);
end

function sp = spectrum2d(f, rtol, atol, delta, c)
%SPECTRUM2D  tr, det, discriminant and eigen-decomposition of the reduced 2-D Jacobian.
    u = [f.z(1); f.z(2)];
    A = wk.jacobian_2d(u, c.GAM, c.KHIP, f.P, rtol, atol, delta);
    if isempty(A), sp = []; return; end
    tr = trace(A); dt = det(A);
    disc = tr * tr - 4.0 * dt;
    [V, D] = eig(A); w = diag(D);
    [~, idx] = sortrows([-abs(w), -imag(w)]);      % descending |lambda|, +Im first
    w = w(idx); V = V(:, idx);
    Vinv = inv(V);                                  % rows are the left eigenvectors
    sp = struct('A', A, 'tr', tr, 'det', dt, 'disc', disc, 'lam', w, 'right', V, 'left', Vinv);
end

function str = pyrepr(x)
%PYREPR  Shortest decimal that round-trips, as Python's repr(float) prints it.
    if isnan(x), str = 'nan'; return; end
    if isinf(x), if x > 0, str = 'inf'; else, str = '-inf'; end, return; end
    for p = 1:17
        str = sprintf('%.*g', p, x);
        if str2double(str) == x, break; end
    end
    % Python uses exponent notation only below 1e-4 or at/above 1e16; %g switches at
    % the precision, so re-express in fixed notation where Python would.
    ax = abs(x);
    if ax ~= 0 && ax >= 1e-4 && ax < 1e16 && contains(str, 'e')
        for p = 1:20
            str = sprintf('%.*f', p, x);
            if str2double(str) == x, break; end
        end
    end
    if ~isempty(regexp(str, '^-?\d+$', 'once')), str = [str '.0']; end
    str = regexprep(str, 'e([+-])(\d)$', 'e$10$2');  % e-7 -> e-07 as Python prints
end
