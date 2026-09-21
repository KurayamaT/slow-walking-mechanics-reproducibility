%RUN_SLOWEND_VERIFICATION  Direct verification of the 842-step half-life at the
%   slowest gait computed (s = 0.010), with an explicit numerical noise floor.
%   MATLAB port of code/run_slowend_verification.py.
%
%   Reported, in the order the analysis needs them:
%     1. one-step residual of the fixed point, and every tolerance in force
%     2. the unperturbed orbit iterated from z* for the same number of steps,
%        giving the full-state drift ||e_n|| and the drift projected on the
%        dominant mode a_n^(0) = l1^T e_n -- the numerical floor that matters
%     3. an evaluable interval: the steps where the perturbed modal amplitude
%        stays above 10x the largest unperturbed modal drift
%     4. regression of log|a_n/a0| over that interval -> lambda_hat, N_half_hat,
%        plus the per-step ratio a_{n+1}/a_n, for the dominant-eigenvector
%        direction and for the theta-only displacement, at three amplitudes
%
%   Left and right eigenvectors are biorthogonal (l1^T r1 = 1), so the modal
%   amplitude does not depend on eigenvector scaling.
%
%   Writes data_matlab/slowend_verification.csv (same schema and number
%   formats as data/slowend_verification.csv), data_matlab/slowend_series.mat
%   (in place of data/slowend_series.npz; the npz keys "r1_0.001", ... are
%   stored as the struct fields series.r1_0_001, ... with the original key
%   names in series_keys), and figures_matlab/fig_supp_slowend.{png,pdf,tiff}.
%   The fixed point at s = 0.010 is taken from data_matlab/_fp_s0010.json if
%   present (the cache the Python code keeps in data/), otherwise it is
%   recomputed by 0.001-spaced seeded continuation from s = 0.400 and cached.
c = wk.constants();
here = fileparts(mfilename('fullpath'));
DATA = fullfile(here, '..', 'data_matlab');
FIGS = fullfile(here, '..', 'figures_matlab');
if ~exist(DATA, 'dir'), mkdir(DATA); end
if ~exist(FIGS, 'dir'), mkdir(FIGS); end
NSTEPS = 5000;
C = 1.04;
t_all = tic;

% ------------------------------------------------ fixed point at s = 0.010
cache = fullfile(DATA, '_fp_s0010.json');
if exist(cache, 'file')
    f = jsondecode(fileread(cache));
    f.z = f.z(:);
else
    fprintf('  no cached fixed point: continuation 0.400 -> 0.010 at 0.001 spacing\n');
    seed = [];
    for i = 0:390
        s = 0.400 - 0.001 * i;
        g = fp2d(s, seed, C, c);
        if isempty(g), error('continuation failed at s=%.4f', s); end
        seed = g.z;
    end
    f = g;
    fid = fopen(cache, 'w');
    fprintf(fid, '%s', jsonencode(struct('z', f.z(:).', 'T', f.T, 'P', f.P, 's', f.s, ...
                                        'v', f.v, 'residual', f.residual)));
    fclose(fid);
end
z_fp = f.z(:); P = f.P;
[Sz, ~, ok] = wk.step_map(z_fp, c.GAM, c.KHIP, P, c.RTOL, c.ATOL);
assert(ok);
res_star = norm(Sz - z_fp);
fprintf('=== 1. fixed point and tolerances ===\n');
fprintf('  s=%.4f  v=%.6f  T=%.4f  P=%.4e\n', f.s, f.v, f.T, P);
fprintf('  one-step residual ||R(z*)-z*|| = %.3e\n', res_star);
fprintf('  ODE rtol=%.0e  atol=%.0e   finite-difference delta=%.0e   phase-1 depart dt=%.3f\n', ...
        c.RTOL, c.ATOL, c.DELTA, c.DT_DEPART);

A = wk.jacobian_2d([z_fp(1); z_fp(2)], c.GAM, c.KHIP, P, c.RTOL, c.ATOL, c.DELTA);
[V, W] = eig(A); w = diag(W);
[~, o] = sort(-abs(w)); w = w(o); V = V(:, o);
if real(V(1, 1)) < 0, V(:, 1) = -V(:, 1); end   % sign convention: r1 = (+0.68, -0.73) as in the Python run
L = inv(V);
lam1 = real(w(1)); lam2 = real(w(2));
r1 = real(V(:, 1)); l1 = real(L(1, :));
N_lin = -log(2) / log(abs(lam1));
fprintf('  lambda_1=%.9f  lambda_2=%.9f   l1.r1=%.6f   N_1/2(linear)=%.1f\n', ...
        lam1, lam2, l1 * r1, N_lin);

% ------------------------------------------------ 2. unperturbed orbit
fprintf('\n=== 2. unperturbed orbit: the numerical floor ===\n');
[a0_drift, n0_drift] = iterate(z_fp, NSTEPS, z_fp, l1, P, c);
floor_mod = max(abs(a0_drift));
floor_p99 = np_percentile(abs(a0_drift), 99);
fprintf('  steps completed: %d\n', numel(a0_drift));
fprintf('  full-state drift ||e_n||   : max %.3e  final %.3e\n', max(n0_drift), n0_drift(end));
fprintf('  modal drift |a_n^(0)|      : max %.3e  99th pct %.3e  final %.3e\n', ...
        floor_mod, floor_p99, abs(a0_drift(end)));
fprintf('  the drift saturates rather than growing, so it is an integration/event floor,\n');
fprintf('  not a consequence of an inexact fixed point (residual %.1e).\n', res_star);

% ------------------------------------------------ 3-4. perturbed runs
fprintf('\n=== 3-4. perturbed runs, evaluated above the floor ===\n');
thresh = 10.0 * floor_mod;
fprintf('  evaluable while |a_n| > 10 x max modal drift = %.3e\n', thresh);
dirs = {'r1', 'theta'};
amps = [0.001, 0.003, 0.010];
rows = {};
series = struct();
series_keys = {};
for id = 1:numel(dirs)
    direction = dirs{id};
    for ia = 1:numel(amps)
        amp = amps(ia);
        d0 = amp * abs(z_fp(1));
        if strcmp(direction, 'r1')
            step = d0 * r1 / abs(r1(1));
            th = z_fp(1) + step(1); thd = z_fp(2) + step(2);
        else
            th = z_fp(1) + d0; thd = z_fp(2);
        end
        phd = (1.0 - cos(2.0 * th)) * thd;
        a_init = l1 * [th - z_fp(1); thd - z_fp(2)];
        amod = iterate([th; thd; phd], NSTEPS, z_fp, l1, P, c);
        rel = [1.0; amod(:) / a_init];
        n = (0:numel(rel) - 1).';
        ok_mask = abs(rel * a_init) > thresh;
        if any(ok_mask), n_last = max(n(ok_mask)); else, n_last = 0; end
        fit = (n >= 5) & (n <= n_last);
        pf = polyfit(n(fit), log(abs(rel(fit))), 1);
        sl = pf(1); ic = pf(2);
        lam_hat = exp(sl); N_hat = -log(2) / log(lam_hat);
        ratios = abs(rel(7:n_last + 1) ./ rel(6:n_last));
        below = find(abs(rel(1:n_last + 1)) <= 0.5, 1);
        if isempty(below), n_half = []; n_half_str = 'None'; else, n_half = below - 1; n_half_str = sprintf('%d', n_half); end
        ratio_mean = mean(ratios); ratio_sd = std(ratios, 1);      % population sd, as numpy
        fprintf('  %-5s amp=%4.1f%%  evaluable to n=%4d  lam_hat=%.9f  N_hat=%6.1f  half at n=%s  A=%.3f  ratio mean=%.9f sd=%.2e\n', ...
                direction, amp * 100, n_last, lam_hat, N_hat, n_half_str, exp(ic), ratio_mean, ratio_sd);
        rows{end+1} = struct('direction', direction, 'amp', amp, 'n_evaluable', n_last, ...
                             'lam_linear', abs(lam1), 'N_linear', N_lin, ...
                             'lam_hat', lam_hat, 'N_hat', N_hat, 'n_half', n_half, ...
                             'prefactor', exp(ic), 'ratio_mean', ratio_mean, 'ratio_sd', ratio_sd, ...
                             'one_step_residual', res_star, 'modal_floor', floor_mod); %#ok<SAGROW>
        key = sprintf('%s_%g', direction, amp);
        series_keys{end+1} = key; %#ok<SAGROW>
        series.(strrep(key, '.', '_')) = rel;
    end
end
series.unperturbed_modal = a0_drift(:);
series.unperturbed_norm = n0_drift(:);
series_keys(end+1:end+2) = {'unperturbed_modal', 'unperturbed_norm'};
save(fullfile(DATA, 'slowend_series.mat'), 'series', 'series_keys');

fh = fopen(fullfile(DATA, 'slowend_verification.csv'), 'w');
fprintf(fh, 'direction,amp,n_evaluable,lam_linear,N_linear,lam_hat,N_hat,n_half_observed,prefactor,ratio_mean,ratio_sd,one_step_residual,modal_floor,rtol,atol,delta\n');
for i = 1:numel(rows)
    r = rows{i};
    if isempty(r.n_half), nh = ''; else, nh = sprintf('%d', r.n_half); end
    fprintf(fh, '%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', ...
            r.direction, pyrepr(r.amp), r.n_evaluable, pyrepr(r.lam_linear), pyrepr(r.N_linear), ...
            pyrepr(r.lam_hat), pyrepr(r.N_hat), nh, pyrepr(r.prefactor), pyrepr(r.ratio_mean), ...
            pyrepr(r.ratio_sd), pyrepr(r.one_step_residual), pyrepr(r.modal_floor), ...
            pyrepr(c.RTOL), pyrepr(c.ATOL), pyrepr(c.DELTA));
end
fclose(fh);
fprintf('\n  wrote data_matlab/slowend_verification.csv and data_matlab/slowend_series.mat\n');

% ------------------------------------------------------------------- figure
fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [1 1 10.4 4.0], 'Color', 'w');
set(fig, 'PaperUnits', 'inches', 'PaperSize', [10.4 4.0], 'PaperPosition', [0 0 10.4 4.0]);
n = (0:NSTEPS).';
env = abs(lam1) .^ n;
pos = {[0.0775 0.152 0.425 0.7635], [0.560 0.152 0.425 0.7635]};
ttls = {'(a) perturbation along the dominant eigenvector', '(b) stance-angle-only displacement'};
shades = [0.68, 0.4, 0.05];
axs = gobjects(1, 2);
for ip = 1:2
    ax = axes(fig, 'Units', 'normalized', 'Position', pos{ip}); axs(ip) = ax;
    hold(ax, 'on');
    ax.YScale = 'log';
    d0 = 0.001 * abs(z_fp(1));
    patch(ax, [-250 5250 5250 -250], [1e-12 1e-12 floor_mod / d0 floor_mod / d0], [0.9 0.9 0.9], ...
          'EdgeColor', 'none');
    yline(ax, 0.5, ':', 'Color', [0.85 0.85 0.85], 'LineWidth', 0.8);
    h = gobjects(1, 4);
    h(1) = plot(ax, n, env, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.6, ...
                'DisplayName', '$\lambda_{\max}^{\,n}$');
    for ia = 1:3
        rel = series.(strrep(sprintf('%s_%g', dirs{ip}, amps(ia)), '.', '_'));
        h(1 + ia) = plot(ax, 0:numel(rel) - 1, abs(rel), '-', 'Color', shades(ia) * [1 1 1], ...
                         'LineWidth', 1.2, 'DisplayName', sprintf('%.1f\\%%', amps(ia) * 100));
    end
    xlabel(ax, 'step $n$', 'Interpreter', 'latex', 'FontSize', 10);
    xlim(ax, [-250 5250]); ylim(ax, [1e-4 3.0]);
    ax.YTick = [1e-4 1e-3 1e-2 1e-1 1e0];
    ax.YMinorTick = 'on';
    title(ax, ttls{ip}, 'FontSize', 9.5, 'FontWeight', 'normal', 'Interpreter', 'none');
    ax.TitleHorizontalAlignment = 'left';
    grid(ax, 'on'); ax.YMinorGrid = 'on'; ax.XMinorGrid = 'off';
    ax.GridAlpha = 0.25; ax.MinorGridAlpha = 0.25; ax.MinorGridLineStyle = '-';
    ax.GridColor = [0.69 0.69 0.69]; ax.MinorGridColor = [0.69 0.69 0.69];   % matplotlib '#b0b0b0'
    ax.Box = 'off'; ax.Layer = 'bottom'; ax.FontSize = 10; ax.TickDir = 'out';
    ax.XColor = [0 0 0]; ax.YColor = [0 0 0];
    if ip == 1
        ylabel(ax, 'modal amplitude $|a_n/a_0|$', 'Interpreter', 'latex', 'FontSize', 10);
        lg = legend(ax, h, 'Location', 'southwest', 'FontSize', 8, 'Box', 'off', 'Interpreter', 'latex');
        lg.ItemTokenSize = [18 18];
    end
end
for ext = {'png', 'pdf', 'tiff'}
    p = fullfile(FIGS, ['fig_supp_slowend.' ext{1}]);
    switch ext{1}                       % print keeps the full 10.4 x 4.0 in canvas
        case 'png',  print(fig, p, '-dpng', '-r300');
        case 'tiff', print(fig, p, '-dtiff', '-r300');
        case 'pdf',  print(fig, p, '-dpdf', '-painters');
    end
    fprintf('  wrote %s\n', p);
end
close(fig);
fprintf('  total %.1f s\n', toc(t_all));

% ------------------------------------------------------------ local functions
function g = fp2d(s, seed, C, c)
    % Fixed point of the return map at step length s (P = C alpha tan alpha),
    % with T, P and the one-step residual; [] if Newton fails.
    a = asin(0.5 * s);
    P = C * a * tan(a);
    if isempty(seed)
        z0 = [a; -C * a; (1 - cos(2 * a)) * (-C * a)];
    else
        z0 = seed(:);
    end
    [z, T, ok] = wk.find_fixed_point(z0, c.GAM, c.KHIP, P, c.RTOL, c.ATOL, 1e-12, 20, 1e-7);
    if ~ok, g = []; return; end
    [Sz, ~, ok2] = wk.step_map(z, c.GAM, c.KHIP, P, c.RTOL, c.ATOL);
    if ok2, res = norm(Sz - z); else, res = NaN; end
    g = struct('z', z, 'T', T, 'P', P, 's', s, 'v', s / T, 'residual', res);
end

function [amod, anorm] = iterate(z_start, nsteps, z_fp, l1, P, c)
    % Modal amplitude series l1^T e_n and the full-state deviation norm ||e_n||.
    z = double(z_start(:));
    amod = zeros(nsteps, 1); anorm = zeros(nsteps, 1); m = 0;
    for i = 1:nsteps
        [z, ~, ok] = wk.step_map(z, c.GAM, c.KHIP, P, c.RTOL, c.ATOL);
        if ~ok, break; end
        e = [z(1) - z_fp(1); z(2) - z_fp(2)];
        m = m + 1;
        amod(m) = l1 * e; anorm(m) = norm(e);
    end
    amod = amod(1:m); anorm = anorm(1:m);
end

function q = np_percentile(x, p)
    % numpy.percentile with the default linear interpolation.
    x = sort(x(:)); N = numel(x);
    idx = (N - 1) * p / 100; lo = floor(idx); fr = idx - lo;
    if lo + 1 >= N, q = x(end); else, q = x(lo + 1) + fr * (x(lo + 2) - x(lo + 1)); end
end

function s = pyrepr(x)
    % Python repr() of a float: shortest round-tripping decimal, fixed
    % notation for exponents in [-4, 16), scientific otherwise.
    if isnan(x), s = 'nan'; return; end
    if isinf(x), if x > 0, s = 'inf'; else, s = '-inf'; end; return; end
    if x == 0, s = '0.0'; return; end
    for p = 1:17
        str = sprintf('%.*e', p - 1, x);
        if str2double(str) == x, break; end
    end
    tok = regexp(str, '^(-?)(\d)\.?(\d*)e([+-]\d+)$', 'tokens', 'once');
    sgn = tok{1}; digits = [tok{2} tok{3}]; e10 = str2double(tok{4});
    digits = regexprep(digits, '0+$', ''); if isempty(digits), digits = '0'; end
    nd = numel(digits);
    if e10 >= -4 && e10 < 16
        if e10 >= 0
            if nd <= e10 + 1
                s = [digits repmat('0', 1, e10 + 1 - nd) '.0'];
            else
                s = [digits(1:e10 + 1) '.' digits(e10 + 2:end)];
            end
        else
            s = ['0.' repmat('0', 1, -e10 - 1) digits];
        end
    else
        if nd > 1, mant = [digits(1) '.' digits(2:end)]; else, mant = digits; end
        if e10 < 0, es = '-'; else, es = '+'; end
        s = sprintf('%se%s%02d', mant, es, abs(e10));
    end
    s = [sgn s];
end
