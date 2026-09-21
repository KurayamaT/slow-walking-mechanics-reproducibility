%RUN_SUPP_VALIDATION  Supplementary S1 — numerical implementation and its validation.
%   Port of code/run_supp_validation.py.
%     (i)   convergence of |lambda_max| with the central-difference step delta
%     (ii)  insensitivity to the ODE tolerance rtol
%     (iii) 3-D post-impact Jacobian vs the reduced 2-D Jacobian
%     (iv)  the structurally zero third multiplier
%   Writes data_matlab/supp_validation.csv, supp_validation_delta.csv,
%          supp_validation_branch.csv and figures_matlab/fig_supp_validation.{png,pdf,tiff}
C = 1.04;
c = wk.constants();
GAITS  = [0.010, 0.080, 0.402];
LABELS = {'slowest, $q=0.010$', 'low speed, $q=0.080$', 'plateau, $q=0.402$'};
DELTAS = [1e-9 3e-9 1e-8 3e-8 1e-7 3e-7 1e-6 3e-6 1e-5 3e-5 1e-4];
RTOLS  = [1e-13 1e-12 1e-11 1e-10];

here = fileparts(mfilename('fullpath'));
datadir = fullfile(here, '..', 'data_matlab');
figdir  = fullfile(here, '..', 'figures_matlab');
if ~exist(datadir, 'dir'), mkdir(datadir); end
if ~exist(figdir, 'dir'),  mkdir(figdir);  end
tic;

% ---------------------------------------------- branch sweep (every 20th point)
fprintf('=== branch sweep: structural zero and 2-D/3-D agreement ===\n');
a0 = asin(0.5 * 0.40);
seed = [a0; -C * a0; (1 - cos(2 * a0)) * (-C * a0)];
[z0, ~, ok] = wk.find_fixed_point(seed, c.GAM, c.KHIP, push_off(0.40, C), c.RTOL, c.ATOL, 1e-12, 20, 1e-7);
assert(ok, 'anchor solve failed');
B = struct('s', {}, 'v', {}, 'lam3_abs', {}, 'dev_2d_3d', {}, 'residual', {});
REFS = containers.Map('KeyType', 'double', 'ValueType', 'any');
for dir_i = 1:2
    zc = z0;
    if dir_i == 1, sgrid = 0.399:-0.001:0.010; else, sgrid = 0.401:0.001:0.800; end
    for i = 1:numel(sgrid)
        st = sgrid(i); P = push_off(st, C);
        [z2, T2, ok2] = wk.find_fixed_point(zc, c.GAM, c.KHIP, P, c.RTOL, c.ATOL, 1e-12, 20, 1e-7);
        if ~ok2, break; end
        zc = z2;
        for gi = 1:numel(GAITS)
            if abs(st - GAITS(gi)) < 5e-4, REFS(GAITS(gi)) = {zc, T2}; end
        end
        if mod(i - 1, 20) ~= 0, continue; end   % Python records i % 20 == 0 (0-based)
        J3 = wk.jacobian_3d(zc, c.GAM, c.KHIP, P, c.RTOL, c.ATOL, 1e-7);
        J2 = wk.jacobian_2d(zc(1:2), c.GAM, c.KHIP, P, c.RTOL, c.ATOL, 1e-7);
        if isempty(J3) || isempty(J2), continue; end
        l3 = wk.sorted_eigs(J3); l2 = wk.sorted_eigs(J2);
        [Sz, ~, ~] = wk.step_map(zc, c.GAM, c.KHIP, P, c.RTOL, c.ATOL);
        B(end+1) = struct('s', st, 'v', st / T2, 'lam3_abs', abs(l3(3)), ...
                          'dev_2d_3d', max(abs(sortc(l3(1:2)) - sortc(l2))), ...
                          'residual', norm(Sz - zc)); %#ok<SAGROW>
    end
end
[~, ord] = sort([B.v]); B = B(ord);
fprintf('recorded %d points, v in [%.5f, %.5f]\n', numel(B), B(1).v, B(end).v);
fh = fopen(fullfile(datadir, 'supp_validation_branch.csv'), 'w');
fprintf(fh, 's,v,lam3_abs,dev_2d_3d,residual\n');
for i = 1:numel(B)
    fprintf(fh, '%.17g,%.17g,%.17g,%.17g,%.17g\n', B(i).s, B(i).v, B(i).lam3_abs, B(i).dev_2d_3d, B(i).residual);
end
fclose(fh);

% ------------------------------------- delta / rtol convergence at three gaits
fprintf('=== delta / rtol convergence at three reference gaits ===\n');
D = struct('s', {}, 'v', {}, 'kind', {}, 'setting', {}, 'lam_max', {}, 'N_half', {});
S = struct('s', {}, 'v', {}, 'T', {}, 'P', {}, 'residual', {}, 'lam1', {}, 'lam2', {}, ...
           'lam3_abs', {}, 'dev_2d_3d', {}, 'N_half', {}, 'N_half_delta1e5', {});
for gi = 1:numel(GAITS)
    s = GAITS(gi); r = REFS(s); z = r{1}; T = r{2};
    P = push_off(s, C); v = s / T;
    [Sz, ~, ~] = wk.step_map(z, c.GAM, c.KHIP, P, c.RTOL, c.ATOL);
    res = norm(Sz - z);
    fprintf('s=%.3f  v=%.6f  T=%.4f  P=%.4e  ||R(z*)-z*||=%.2e\n', s, v, T, P, res);
    for d = DELTAS
        lam = wk.sorted_eigs(wk.jacobian_3d(z, c.GAM, c.KHIP, P, c.RTOL, c.ATOL, d));
        D(end+1) = struct('s', s, 'v', v, 'kind', 'delta', 'setting', d, ...
                          'lam_max', abs(lam(1)), 'N_half', -log(2) / log(abs(lam(1)))); %#ok<SAGROW>
    end
    for rt = RTOLS
        lam = wk.sorted_eigs(wk.jacobian_3d(z, c.GAM, c.KHIP, P, rt, c.ATOL, 1e-7));
        D(end+1) = struct('s', s, 'v', v, 'kind', 'rtol', 'setting', rt, ...
                          'lam_max', abs(lam(1)), 'N_half', -log(2) / log(abs(lam(1)))); %#ok<SAGROW>
    end
    l3 = wk.sorted_eigs(wk.jacobian_3d(z, c.GAM, c.KHIP, P, c.RTOL, c.ATOL, 1e-7));
    l2 = wk.sorted_eigs(wk.jacobian_2d(z(1:2), c.GAM, c.KHIP, P, c.RTOL, c.ATOL, 1e-7));
    dev = max(abs(sortc(l3(1:2)) - sortc(l2)));
    lam5 = wk.sorted_eigs(wk.jacobian_3d(z, c.GAM, c.KHIP, P, c.RTOL, c.ATOL, 1e-5));
    S(end+1) = struct('s', s, 'v', v, 'T', T, 'P', P, 'residual', res, ...
                      'lam1', l3(1), 'lam2', l3(2), 'lam3_abs', abs(l3(3)), 'dev_2d_3d', dev, ...
                      'N_half', -log(2) / log(abs(l3(1))), ...
                      'N_half_delta1e5', -log(2) / log(abs(lam5(1)))); %#ok<SAGROW>
    fprintf('   |lambda_3| = %.2e   max|lambda(3D)-lambda(2D)| = %.2e\n', abs(l3(3)), dev);
end
fh = fopen(fullfile(datadir, 'supp_validation_delta.csv'), 'w');
fprintf(fh, 's,v,kind,setting,lam_max,N_half\n');
for i = 1:numel(D)
    fprintf(fh, '%.17g,%.17g,%s,%.17g,%.17g,%.17g\n', D(i).s, D(i).v, D(i).kind, D(i).setting, D(i).lam_max, D(i).N_half);
end
fclose(fh);
fh = fopen(fullfile(datadir, 'supp_validation.csv'), 'w');
fprintf(fh, 's,v,T,P,residual,lam1,lam2,lam3_abs,dev_2d_3d,N_half,N_half_delta1e5\n');
for i = 1:numel(S)
    fprintf(fh, '%.17g,%.17g,%.17g,%.17g,%.17g,%s,%s,%.17g,%.17g,%.17g,%.17g\n', ...
            S(i).s, S(i).v, S(i).T, S(i).P, S(i).residual, pyc(S(i).lam1), pyc(S(i).lam2), ...
            S(i).lam3_abs, S(i).dev_2d_3d, S(i).N_half, S(i).N_half_delta1e5);
end
fclose(fh);

% ------------------------------------------------------------------- figure S1
COL = [0.55 0.55 0.55; 0.25 0.25 0.25; 0 0 0];
LS  = {'-', '--', '-.'}; MK = {'o', 's', '^'};
f = figure('Units', 'inches', 'Position', [1 1 9.6 3.7], 'Color', 'w');
ax = subplot(1, 2, 1); hold(ax, 'on');
for gi = 1:numel(GAITS)
    m = strcmp({D.kind}, 'delta') & [D.s] == GAITS(gi);
    x = [D(m).setting]; y = [D(m).lam_max];
    ref = y([D(m).setting] == 1e-9);
    loglog(ax, x, max(abs(y - ref) / ref, 1e-17), LS{gi}, 'Color', COL(gi, :), ...
           'Marker', MK{gi}, 'MarkerSize', 3.5, 'LineWidth', 1.3, 'DisplayName', LABELS{gi});
end
set(ax, 'XScale', 'log', 'YScale', 'log');
xline(ax, 1e-7, ':', 'Color', [0.7 0.7 0.7], 'LineWidth', 1.2, 'HandleVisibility', 'off');
text(ax, 1.25e-7, 3e-13, '$\delta$ used', 'Interpreter', 'latex', 'FontSize', 8.5, 'Color', [0.35 0.35 0.35]);
xlabel(ax, 'central-difference step $\delta$', 'Interpreter', 'latex');
ylabel(ax, '$|\lambda_{\max}(\delta)-\lambda_{\max}(10^{-9})|/\lambda_{\max}(10^{-9})$', 'Interpreter', 'latex');
title(ax, '(a) finite-difference convergence', 'FontSize', 10, 'FontWeight', 'normal');
ax.TitleHorizontalAlignment = 'left';
legend(ax, 'Interpreter', 'latex', 'FontSize', 8, 'Box', 'off', 'Location', 'southwest');
grid(ax, 'on'); ax.GridColor = [0.9 0.9 0.9]; ax.GridAlpha = 1; ax.LineWidth = 0.4; box(ax, 'off');

ax = subplot(1, 2, 2); hold(ax, 'on');
vv = [B.v];
semilogy(ax, vv, max([B.dev_2d_3d], 1e-18), '-',  'Color', [0 0 0],       'LineWidth', 1.4, 'DisplayName', '$\max_i|\lambda_i^{\,3D}-\lambda_i^{\,2D}|$');
semilogy(ax, vv, max([B.lam3_abs],  1e-18), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.4, 'DisplayName', '$|\lambda_3|$ (structurally zero)');
semilogy(ax, vv, max([B.residual],  1e-18), ':',  'Color', [0.3 0.3 0.3], 'LineWidth', 1.4, 'DisplayName', '$\|R(\mathbf{z}^{*})-\mathbf{z}^{*}\|$');
set(ax, 'YScale', 'log'); ylim(ax, [1e-16 1e-6]);
xlabel(ax, 'dimensionless speed $v$', 'Interpreter', 'latex');
ylabel(ax, 'magnitude');
title(ax, '(b) reduction and structural zero, along the branch', 'FontSize', 10, 'FontWeight', 'normal');
ax.TitleHorizontalAlignment = 'left';
legend(ax, 'Interpreter', 'latex', 'FontSize', 8, 'Box', 'off', 'Location', 'southeast');
grid(ax, 'on'); ax.GridColor = [0.9 0.9 0.9]; ax.GridAlpha = 1; ax.LineWidth = 0.4; box(ax, 'off');

base = fullfile(figdir, 'fig_supp_validation');
exportgraphics(f, [base '.png'], 'Resolution', 300);
exportgraphics(f, [base '.pdf'], 'ContentType', 'vector');
exportgraphics(f, [base '.tiff'], 'Resolution', 300);
close(f);
fprintf('saved: %s.{png,pdf,tiff}\n', base);

fprintf('\n=== summary (for Table S1) ===\n');
for i = 1:numel(S)
    fprintf('s=%.3f v=%.6f  N_1/2=%.1f  N_1/2(delta=1e-5)=%.1f  |lam3|=%.1e  dev=%.1e\n', ...
            S(i).s, S(i).v, S(i).N_half, S(i).N_half_delta1e5, S(i).lam3_abs, S(i).dev_2d_3d);
end
fprintf('total %.1f s\n', toc);

function P = push_off(s, C)
    a = asin(0.5 * s);
    P = C * a * tan(a);
end

function w = sortc(v)
    % numpy.sort_complex: sort by real part, then imaginary part
    [~, i] = sortrows([real(v(:)), imag(v(:))]);
    w = v(i);
end

function str = pyc(z)
    % Render a complex number the way Python's csv writer renders complex()
    if imag(z) >= 0
        str = sprintf('(%.17g+%.17gj)', real(z), imag(z));
    else
        str = sprintf('(%.17g-%.17gj)', real(z), abs(imag(z)));
    end
end
