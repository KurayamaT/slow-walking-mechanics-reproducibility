%PLOT_FIG_SUPP_DECOMPOSITION  Supplementary Figure S5: where the step-length dependence
%   of the dominant multiplier comes from. Reads data/jacobian_decomposition.csv and
%   data/jacobian_freeze_test.csv (run_jacobian_decomposition.m, run_freeze_test.m).
%   (a) d lambda_max / ds on the real regime, attributed to the reset factor and the
%       swing/event factor, with the reset factor split by collision angle, pre-impact
%       velocity and prescribed push-off. (b) Freeze test: 1 - lambda_max along the
%       branch, true, with the swing/event map frozen at s = 0.010, and with the reset
%       map frozen at s = 0.010.
here = fileparts(mfilename('fullpath'));
D = readtable(fullfile(here, '..', 'data_matlab', 'jacobian_decomposition.csv'));
F = readtable(fullfile(here, '..', 'data_matlab', 'jacobian_freeze_test.csv'));
figdir = fullfile(here, '..', 'figures_matlab');   % single MATLAB figure home
D = D(D.is_complex == 0 & D.q <= 0.130, :); D = sortrows(D, 'q');   % cut in q, as S6 does
F = sortrows(F, 'q');

f = figure('Units', 'inches', 'Position', [1 1 9.6 3.8], 'Color', 'w');
G = [0.9 0.9 0.9];

ax = subplot(1, 2, 1); hold(ax, 'on');
plot(ax, D.s, -D.dlam_dq_fd,  '-',  'Color', [0 0 0],       'LineWidth', 2.2, 'DisplayName', '$-\,d\lambda_{\max}/dq$ (finite difference)');
plot(ax, D.s, -D.c_reset,     '--', 'Color', [0.25 0.25 0.25], 'LineWidth', 1.5, 'DisplayName', 'reset factor $B$');
plot(ax, D.s, -D.c_reset_geom, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.6, 'DisplayName', '\quad of which collision angle $\alpha$');
plot(ax, D.s, -D.c_swing,     '-.', 'Color', [0.6 0.6 0.6],    'LineWidth', 1.5, 'DisplayName', 'swing/event factor $A$');
xlabel(ax, 'step length $s$', 'Interpreter', 'latex');
ylabel(ax, '$-\,d\lambda_{\max}/dq$', 'Interpreter', 'latex');
title(ax, '(a) attribution of the step-length dependence', 'FontSize', 10, 'FontWeight', 'normal');
ax.TitleHorizontalAlignment = 'left';
legend(ax, 'Interpreter', 'latex', 'FontSize', 8, 'Box', 'off', 'Location', 'northwest');
xlim(ax, [0 0.135]); grid(ax, 'on'); ax.GridColor = G; ax.GridAlpha = 1; box(ax, 'off');

ax = subplot(1, 2, 2); hold(ax, 'on');
semilogy(ax, F.s, 1 - F.lam_true,              '-',  'Color', [0 0 0],       'LineWidth', 2.2, 'DisplayName', 'full map');
semilogy(ax, F.s, 1 - F.lam_Afrozen_q010,  '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5, 'DisplayName', 'swing/event map frozen at the $q=0.010$ gait ($s=0.0099$)');
semilogy(ax, F.s, 1 - F.lam_Bfrozen_q010,  ':',  'Color', [0.5 0.5 0.5], 'LineWidth', 1.8, 'DisplayName', 'reset map frozen at the $q=0.010$ gait ($s=0.0099$)');
set(ax, 'YScale', 'log');
xlabel(ax, 'step length $s$', 'Interpreter', 'latex');
ylabel(ax, 'recovery margin $1-\lambda_{\max}$', 'Interpreter', 'latex');
title(ax, '(b) freeze test', 'FontSize', 10, 'FontWeight', 'normal');
ax.TitleHorizontalAlignment = 'left';
legend(ax, 'Interpreter', 'latex', 'FontSize', 8, 'Box', 'off', 'Location', 'northwest');
xlim(ax, [0 0.145]); ylim(ax, [5e-4 0.4]); grid(ax, 'on'); ax.GridColor = G; ax.GridAlpha = 1; box(ax, 'off');

base = fullfile(figdir, 'fig_supp_decomposition');
exportgraphics(f, [base '.png'], 'Resolution', 300);
exportgraphics(f, [base '.pdf'], 'ContentType', 'vector');
exportgraphics(f, [base '.tiff'], 'Resolution', 300);
close(f);
fprintf('saved: %s.{png,pdf,tiff}\n', base);
