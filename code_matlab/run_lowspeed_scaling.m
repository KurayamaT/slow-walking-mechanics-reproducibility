%RUN_LOWSPEED_SCALING  Low-speed scaling of the recovery margin. MATLAB port of
%   code/run_lowspeed_scaling.py.
%
%   Descriptive fits, not statistical tests: the continuation points are a dense
%   deterministic sequence, not independent samples, so R^2 reports goodness of
%   fit and nothing more. The exponent drifts with the fitting range, which is
%   the point worth recording -- the quadratic behaviour is local to the slow
%   end, not a law for the whole branch.
%
%   Reads  data_matlab/master_fixed_khip_extended.csv (the MATLAB-generated branch).
%   Writes data_matlab/lowspeed_scaling.csv (same header and Python-repr number
%   format as data/lowspeed_scaling.csv), the two-panel reference figure,
%   figures_matlab/figure4_scaling{,_color}.{png,pdf,tiff}, and the single-panel
%   JBSE main Figure 4, figure4a_scaling_jbse{,_color}.{png,pdf,tiff}.
here = fileparts(mfilename('fullpath'));
DATA = fullfile(here, '..', 'data_matlab');
FIGS = fullfile(here, '..', 'figures_matlab');
figure_dir = getenv('SLOWWALK_SCALING_FIGURE_DIR');
if ~isempty(figure_dir), FIGS = figure_dir; end
if ~exist(FIGS, 'dir'), mkdir(FIGS); end

d = readtable(fullfile(DATA, 'master_fixed_khip_extended.csv'), 'TextType', 'string');
d = d(d.lam_max < 1.0, :);
d = sortrows(d, 's');
s = d.s;
lam = d.lam_max;
marg = 1.0 - lam;
nhalf = -log(2.0) ./ log(lam);

plot_only = strcmp(getenv('SLOWWALK_SCALING_PLOT_ONLY'), '1');
if ~plot_only
fprintf('exponent depends on the fitting range -- the quadratic form is local to the slow end\n');
out = struct('s_max', {}, 'n', {}, 'exponent', {}, 'coefficient', {}, 'r_squared', {});
for smax = [0.03, 0.05, 0.10, 0.15, 0.30, 0.80]
    m = s <= smax;
    if sum(m) < 5, continue; end
    p = polyfit(log(s(m)), log(marg(m)), 1);
    sl = p(1); ic = p(2);
    pred = sl * log(s(m)) + ic;
    lm = log(marg(m));
    r2 = 1.0 - sum((lm - pred) .^ 2) / sum((lm - mean(lm)) .^ 2);
    fprintf('  s <= %.2f  n=%4d  exponent=%.4f  coefficient=%.4f  R2=%.6f\n', ...
            smax, sum(m), sl, exp(ic), r2);
    out(end+1) = struct('s_max', smax, 'n', sum(m), 'exponent', sl, ...
                        'coefficient', exp(ic), 'r_squared', r2); %#ok<SAGROW>
end
fh = fopen(fullfile(DATA, 'lowspeed_scaling.csv'), 'w');
fprintf(fh, 's_max,n,exponent,coefficient,r_squared\n');
for i = 1:numel(out)
    fprintf(fh, '%s,%d,%s,%s,%s\n', pyrepr(out(i).s_max), out(i).n, ...
            pyrepr(out(i).exponent), pyrepr(out(i).coefficient), pyrepr(out(i).r_squared));
end
fclose(fh);
end

ratio = marg ./ s .^ 2;
fprintf('\n(1-rho)/s^2 at the slow end: %.4f at s=%.6f, %.4f at s=%.6f\n', ...
        ratio(1), s(1), ratio(11), s(11));
nh_s2 = nhalf .* s .^ 2;
fprintf('N_half * s^2               : %.5f at s=%.6f\n', nh_s2(1), s(1));
fprintf('collision deficit (1-cos2a)/s^2 is exactly 0.5, so the ratio of coefficients is %.2f\n', ...
        ratio(1) / 0.5);

% fit actually drawn in panel (a): the s <= 0.05 descriptive fit
m05 = s <= 0.05;
if plot_only
    fits = readtable(fullfile(DATA, 'lowspeed_scaling.csv'));
    fit05 = fits(fits.s_max == 0.05, :);
    assert(height(fit05) == 1 && fit05.n == sum(m05));
    sl05 = fit05.exponent; C05 = fit05.coefficient;
else
    p05 = polyfit(log(s(m05)), log(marg(m05)), 1);
    sl05 = p05(1); C05 = exp(p05(2));
end

draw(s, marg, ratio, m05, sl05, C05, struct('data', [0.067 0.067 0.067], 'fit', [0.478 0.478 0.478], 'coll', [0.55 0.55 0.55]), '', FIGS, false);
draw(s, marg, ratio, m05, sl05, C05, struct('data', [0 114 178]/255, 'fit', [213 94 0]/255, 'coll', [0 158 115]/255), '_color', FIGS, false);
draw_panel_a(s, marg, m05, sl05, C05, struct('data', [0.067 0.067 0.067], 'fit', [0.478 0.478 0.478]), '', FIGS);
draw_panel_a(s, marg, m05, sl05, C05, struct('data', [0 114 178]/255, 'fit', [213 94 0]/255), '_color', FIGS);
if strcmp(getenv('SLOWWALK_SCALING_REFERENCE'), '1')
    draw(s, marg, ratio, m05, sl05, C05, struct('data', [0.067 0.067 0.067], 'fit', [0.478 0.478 0.478], 'coll', [0.55 0.55 0.55]), '', FIGS, true);
    draw(s, marg, ratio, m05, sl05, C05, struct('data', [0 114 178]/255, 'fit', [213 94 0]/255, 'coll', [0 158 115]/255), '_color', FIGS, true);
end

function draw(s, marg, ratio, m05, sl05, C05, P, suffix, FIGS, reference)
    FS = 10;                                   % matplotlib font.size
    W = 9.8; H = 3.9;                          % figsize (inches)
    fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [1 1 W H], 'Color', 'w');
    fig.PaperUnits = 'inches'; fig.PaperSize = [W H]; fig.PaperPosition = [0 0 W H];
    ax = gobjects(1, 2);
    for k = 1:2
        ax(k) = axes(fig, 'Units', 'normalized', 'FontName', 'Helvetica', 'FontSize', FS, ...
                     'LabelFontSizeMultiplier', 1, 'TitleFontSizeMultiplier', 1, ...
                     'TickDir', 'out', 'TickLength', [0.0035 0.0035], 'Box', 'off', ...
                     'TickLabelInterpreter', 'tex', 'Layer', 'bottom', ...
                     'GridAlpha', 0.25, 'MinorGridAlpha', 0.25, 'MinorGridLineStyle', '-', ...
                     'GridColor', [0 0 0], 'MinorGridColor', [0 0 0]);
        hold(ax(k), 'on');
    end

    % (a) whole branch, log-log: margin and the s<=0.05 fit
    a = ax(1);
    set(a, 'XScale', 'log', 'YScale', 'log');
    h1 = plot(a, s, marg, '-', 'Color', P.data, 'LineWidth', 1.7);
    h2 = plot(a, s(m05), C05 * s(m05) .^ sl05, '--', 'Color', P.fit, 'LineWidth', 1.6);
    if reference
        h3 = plot(a, s, s .^ 2 / 2, ':', 'Color', P.coll, 'LineWidth', 1.5);
    end
    xlabel(a, 'step length {\its}', 'Interpreter', 'tex');
    ylabel(a, '1 - |\lambda_{max}|', 'Interpreter', 'tex');
    xlim(a, mpl_loglim(s));
    ylim(a, mpl_loglim([marg; C05 * s(m05) .^ sl05]));
    legend(a, [h1 h2], {'1 - |\lambda_{max}|', ...
           sprintf('fit for {\\its} \\leq 0.05: 1 - |\\lambda_{max}| = %.2f {\\its}^{%.3f}', C05, sl05)}, ...
           'FontSize', 8, 'Box', 'off', 'Location', 'southeast', 'Interpreter', 'tex');
    if reference
        ylim(a, mpl_loglim([marg; C05 * s(m05) .^ sl05; s .^ 2 / 2]));
        legend(a, [h1 h2 h3], {'1 - |\lambda_{max}|', ...
               sprintf('fit for {\\its} \\leq 0.05: 1 - |\\lambda_{max}| = %.2f {\\its}^{%.3f}', C05, sl05), ...
               'collision deficit {\its}^2/2'}, 'FontSize', 8, 'Box', 'off', 'Location', 'southeast', 'Interpreter', 'tex');
    end
    grid(a, 'on'); a.XMinorGrid = 'on'; a.YMinorGrid = 'on';
    lefttitle(a, '(a) recovery margin is nearly quadratic at low {\its}', 9.5);
    if reference
        lefttitle(a, '(a) both quadratic at low {\its}, with different coefficients', 9.5);
    end

    % (b) slow end only, linear: the local ratio
    b = ax(2);
    lo = s <= 0.05;
    h4 = plot(b, s(lo), ratio(lo), '-', 'Color', P.data, 'LineWidth', 1.8);
    if reference
        plot(b, [0.01 0.05], [0.5 0.5], ':', 'Color', P.coll, 'LineWidth', 1.3, 'HandleVisibility', 'off');
        text(b, 0.0125, 1.1, '(1 - cos 2\alpha)/{\its}^2 = 0.5 exactly', 'FontSize', 8.5, 'Color', P.coll, ...
             'Interpreter', 'tex', 'FontName', 'Helvetica', 'VerticalAlignment', 'baseline');
    end
    xlim(b, [0.01 0.05]); ylim(b, [0 10]);
    xticks(b, 0.01:0.005:0.05); xtickformat(b, '%.3f'); yticks(b, 0:2:10);
    xlabel(b, 'step length {\its}', 'Interpreter', 'tex');
    ylabel(b, '(1 - |\lambda_{max}|)/{\its}^2', 'Interpreter', 'tex');
    legend(b, h4, {'(1 - |\lambda_{max}|)/{\its}^2'}, 'FontSize', 8, 'Box', 'off', ...
           'Location', 'northeast', 'Interpreter', 'tex');
    grid(b, 'on');
    lefttitle(b, '(b) local ratio approaches 8.43 at low {\its}', 9.5);
    if reference
        lefttitle(b, '(b) the coefficient against the exact collision value', 9.5);
    end

    % tight_layout(): pad = 1.08 * font.size points, two columns of equal width
    drawnow;
    padx = 1.08 * FS / 72 / W; pady = 1.08 * FS / 72 / H;
    for k = 1:2
        ti = ax(k).TightInset;
        x0 = (k - 1) * 0.5;
        ax(k).Position = [x0 + ti(1) + padx, ti(2) + pady, 0.5 - ti(1) - ti(3) - 2 * padx, 1 - ti(2) - ti(4) - 2 * pady];
    end
    drawnow;

    % annotate("~8.23 at the slow end", (s[0], ratio[0]), offset (16, -26) points, plain line)
    pt = [1 / (72 * W), 1 / (72 * H)];                 % one point in figure units
    P_xy = data2fig(b, s(1), ratio(1));
    P_txt = P_xy + [16, -26] .* pt;
    text(b, s(1), ratio(1), sprintf('\\approx %.2f at the slow end', ratio(1)), 'Units', 'data', ...
         'FontSize', 8.5, 'Color', P.data, 'Interpreter', 'tex', 'FontName', 'Helvetica', ...
         'HorizontalAlignment', 'left', 'VerticalAlignment', 'baseline', ...
         'Position', fig2data(b, P_txt));
    % matplotlib joins xy to the centre of the text box, clipped at the box edge:
    % for a 95 pt wide, 9 pt high box that is the top edge, at offset (45, -16) pt
    box_w = 95; box_lo = -2; box_hi = 7;
    ctr = [16 + box_w / 2, -26 + (box_lo + box_hi) / 2];
    tpar = (-26 + box_hi) / ctr(2);
    endp = P_xy + [ctr(1) * tpar, -26 + box_hi] .* pt;
    annotation(fig, 'line', [P_xy(1) endp(1)], [P_xy(2) endp(2)], 'Color', P.data, 'LineWidth', 0.8);

    base = fullfile(FIGS, ['figure4_scaling' suffix]);
    if reference
        base = fullfile(FIGS, ['scaling_collision_reference' suffix]);
    end
    print(fig, [base '.png'], '-dpng', '-r300');
    print(fig, [base '.tiff'], '-dtiff', '-r300');
    print(fig, [base '.pdf'], '-dpdf', '-vector');
    close(fig);
    fprintf('wrote %s.{png,pdf,tiff}\n', base);
end

function draw_panel_a(s, marg, m05, sl05, C05, P, suffix, FIGS)
    FS = 10;
    W = 5.8; H = 4.2;
    fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [1 1 W H], 'Color', 'w');
    fig.PaperUnits = 'inches'; fig.PaperSize = [W H]; fig.PaperPosition = [0 0 W H];
    ax = axes(fig, 'Units', 'normalized', 'FontName', 'Helvetica', 'FontSize', FS, ...
              'LabelFontSizeMultiplier', 1, 'TitleFontSizeMultiplier', 1, ...
              'TickDir', 'out', 'TickLength', [0.0035 0.0035], 'Box', 'off', ...
              'TickLabelInterpreter', 'tex', 'Layer', 'bottom', ...
              'GridAlpha', 0.25, 'MinorGridAlpha', 0.25, 'MinorGridLineStyle', '-', ...
              'GridColor', [0 0 0], 'MinorGridColor', [0 0 0]);
    hold(ax, 'on');
    set(ax, 'XScale', 'log', 'YScale', 'log');
    h1 = plot(ax, s, marg, '-', 'Color', P.data, 'LineWidth', 1.7);
    h2 = plot(ax, s(m05), C05 * s(m05) .^ sl05, '--', 'Color', P.fit, 'LineWidth', 1.6);
    xlabel(ax, 'step length {\its}', 'Interpreter', 'tex');
    ylabel(ax, '1 - |\lambda_{max}|', 'Interpreter', 'tex');
    xlim(ax, mpl_loglim(s));
    ylim(ax, mpl_loglim([marg; C05 * s(m05) .^ sl05]));
    legend(ax, [h1 h2], {'1 - |\lambda_{max}|', ...
           sprintf('fit for {\\its} \\leq 0.05: 1 - |\\lambda_{max}| = %.2f {\\its}^{%.3f}', C05, sl05)}, ...
           'FontSize', 8, 'Box', 'off', 'Location', 'southeast', 'Interpreter', 'tex');
    grid(ax, 'on'); ax.XMinorGrid = 'on'; ax.YMinorGrid = 'on';
    lefttitle(ax, 'recovery margin is nearly quadratic at low {\its}', 9.5);
    drawnow;
    padx = 1.08 * FS / 72 / W; pady = 1.08 * FS / 72 / H;
    ti = ax.TightInset;
    ax.Position = [ti(1) + padx, ti(2) + pady, 1 - ti(1) - ti(3) - 2 * padx, 1 - ti(2) - ti(4) - 2 * pady];
    drawnow;
    base = fullfile(FIGS, ['figure4a_scaling_jbse' suffix]);
    print(fig, [base '.png'], '-dpng', '-r300');
    print(fig, [base '.tiff'], '-dtiff', '-r300');
    print(fig, [base '.pdf'], '-dpdf', '-vector');
    close(fig);
    fprintf('wrote %s.{png,pdf,tiff}\n', base);
end

function lefttitle(ax, str, fs)
    t = title(ax, str, 'FontSize', fs, 'FontWeight', 'normal', 'Interpreter', 'tex', 'FontName', 'Helvetica');
    t.Units = 'normalized'; t.HorizontalAlignment = 'left'; t.Position(1) = 0;
end

function lim = mpl_loglim(y)
    % matplotlib's default 5 % margins on a log axis
    lo = log10(min(y(y > 0))); hi = log10(max(y));
    m = 0.05 * (hi - lo);
    lim = [10 ^ (lo - m), 10 ^ (hi + m)];
end

function p = data2fig(ax, x, y)
    % data coordinates -> normalised figure coordinates (linear axes)
    pos = ax.Position; xl = xlim(ax); yl = ylim(ax);
    p = [pos(1) + pos(3) * (x - xl(1)) / (xl(2) - xl(1)), ...
         pos(2) + pos(4) * (y - yl(1)) / (yl(2) - yl(1))];
end

function q = fig2data(ax, p)
    pos = ax.Position; xl = xlim(ax); yl = ylim(ax);
    q = [xl(1) + (p(1) - pos(1)) / pos(3) * (xl(2) - xl(1)), ...
         yl(1) + (p(2) - pos(2)) / pos(4) * (yl(2) - yl(1))];
end

function str = pyrepr(x)
    % shortest decimal string that round-trips, as Python's repr(float) writes
    for prec = 1:17
        str = sprintf('%.*g', prec, x);
        if str2double(str) == x, break; end
    end
    if ~contains(str, '.') && ~contains(str, 'e'), str = [str '.0']; end
end
