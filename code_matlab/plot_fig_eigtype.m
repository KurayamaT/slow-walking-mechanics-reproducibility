%PLOT_FIG_EIGTYPE  Figure 3: the dominant non-zero Floquet multiplier changes
%   TYPE at the transition speed v* (two positive real multipliers below v*, a
%   complex-conjugate pair above). MATLAB port of code/plot_fig_eigtype.py.
%
%   Renders two coexisting versions:
%     figures_matlab/fig_eigtype.{png,pdf,tiff}        grayscale
%     figures_matlab/fig_eigtype_color.{png,pdf,tiff}  Okabe-Ito: blue Re,
%                                                      orange +/-Im, black |lambda_max|
%   Reads: data_matlab/master_fixed_khip_extended.csv (the MATLAB-generated branch).
here = fileparts(mfilename('fullpath'));
csvfile = fullfile(here, '..', 'data_matlab', 'master_fixed_khip_extended.csv');
figdir = fullfile(here, '..', 'figures_matlab');
if ~exist(figdir, 'dir'), mkdir(figdir); end
OUT = fullfile(figdir, 'fig_eigtype.png');

d = readtable(csvfile, 'TextType', 'string');
keep = (d.v >= min(d.v) - 1e-12) & (d.v <= max(d.v) + 1e-12);   % see plot_fig2_combined.m
d = d(keep, :);
d = sortrows(d, 'v');
v = d.v;
l1r = d.lam1_re; l1i = d.lam1_im;
l2r = d.lam2_re; l2i = d.lam2_im;
lammax = d.lam_max;

is_complex = (l1i ~= 0) | (l2i ~= 0);
% v* comes from the DEDICATED grid of section 3.3 (spacing 4e-4), not from this
% 0.001 branch: the coarse branch brackets the coalescence three grid points
% late and would print 0.0406 where the paper reports 0.0403.
tsf = fullfile(here, '..', 'data_matlab', 'transition_spectrum.csv');
ts = readtable(tsf, 'TextType', 'string');
ts_cx = abs(ts.l1_im) > 1e-10;
i_first_cx = find(ts_cx, 1);
if isempty(i_first_cx)
    vstar = NaN;
else
    vstar = 0.5 * (ts.v(i_first_cx - 1) + ts.v(i_first_cx));   % bracket midpoint
end
fprintf('v* = %.6g   (dedicated grid, bracket [%.6f, %.6f])\n', ...
        vstar, ts.v(i_first_cx - 1), ts.v(i_first_cx));

make(v, l1r, l1i, l2r, l2i, lammax, vstar, [0.45 0.45 0.45], [0.62 0.62 0.62], [0.067 0.067 0.067], '', OUT);   % grayscale
make(v, l1r, l1i, l2r, l2i, lammax, vstar, [0 114 178]/255, [230 159 0]/255, [0 0 0], '_color', OUT);          % Okabe-Ito

function make(v, l1r, l1i, l2r, l2i, lammax, vstar, cRe, cIm, cMax, suffix, OUT)
    FS = 11;                                   % matplotlib font.size
    W = 7.4; H = 5.3;                          % figsize (inches)
    fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [1 1 W H], 'Color', 'w');
    fig.PaperUnits = 'inches'; fig.PaperSize = [W H]; fig.PaperPosition = [0 0 W H];
    ax = axes(fig, 'Units', 'normalized', 'FontName', 'Helvetica', 'FontSize', FS, ...
              'LabelFontSizeMultiplier', 1, 'TitleFontSizeMultiplier', 1, ...
              'TickDir', 'out', 'TickLength', [0.0035 0.0035], 'Box', 'off', ...
              'TickLabelInterpreter', 'tex', 'Layer', 'bottom');
    hold(ax, 'on');
    % zorder 0/1: zero line and the transition speed
    plot(ax, [0 0.235], [0 0], '-', 'Color', [0.85 0.85 0.85], 'LineWidth', 0.8, 'HandleVisibility', 'off');
    plot(ax, [vstar vstar], [-0.90 1.32], ':', 'Color', [0.55 0.55 0.55], 'LineWidth', 1.2, 'HandleVisibility', 'off');
    % zorder 2: imaginary parts +/- (dashed; zero below v*)
    hIm = plot(ax, v, l1i, '--', 'Color', cIm, 'LineWidth', 1.5);
    plot(ax, v, l2i, '--', 'Color', cIm, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    % zorder 3: real parts of the two non-zero multipliers
    hRe = plot(ax, v, l1r, '-', 'Color', cRe, 'LineWidth', 1.6);
    plot(ax, v, l2r, '-', 'Color', cRe, 'LineWidth', 1.6, 'HandleVisibility', 'off');
    % zorder 5: dominant magnitude (bold)
    hMax = plot(ax, v, lammax, '-', 'Color', cMax, 'LineWidth', 2.4);

    xlabel(ax, 'Dimensionless speed  {\itv}', 'FontSize', FS, 'Interpreter', 'tex');
    ylabel(ax, 'Floquet multiplier', 'FontSize', FS, 'Interpreter', 'tex');
    xlim(ax, [0.0 0.235]); ylim(ax, [-0.90 1.32]);
    xticks(ax, 0:0.05:0.20); xtickformat(ax, '%.2f');
    yticks(ax, -0.75:0.25:1.0); ytickformat(ax, '%.2f');
    lg = legend(ax, [hRe hIm hMax], {'Re\lambda_{1,2}', '\pm Im\lambda', '|\lambda_{max}|'}, ...
                'FontSize', 8.8, 'Box', 'off', 'Location', 'east', 'Interpreter', 'tex');
    lg.ItemTokenSize = [24 18];

    % tight_layout(): pad = 1.08 * font.size points on every side
    drawnow;
    ti = ax.TightInset;
    padx = 1.08 * FS / 72 / W; pady = 1.08 * FS / 72 / H;
    ax.Position = [ti(1) + padx, ti(2) + pady, 1 - ti(1) - ti(3) - 2 * padx, 1 - ti(2) - 2 * pady];  % no title: ti(4) is its reserved space
    drawnow;

    % the two centred multi-line labels (matplotlib centres every line; MATLAB
    % left-aligns the lines of one block, so each line is placed separately
    % with matplotlib's 1.2 x font-size line spacing)
    dy = 1.2 * 8.3 / 72 / (H * ax.Position(4)) * (1.32 + 0.90);     % line spacing in data units
    lines_lo = {'below {\itv}^*:', 'two positive real', '(monotonic decay)'};
    lines_hi = {'above {\itv}^*:', 'complex-conjugate pair', '(oscillatory decay)'};
    for k = 1:3
        text(ax, vstar * 0.5, 1.27 - (k - 1) * dy, lines_lo{k}, ...
             'FontSize', 8.3, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
             'Color', [0.3 0.3 0.3], 'Interpreter', 'tex', 'FontName', 'Helvetica');
        text(ax, (vstar + 0.231) / 2.0, 1.27 - (k - 1) * dy, lines_hi{k}, ...
             'FontSize', 8.3, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
             'Color', [0.3 0.3 0.3], 'Interpreter', 'tex', 'FontName', 'Helvetica');
    end

    txt = sprintf('transition speed {\\itv}^* \\approx %.4f', vstar);
    tx = 0.060; ty = -0.83;
    text(ax, tx, ty, txt, 'FontSize', 9, 'Color', [0.25 0.25 0.25], 'Interpreter', 'tex', ...
         'HorizontalAlignment', 'left', 'VerticalAlignment', 'baseline', 'FontName', 'Helvetica');
    arrow_start = data2fig(ax, tx - 0.003, ty + 0.018);
    arrow_end = data2fig(ax, vstar, -0.77);
    annotation(fig, 'arrow', [arrow_start(1) arrow_end(1)], [arrow_start(2) arrow_end(2)], ...
               'Color', [0.5 0.5 0.5], 'LineWidth', 0.9, 'HeadStyle', 'vback2', ...
               'HeadLength', 5, 'HeadWidth', 5);

    out = strrep(OUT, '.png', [suffix '.png']);
    print(fig, out, '-dpng', '-r300');
    print(fig, strrep(out, '.png', '.tiff'), '-dtiff', '-r300');
    print(fig, strrep(out, '.png', '.pdf'), '-dpdf', '-vector');
    close(fig);
    fprintf('saved: %s\n', out);
end

function p = data2fig(ax, x, y)
    % data coordinates -> normalised figure coordinates (linear axes)
    pos = ax.Position; xl = xlim(ax); yl = ylim(ax);
    p = [pos(1) + pos(3) * (x - xl(1)) / (xl(2) - xl(1)), ...
         pos(2) + pos(4) * (y - yl(1)) / (yl(2) - yl(1))];
end
