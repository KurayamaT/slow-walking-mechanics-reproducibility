%RUN_CSWEEP_EXTENDED  Push-off-coefficient robustness (port of
%   code/run_csweep_extended.py).  For c in {0.80, 1.00, 1.04, 1.20} the branch
%   with P(s) = c alpha tan(alpha) is traced by csweep_branch.m (seeded
%   continuation over s in [0.01, 0.80], anchored at s = 0.40 by continuation
%   in c from 1.04).  Branches are read from data_matlab/branches/csweep_c<c>.mat
%   and computed here if missing (several minutes each; to parallelise, call
%   csweep_branch(c) in separate MATLAB sessions first).
%   Writes data_matlab/supp_csweep.csv (same schema as data/supp_csweep.csv)
%          figures_matlab/sens_pushoff_c{,_color}.{png,tiff,pdf}
CS = [0.80, 1.00, 1.04, 1.20];
here = fileparts(mfilename('fullpath'));
datadir = fullfile(here, '..', 'data_matlab');
figdir = fullfile(here, '..', 'figures_matlab');
if ~exist(datadir, 'dir'), mkdir(datadir); end
if ~exist(figdir, 'dir'), mkdir(figdir); end
OUT = fullfile(figdir, 'sens_pushoff_c.png');

tic;
traces = cell(1, numel(CS));      % {c, v, nh}
fprintf('%5s %4s %7s %7s %8s %6s %7s %6s\n', 'c', 'n', 'v_lo', 'Nh_lo', 'v~0.016', 'Nh', 'v_hi', 'Nh_hi');
for i = 1:numel(CS)
    f = fullfile(datadir, 'branches', sprintf('csweep_c%.2f.mat', CS(i)));
    if ~exist(f, 'file'), csweep_branch(CS(i)); end
    b = load(f);
    v = b.v(:)'; lm = b.lam_max(:)';
    nh = -log(2.0) ./ log(min(max(lm, 1e-12), 0.9999999));   % np.clip(lm, 1e-12, 0.9999999)
    % b.s is the branch parameter q; alpha_h = Z(1,:) is what the gait realises.
    a_h = b.Z(1, :); s_real = 2.0 * sin(a_h); v_real = s_real ./ b.T(:)';
    traces{i} = {CS(i), v_real, nh, b.s(:)', a_h, s_real};
    v = v_real;
    [~, i_lo] = min(v); [~, i_mid] = min(abs(v - 0.016)); [~, i_hi] = max(v);
    fprintf('%5.2f %4d %7.4f %7.1f %8.4f %6.1f %7.4f %6.1f\n', CS(i), numel(v), ...
            v(i_lo), nh(i_lo), v(i_mid), nh(i_mid), v(i_hi), nh(i_hi));
end

% the traced branches, so Supplementary S4 can be replotted without recomputing
fh = fopen(fullfile(datadir, 'supp_csweep.csv'), 'w');
fprintf(fh, 'c,q,alpha_h,s,v,lam_max,N_half\n');
for i = 1:numel(CS)
    [c, v, nh, qq, a_h, s_real] = traces{i}{:};
    lm = exp(-log(2.0) ./ nh);
    for j = 1:numel(v)
        fprintf(fh, '%.2f,%.3f,%.10g,%.10g,%.10g,%.10f,%.4f\n', ...
                c, qq(j), a_h(j), s_real(j), v(j), lm(j), nh(j));
    end
end
fclose(fh);
fprintf('saved: %s\n', fullfile(datadir, 'supp_csweep.csv'));

% grayscale: {colour, linestyle, linewidth} in CS order
MONO  = {{[0 0 0], '-', 1.6}, {[0.30 0.30 0.30], '--', 1.7}, {[0 0 0], '-', 2.6}, {[0.45 0.45 0.45], ':', 1.8}};
% colorblind-safe (Okabe-Ito): c=0.8 blue, 1.0 green, 1.04 vermillion (bold, main), 1.2 purple
COLOR = {{'#0072B2', '-', 1.8}, {'#009E73', '--', 1.9}, {'#D55E00', '-', 2.6}, {'#CC79A7', ':', 2.0}};
make_figure(traces, MONO, '', OUT);
make_figure(traces, COLOR, '_color', OUT);
fprintf('done (%.1f s)\n', toc);

function make_figure(traces, palette, suffix, OUT)
    fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1 1 7.4 4.9]);
    set(fig, 'DefaultAxesFontSize', 11, 'DefaultAxesFontName', 'Helvetica', ...
             'DefaultTextFontName', 'Helvetica');
    ax = axes(fig); hold(ax, 'on');
    h = gobjects(1, numel(traces)); labels = cell(1, numel(traces)); i_main = 0;
    for i = 1:numel(traces)
        [c, v, nh] = traces{i}{:};
        [col, ls, lw] = palette{i}{:};
        lab = sprintf('{\\itc} = %.2f', c);
        if abs(c - 1.04) < 1e-9, lab = [lab '  (main)']; i_main = i; end
        h(i) = plot(ax, v, nh, 'LineStyle', ls, 'Color', col, 'LineWidth', lw);
        labels{i} = lab;
    end
    if i_main > 0, uistack(h(i_main), 'top'); end           % zorder 4 for the main branch
    hx = xline(ax, 0.041, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0);
    uistack(hx, 'bottom');
    set(ax, 'YScale', 'log', 'XLim', [0 0.235], 'YLim', [1 1100], 'Box', 'off', ...
            'TickDir', 'out', 'Layer', 'top');
    ax.XAxis.TickLabelFormat = '%.2f';
    ax.YTick = [1 10 100 1000];
    ax.YMinorTick = 'off';
    xlabel(ax, 'Dimensionless speed  {\itv}', 'Interpreter', 'tex');
    ylabel(ax, 'Perturbation half-life  {\itN}_{1/2} (steps)', 'Interpreter', 'tex');
    title(ax, ['Push-off coefficient {\itc} in {\itP}({\its}) = {\itc}{\it\alpha}tan{\it\alpha}: ' ...
               'low-speed weakening is robust'], 'Interpreter', 'tex', 'FontSize', 10.5, 'FontWeight', 'bold');
    ax.TitleHorizontalAlignment = 'left';
    lg = legend(ax, h, labels, 'Location', 'northeast', 'Box', 'off', 'FontSize', 9.5, 'Interpreter', 'tex');
    lg.ItemTokenSize = [28 18];
    % emulate tight_layout: fill the figure, small outer margin
    ax.PositionConstraint = 'outerposition';
    ax.OuterPosition = [0.005 0.005 0.99 0.99];
    drawnow;
    % annotation: text at (0.055, 250), arrow to (0.006, 200)
    text(ax, 0.055, 250, {'{\itN}_{1/2} rises steeply toward the slow end for every {\itc},', ...
                          'with no fold encountered along the branch traced'}, ...
         'FontSize', 9, 'Color', [0.3 0.3 0.3], 'HorizontalAlignment', 'left', ...
         'VerticalAlignment', 'bottom', 'Interpreter', 'tex');
    p0 = data2fig(ax, 0.055, 250); p1 = data2fig(ax, 0.006, 200);
    annotation(fig, 'arrow', [p0(1) p1(1)], [p0(2) p1(2)], 'Color', [0.55 0.55 0.55], ...
               'LineWidth', 0.9, 'HeadStyle', 'plain', 'HeadLength', 6, 'HeadWidth', 6);
    % export at the figure size (7.4 x 4.9 in): png/tiff at 300 dpi, pdf vector
    set(fig, 'PaperUnits', 'inches', 'PaperSize', [7.4 4.9], 'PaperPosition', [0 0 7.4 4.9], ...
             'InvertHardcopy', 'off');
    out = strrep(OUT, '.png', [suffix '.png']);
    print(fig, out, '-dpng', '-r300');
    print(fig, strrep(out, '.png', '.tiff'), '-dtiff', '-r300');
    print(fig, strrep(out, '.png', '.pdf'), '-dpdf', '-painters');
    close(fig);
    fprintf('saved: %s\n', out);
end

function p = data2fig(ax, x, y)
    % data coordinates -> normalized figure coordinates (log y axis)
    pos = ax.Position; xl = ax.XLim; yl = ax.YLim;
    fx = (x - xl(1)) / (xl(2) - xl(1));
    fy = (log10(y) - log10(yl(1))) / (log10(yl(2)) - log10(yl(1)));
    p = [pos(1) + fx * pos(3), pos(2) + fy * pos(4)];
end
