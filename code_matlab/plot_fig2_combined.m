%PLOT_FIG2_COMBINED  Figure 2 (combined a,b) from the EXTENDED family.
%   MATLAB port of code/plot_fig2_combined.py.
%     (a) |lambda_max| (solid, left) and N_1/2 on a LOG axis (dashed, right) vs v,
%         read from data_matlab/master_fixed_khip_extended.csv.
%     (b) nonlinear return-map recovery at a low-speed gait (real, slow
%         monotonic) and a plateau gait (complex, oscillatory decay), read
%         from data_matlab/fig2b_traces.csv (run_fig2b_traces.m).
%   Renders two coexisting versions with the same basenames as the Python figures:
%     figures_matlab/figure2_combined.{png,pdf,tiff}        grayscale
%     figures_matlab/figure2_combined_color.{png,pdf,tiff}  Okabe-Ito (blue / vermillion)
here = fileparts(mfilename('fullpath'));
ROOT = fullfile(here, '..');
EXT = fullfile(ROOT, 'data_matlab', 'master_fixed_khip_extended.csv');
TRACES = fullfile(ROOT, 'data_matlab', 'fig2b_traces.csv');
FIGS = fullfile(ROOT, 'figures_matlab');
if ~exist(FIGS, 'dir'), mkdir(FIGS); end

g = @(v) [v v v];                                      % matplotlib grey string -> RGB
main(struct('lam', [0.067 0.067 0.067], 'half', [0.478 0.478 0.478], 'low', [0 0 0], 'plat', g(0.45)), '', EXT, TRACES, FIGS);
main(struct('lam', hex('#0072B2'), 'half', hex('#D55E00'), 'low', hex('#0072B2'), 'plat', hex('#D55E00')), '_color', EXT, TRACES, FIGS);

% ------------------------------------------------------------------------
function main(P, suffix, EXT, TRACES, FIGS)
    fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [1 1 12.4 4.8], ...
                 'Color', 'w', 'InvertHardcopy', 'off');
    % axes rectangles reproduce matplotlib's tight_layout(w_pad=4.0) result
    axA = axes(fig, 'Position', [0.0590 0.1320 0.3862 0.7935]);
    axB = axes(fig, 'Position', [0.5935 0.1320 0.3850 0.7935]);
    panel_a(axA, P, EXT);
    panel_b(axB, P, TRACES, EXT);
    base = fullfile(FIGS, ['figure2_combined' suffix]);
    set(fig, 'PaperUnits', 'inches', 'PaperSize', [12.4 4.8], 'PaperPositionMode', 'manual', ...
             'PaperPosition', [0 0 12.4 4.8]);
    print(fig, [base '.png'], '-dpng', '-r300');
    print(fig, [base '.tiff'], '-dtiff', '-r300');
    print(fig, [base '.pdf'], '-dpdf', '-vector');
    close(fig);
    fprintf('saved: %s.{png,pdf,tiff}\n', base);
end

function panel_a(ax1, P, EXT)
    d = readtable(EXT, 'TextType', 'string');
    keep = (d.v >= min(d.v) - 1e-12) & (d.v <= max(d.v) + 1e-12);   % keyed off the data: a hard
% 0.0029 lower clip silently dropped the realised slowest gait (v = 0.002860)
    d = d(keep, :);
    d = sortrows(d, 'v');
    v = d.v; lam = d.lam_max;
    half = -log(2.0) ./ log(min(max(lam, 1e-12), 0.9999999));
    ts = readtable(fullfile(fileparts(EXT), 'transition_spectrum.csv'), 'TextType', 'string');
    i_first_cx = find(abs(ts.l1_im) > 1e-10, 1);
    assert(~isempty(i_first_cx) && i_first_cx > 1, 'Transition bracket not found.');
    vstar = 0.5 * (ts.v(i_first_cx - 1) + ts.v(i_first_cx));

    style_axes(ax1);
    yyaxis(ax1, 'left');
    hold(ax1, 'on');
    plot(ax1, v, lam, '-', 'Color', P.lam, 'LineWidth', 2.2);
    xlabel(ax1, 'Dimensionless speed  {\itv}');
    ylabel(ax1, 'Spectral radius  |\lambda_{max}|', 'Color', P.lam);
    ylim(ax1, [0.5 1.02]); xlim(ax1, [0 0.235]);
    xline(ax1, vstar, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0);
    ax1.YAxis(1).Color = P.lam;
    xticks(ax1, 0:0.05:0.20); xtickformat(ax1, '%.2f');
    yticks(ax1, 0.5:0.1:1.0); ytickformat(ax1, '%.1f');

    yyaxis(ax1, 'right');
    plot(ax1, v, half, '--', 'Color', P.half, 'LineWidth', 2.0);
    set(ax1, 'YScale', 'log');
    ylim(ax1, [1 1100]);
    ylabel(ax1, 'Perturbation half-life  {\itN}_{1/2} (steps, log)', 'Color', P.half);
    ax1.YAxis(2).Color = P.half;
    yticks(ax1, [1 10 100 1000]);
    yyaxis(ax1, 'left');

    text(ax1, 0.045, 0.90, {'|\lambda_{max}| \rightarrow 1, {\itN}_{1/2} grows steeply', ...
         'toward the slow end', '(no fold encountered along the branch traced)'}, ...
         'FontSize', 8, 'Color', g3(0.25), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
    arrow_norm(ax1, [0.044 0.952], [0.006 0.985], 'plain', 0.8, g3(0.5));
    text(ax1, 0.15, 0.74, 'stable-recovery plateau', 'FontSize', 8, 'HorizontalAlignment', 'center', ...
         'Color', g3(0.3));
    left_title(ax1, '(a)  Recovery weakens toward the slow end of the computed branch');
    box(ax1, 'off');
end

function panel_b(axB, P, TRACES, EXT)
    cLow = P.low; cPlat = P.plat;
    d = readtable(TRACES, 'TextType', 'string');
    w = sortrows(d(d.gait == "low speed", :), 'n');
    p = sortrows(d(d.gait == "plateau", :), 'n');
    lam_w = w.envelope(2); lam_p = p.envelope(2);          % envelope(n=1) = |lambda_max|
    branch = readtable(EXT, 'TextType', 'string');
    v_w = trace_speed(w, branch, lam_w);
    v_p = trace_speed(p, branch, lam_p);
    nhw = log(2) / -log(lam_w);
    nhp = log(2) / -log(lam_p);
    N = max(w.n);

    style_axes(axB);
    hold(axB, 'on');
    yline(axB, 0.0, '-', 'Color', g3(0.85), 'LineWidth', 0.8);
    yline(axB, 0.5, '-', 'Color', g3(0.8), 'LineWidth', 0.9);

    nf = linspace(0, N, 600);
    env_p = lam_p .^ nf;
    plot(axB, nf, env_p, ':', 'Color', g3(0.6), 'LineWidth', 1.0);
    plot(axB, nf, -env_p, ':', 'Color', g3(0.6), 'LineWidth', 1.0);

    hP = plot(axB, p.n, p.dev_signed, '--s', 'Color', cPlat, 'MarkerSize', 3.8, 'LineWidth', 1.6, ...
              'MarkerFaceColor', cPlat, 'MarkerEdgeColor', cPlat);
    hW = plot(axB, w.n, w.dev_signed, '-o', 'Color', cLow, 'MarkerSize', 4.2, 'LineWidth', 1.8, ...
              'MarkerFaceColor', cLow, 'MarkerEdgeColor', cLow);
    uistack(hW, 'top');

    plot(axB, [nhw nhw], [0 0.5], ':', 'Color', cLow, 'LineWidth', 1.2);
    plot(axB, [nhp nhp], [0 lam_p ^ nhp], ':', 'Color', cPlat, 'LineWidth', 1.2);
    text(axB, nhw + 2.0, 0.70, sprintf('{\\itN}_{1/2} \\approx %.0f steps', nhw), 'FontSize', 9.5, ...
         'Color', cLow, 'VerticalAlignment', 'baseline');
    arrow_norm(axB, [nhw + 1.9, 0.665], [nhw, 0.5], 'vback2', 0.9, cLow);
    text(axB, nhp + 2.6, 0.20, sprintf('envelope {\\itN}_{1/2} \\approx %.0f', nhp), 'FontSize', 9.5, ...
         'Color', cPlat, 'VerticalAlignment', 'baseline');
    arrow_norm(axB, [nhp + 2.45, 0.245], [nhp, lam_p ^ nhp], 'vback2', 0.9, cPlat);
    text(axB, N - 0.3, 0.52, 'half of initial envelope', 'HorizontalAlignment', 'right', ...
         'VerticalAlignment', 'bottom', 'FontSize', 8, 'Color', g3(0.5));
    text(axB, 0.30 * N, -0.30, 'oscillatory decay on the plateau', 'FontSize', 8.5, ...
         'Color', g3(0.35), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontAngle', 'italic');
    xlabel(axB, 'Step number after perturbation  {\itn}');
    ylabel(axB, 'Normalized stance-angle deviation');
    xlim(axB, [0 N]); ylim(axB, [-0.5 1.05]);
    xticks(axB, 0:5:N); yticks(axB, -0.4:0.2:1.0); ytickformat(axB, '%.1f');
    left_title(axB, '(b)  Perturbation recovery is much slower at low speed');
    lg = legend(axB, [hW hP], ...
        {sprintf('low speed  {\\itv} = %.3f   (|\\lambda| = %.2f, real)', v_w, lam_w), ...
         sprintf('plateau  {\\itv} = %.3f   (|\\lambda| = %.2f, complex)', v_p, lam_p)}, ...
        'FontSize', 8.8, 'Box', 'off', 'Location', 'northeast');
    lg.ItemTokenSize = [22 18];
    box(axB, 'off');
end

function v = trace_speed(trace, branch, lam)
    q = unique(trace.s);
    assert(isscalar(q), 'Trace must contain one branch input q.');
    gait = branch(abs(branch.q - q) < 1e-10, :);
    assert(height(gait) == 1, 'Trace branch input q must match one stored gait.');
    assert(abs(gait.lam_max - lam) < 1e-6, 'Trace and branch multipliers disagree.');
    v = gait.v;
end

% ------------------------------------------------------------------------
function style_axes(ax)
    set(ax, 'FontSize', 11, 'LabelFontSizeMultiplier', 1, 'TitleFontSizeMultiplier', 1, ...
            'TickDir', 'out', 'TickLength', [0.008 0.008], 'LineWidth', 0.8, ...
            'XColor', 'k', 'YColor', 'k', 'Layer', 'top', 'PositionConstraint', 'innerposition');
end

function left_title(ax, str)
    t = title(ax, str, 'FontWeight', 'bold', 'FontSize', 11, 'Interpreter', 'tex');
    t.Units = 'normalized';
    t.HorizontalAlignment = 'left';
    t.Position(1) = 0;
end

function arrow_norm(ax, p0, p1, head, lw, col)
    % Arrow from data point p0 to p1 (left-axis data coordinates), drawn as a
    % figure annotation; 'plain' = open head (matplotlib '->'),
    % 'vback2' = filled head (matplotlib '-|>').
    pos = ax.Position; xl = ax.XLim; yl = ax.YLim;
    f = @(p) [pos(1) + (p(1) - xl(1)) / (xl(2) - xl(1)) * pos(3), ...
              pos(2) + (p(2) - yl(1)) / (yl(2) - yl(1)) * pos(4)];
    a = f(p0); b = f(p1);
    if strcmp(head, 'plain'), hl = 5; hw = 5; else, hl = 7; hw = 6; end
    annotation(ax.Parent, 'arrow', [a(1) b(1)], [a(2) b(2)], 'Color', col, 'LineWidth', lw, ...
               'HeadStyle', head, 'HeadLength', hl, 'HeadWidth', hw);
end

function c = g3(v), c = [v v v]; end
function c = hex(h), c = sscanf(h(2:end), '%2x%2x%2x').' / 255; end
