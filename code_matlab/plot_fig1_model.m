%PLOT_FIG1_MODEL  Figure 1: the actively powered simplest walking model.
%   MATLAB port of code/plot_fig1_model.py (a pure schematic, no computation).
%     main panel  a representative swing-phase configuration: continuous dynamics,
%                 theta, phi, the hip spring, the hip mass, gravity, walking direction.
%     inset       the heel-strike instant: both feet on the ground, the impulsive
%                 push-off P along the trailing leg, the half inter-leg angle alpha,
%                 and the reset itself.
%   Grayscale only (no _color variant, as in the Python original).
%   Writes figures_matlab/figure1_model.{png,pdf,tiff}
here = fileparts(mfilename('fullpath'));
FIGS = fullfile(here, '..', 'figures_matlab');
if ~exist(FIGS, 'dir'), mkdir(FIGS); end

BLACK = [0.067 0.067 0.067]; GREY = [0.333 0.333 0.333]; LIGHT = [0.6 0.6 0.6];

fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [1 1 7.6 4.5], ...
             'Color', 'w', 'InvertHardcopy', 'off');
ax = axes(fig, 'Position', [0.02 0.04 0.64 0.92]);
hold(ax, 'on');
PT_MAIN = 157.8;                      % points per data unit in the main panel (4.864 in / 2.22 units)

% ---------------- main panel: a swing-phase configuration ----------------
F1 = [0.00, 0.00];                    % stance foot, on the ground
TH = 0.165;                           % stance-leg angle from the vertical (hip ahead)
H = F1 + [sin(TH), cos(TH)];
PHI = 0.60;                           % inter-leg angle at the hip
TH_SW = TH + PHI;                     % swing leg is AHEAD of the stance leg
F2 = H + [sin(TH_SW), -cos(TH_SW)];

ground(ax, -0.75, 1.35, 0.0, 26, 0.045, 1.0, BLACK, GREY);
plot(ax, [H(1), H(1)], [H(2) + 0.13, -0.02], ':', 'Color', LIGHT, 'LineWidth', 1.1);

plot(ax, [F1(1), H(1)], [F1(2), H(2)], '-', 'Color', BLACK, 'LineWidth', 3.0);      % stance leg
plot(ax, [H(1), F2(1)], [H(2), F2(2)], '--', 'Color', GREY, 'LineWidth', 2.6);       % swing leg
plot(ax, F1(1), F1(2), 'o', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', BLACK, 'LineWidth', 1.8, 'MarkerSize', 9);
plot(ax, F2(1), F2(2), 'o', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', GREY, 'LineWidth', 1.8, 'MarkerSize', 9);
plot(ax, H(1), H(2), 'o', 'MarkerFaceColor', BLACK, 'MarkerEdgeColor', BLACK, 'MarkerSize', 16);
text(ax, H(1) - 0.30, H(2) + 0.02, '{\itM}', 'FontSize', 14, 'VerticalAlignment', 'baseline');

% hip spring, between the two legs just below the hip
p_st = H + (F1 - H) * 0.26; p_sw = H + (F2 - H) * 0.26;
spring(ax, p_st, p_sw, 5, 0.028, 1.3, BLACK);
mid = 0.5 * (p_st + p_sw);
lab = [mid(1) + 0.34, mid(2) + 0.30];
text(ax, lab(1), lab(2), '{\itk}_{hip}', 'FontSize', 13, 'Color', BLACK, 'VerticalAlignment', 'baseline');
plot(ax, [lab(1) - 0.005, mid(1)], [lab(2) - 0.03, mid(2)], '-', 'Color', GREY, 'LineWidth', 0.9);

a_st = mod(atan2d(F1(2) - H(2), F1(1) - H(1)), 360);     % hip -> stance foot
a_sw = mod(atan2d(F2(2) - H(2), F2(1) - H(1)), 360);     % hip -> swing foot
arc(ax, H, 0.43, a_st, 270, BLACK, 1.1);
text(ax, H(1) - 0.20, H(2) - 0.48, '{\it\theta}', 'FontSize', 14, 'VerticalAlignment', 'baseline');
arc(ax, H, 0.65, a_st, a_sw, GREY, 1.1);
text(ax, H(1) + 0.30, H(2) - 0.70, '$\varphi$', 'FontSize', 14, 'Color', GREY, 'Interpreter', 'latex', ...
     'VerticalAlignment', 'baseline');

arrow(ax, [-0.55, 0.86], [-0.55, 0.32], 1.8, BLACK, PT_MAIN);
text(ax, -0.70, 0.55, '{\itg}', 'FontSize', 14, 'VerticalAlignment', 'baseline');
arrow(ax, [0.84, -0.26], [1.28, -0.26], 1.4, GREY, PT_MAIN);
text(ax, 0.80, -0.40, 'walking direction', 'FontSize', 9.5, 'Color', GREY, 'FontAngle', 'italic', ...
     'VerticalAlignment', 'baseline');

xlim(ax, [-0.80, 1.42]); ylim(ax, [-0.50, 1.30]);
daspect(ax, [1 1 1]); axis(ax, 'off');
text(ax, -0.80 + 0.02 * 2.22, 1.30 + 0.038, 'Swing phase: continuous dynamics', 'FontSize', 11, ...
     'VerticalAlignment', 'bottom');

% ---------------- inset: the heel-strike instant ----------------
ax2 = axes(fig, 'Position', [0.655 0.20 0.335 0.72]);
hold(ax2, 'on');
PT_INSET = 91.7;                      % points per data unit in the inset (2.546 in / 2 units)
rectangle(ax2, 'Position', [-1.0, -0.72, 2.0, 1.92], 'EdgeColor', LIGHT, 'LineWidth', 1.0, 'Clipping', 'off');

A = 0.42;                             % half inter-leg angle, drawn large
Hi = [0.0, cos(A)];
Ft = Hi + [-sin(A), -cos(A)];         % trailing foot
Fl = Hi + [+sin(A), -cos(A)];         % leading foot
ground(ax2, -0.95, 0.95, 0.0, 16, 0.05, 1.0, BLACK, GREY);
plot(ax2, [Ft(1), Hi(1)], [Ft(2), Hi(2)], '-', 'Color', BLACK, 'LineWidth', 2.6);
plot(ax2, [Hi(1), Fl(1)], [Hi(2), Fl(2)], '-', 'Color', GREY, 'LineWidth', 2.6);
plot(ax2, Ft(1), Ft(2), 'o', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', BLACK, 'LineWidth', 1.6, 'MarkerSize', 7);
plot(ax2, Fl(1), Fl(2), 'o', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', GREY, 'LineWidth', 1.6, 'MarkerSize', 7);
plot(ax2, Hi(1), Hi(2), 'o', 'MarkerFaceColor', BLACK, 'MarkerEdgeColor', BLACK, 'MarkerSize', 11);
plot(ax2, [Hi(1), Hi(1)], [Hi(2) + 0.10, 0], ':', 'Color', LIGHT, 'LineWidth', 1.0);

arc(ax2, Hi, 0.25, 270 - rad2deg(A), 270 + rad2deg(A), BLACK, 1.0);
text(ax2, Hi(1) - 0.055, Hi(2) - 0.40, '2{\it\alpha}', 'FontSize', 12, 'VerticalAlignment', 'baseline');

u = (Hi - Ft) / hypot(Hi(1) - Ft(1), Hi(2) - Ft(2));
n = [-u(2), u(1)];
arrow(ax2, Ft + u * 0.04, Ft + u * 0.42, 2.2, BLACK, PT_INSET);
labP = Ft + u * 0.24 - n * 0.20;
text(ax2, labP(1), labP(2), '{\itP}', 'FontSize', 13, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

text(ax2, 0.0, -0.42, '$\dot{\theta}^{+}=\dot{\theta}^{-}\cos 2\alpha-P\sin 2\alpha$', 'FontSize', 10.5, ...
     'HorizontalAlignment', 'center', 'Interpreter', 'latex', 'VerticalAlignment', 'baseline');
text(ax2, 0.0, -0.60, '$s=2\sin\alpha$', 'FontSize', 10, 'HorizontalAlignment', 'center', 'Color', GREY, ...
     'Interpreter', 'latex', 'VerticalAlignment', 'baseline');
xlim(ax2, [-1.0, 1.0]); ylim(ax2, [-0.72, 1.20]);
daspect(ax2, [1 1 1]); axis(ax2, 'off');
text(ax2, -1.0 + 0.03 * 2.0, 1.20 + 0.065, 'Heel strike: impulsive reset', 'FontSize', 10, ...
     'VerticalAlignment', 'bottom');

base = fullfile(FIGS, 'figure1_model');
set(fig, 'PaperUnits', 'inches', 'PaperSize', [7.6 4.5], 'PaperPositionMode', 'manual', ...
         'PaperPosition', [0 0 7.6 4.5]);
print(fig, [base '.png'], '-dpng', '-r300');
print(fig, [base '.tiff'], '-dtiff', '-r300');
print(fig, [base '.pdf'], '-dpdf', '-vector');
close(fig);
fprintf('wrote %s.{png,pdf,tiff}\n', base);

% ------------------------------------------------------------------------
function spring(ax, p0, p1, coils, amp, lw, color)
    % Zig-zag between two points, for the hip spring.
    d = p1 - p0;
    L = hypot(d(1), d(2));
    u = d / L;
    n = [-u(2), u(1)];
    ts = linspace(0, 1, 4 * coils + 1);
    pts = zeros(numel(ts), 2);
    for i = 1:numel(ts)
        i0 = i - 1;
        if i0 == 0 || i0 == numel(ts) - 1
            off = 0;
        elseif mod(i0, 2) == 1
            off = 1;
        else
            off = -1;
        end
        pts(i, :) = p0 + ts(i) * d + n * amp * off;
    end
    plot(ax, pts(:, 1), pts(:, 2), '-', 'Color', color, 'LineWidth', lw, 'LineJoin', 'miter');
end

function ground(ax, x0, x1, y, n, h, lw, BLACK, GREY)
    plot(ax, [x0, x1], [y, y], '-', 'Color', BLACK, 'LineWidth', 1.6);
    for x = linspace(x0, x1 - (x1 - x0) / n, n)
        plot(ax, [x, x - h], [y, y - h], '-', 'Color', GREY, 'LineWidth', lw);
    end
end

function arc(ax, centre, r, th1, th2, color, lw)
    % Counter-clockwise arc from th1 to th2 (degrees), as matplotlib.patches.Arc.
    if th2 < th1, th2 = th2 + 360; end
    t = linspace(th1, th2, 200);
    plot(ax, centre(1) + r * cosd(t), centre(2) + r * sind(t), '-', 'Color', color, 'LineWidth', lw);
end

function arrow(ax, p0, p1, lw, color, pt_per_unit)
    % Filled-head arrow (matplotlib '-|>') drawn in data units; the axes have
    % equal data aspect, so a triangle in data coordinates is undistorted.
    % Head size follows matplotlib: length 0.4, half-width 0.2 of mutation_scale
    % (10 pt), inflated by the line width.
    hl = (4 + 1.2 * lw) / pt_per_unit; hw = (2 + 0.8 * lw) / pt_per_unit;
    d = p1 - p0; u = d / hypot(d(1), d(2)); n = [-u(2), u(1)];
    tip = p1; base = p1 - u * hl;
    plot(ax, [p0(1), base(1)], [p0(2), base(2)], '-', 'Color', color, 'LineWidth', lw);
    patch(ax, [tip(1), base(1) + n(1) * hw, base(1) - n(1) * hw], ...
              [tip(2), base(2) + n(2) * hw, base(2) - n(2) * hw], color, 'EdgeColor', color, 'LineWidth', 0.5);
end
