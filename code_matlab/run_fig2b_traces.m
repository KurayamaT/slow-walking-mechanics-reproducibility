%RUN_FIG2B_TRACES  Nonlinear return-map recovery traces for Figure 2b.
%   MATLAB port of code/run_fig2b_traces.py. Two gaits, matching Results 4.3
%   and Table 1:
%       s = 0.080   low speed, dominant positive real multiplier   (40 steps)
%       s = 0.402   plateau,   dominant complex-conjugate pair     (28 steps)
%   The displacement is applied along the dominant right eigenvector (its real
%   part at the complex-pair gait, after the phase has been pinned so that the
%   stance-angle component is real and positive), scaled so the stance-angle
%   component is exactly one; the amplitude is 0.3% of |theta*|. k_hip and P(s)
%   are held at their periodic-gait values (open-loop recovery). Recorded is the
%   signed stance-angle deviation normalised by its own initial value.
%
%   Writes data_matlab/fig2b_traces.csv with the same schema and number
%   formats as data/fig2b_traces.csv, and prints the consistency checks.
C = 1.04;
AMP = 0.003;                           % 0.3% of |theta*|
AMPS_CHECK = [0.001, 0.003, 0.010];    % shape must not depend on which of these is used
GAITS = {0.402, 28, 'plateau'; 0.080, 40, 'low speed'};
c = wk.constants();

here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'data_matlab');
if ~exist(outdir, 'dir'), mkdir(outdir); end
out = fullfile(outdir, 'fig2b_traces.csv');

tic;
rows = {}; summary = {};
for ig = 1:size(GAITS, 1)
    s = GAITS{ig, 1}; nsteps = GAITS{ig, 2}; label = GAITS{ig, 3};
    [z_fp, T, P, w, V] = solve(s, C, c);
    d = dominant_direction(V);
    lam_max = abs(w(1));
    v = s / T;
    N_half = -log(2) / log(lam_max);
    tr = signed_trace(z_fp, P, d, AMP, nsteps, c);

    % per-step ratio over the first few steps, before the complex mode changes sign
    ratios = abs(tr(2:5) ./ tr(1:4));
    fitted = median(ratios);

    % shape must not depend on amplitude over the range 3.3 reports
    max_dev = 0;
    for a = AMPS_CHECK
        tra = signed_trace(z_fp, P, d, a, nsteps, c);
        max_dev = max(max_dev, max(abs(tra - tr), [], 'omitnan'));
    end

    % value at n = N_1/2 (real-dominant gaits should be at one half)
    nh = round(N_half);
    if nh + 1 <= numel(tr), at_half = tr(nh + 1); else, at_half = NaN; end

    summary(end+1, :) = {label, s, v, lam_max, N_half, fitted, max_dev, at_half, w}; %#ok<SAGROW>
    for n = 0:numel(tr) - 1
        rows(end+1, :) = {label, s, n, tr(n + 1), lam_max ^ n}; %#ok<SAGROW>
    end
end

fh = fopen(out, 'w');
fprintf(fh, 'gait,s,n,dev_signed,envelope\n');
for i = 1:size(rows, 1)
    fprintf(fh, '%s,%.3f,%d,%.10e,%.10e\n', rows{i, 1}, rows{i, 2}, rows{i, 3}, rows{i, 4}, rows{i, 5});
end
fclose(fh);

fprintf('wrote data_matlab/fig2b_traces.csv   amplitude = %.1f%% of |theta*|, as a stance-angle displacement along the dominant direction  (%.1f s)\n', 100 * AMP, toc);
for i = 1:size(summary, 1)
    [label, s, v, lam_max, N_half, fitted, max_dev, at_half, w] = summary{i, :};
    fprintf('  %-10s s=%.3f v=%.4f |lam_max|=%.6f N_1/2=%.2f\n', label, s, v, lam_max, N_half);
    fprintf('             per-step ratio %.6f vs |lam_max| %.6f (err %.1e); dev at n=N_1/2 = %+.4f; shape spread over 0.1-1.0%% = %.2e\n', ...
            fitted, lam_max, abs(fitted - lam_max), at_half, max_dev);
    fprintf('             lam = %s\n', strjoin(arrayfun(@(L) sprintf('%.6f%+.6fi', real(L), imag(L)), w.', 'UniformOutput', false), ', '));
end

function [z_fp, T, P, w, V] = solve(s, C, c)
    % Fixed point, period, push-off and spectrum at step length s. The geometric
    % guess converges for s >= 0.054 (3.2), which covers both gaits used here.
    alpha = asin(0.5 * s);
    P = C * alpha * tan(alpha);
    z0 = [alpha; -C * alpha; (1 - cos(2 * alpha)) * (-C * alpha)];
    [z_fp, T, ok] = wk.find_fixed_point(z0, c.GAM, c.KHIP, P, c.RTOL, c.ATOL, [], [], 1e-7);
    if ~ok, error('fixed point failed at s=%.3f', s); end
    J = wk.jacobian_3d(z_fp, c.GAM, c.KHIP, P, c.RTOL, c.ATOL, c.DELTA);
    [V, D] = eig(J);
    w = diag(D);
    [~, o] = sort(-abs(w));
    w = w(o); V = V(:, o);
end

function d = dominant_direction(V)
    % Real perturbation direction from the dominant right eigenvector: the
    % phase is pinned so the stance-angle component is real and positive, the
    % real part is taken, and the vector is scaled so that d(1) = 1 exactly.
    % (For a complex pair both members give the same real direction after the
    % phase pin, so the tie order of the pair does not matter.)
    r = V(:, 1);
    r = r * exp(-1i * angle(r(1)));
    d = real(r);
    d = d / d(1);
end

function tr = signed_trace(z_fp, P, d, amp, nsteps, c)
    % Signed stance-angle deviation, normalised by its own initial value.
    d0 = amp * abs(z_fp(1));
    z = z_fp(:) + d0 * d(:);
    dev = [];
    for it = 1:nsteps + 1
        dev(end+1) = z(1) - z_fp(1); %#ok<AGROW>
        [z, ~, ok] = wk.step_map(z, c.GAM, c.KHIP, P, c.RTOL, c.ATOL);
        if ~ok, break; end
    end
    tr = dev / dev(1);
end
