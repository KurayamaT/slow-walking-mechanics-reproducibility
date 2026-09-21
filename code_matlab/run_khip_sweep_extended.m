%RUN_KHIP_SWEEP_EXTENDED  Hip-spring robustness with realised step length and speed.
%   Nine k_hip branches from -0.30 to +0.10 are traced over q = 0.010:0.001:0.800.
%   Cached branches are read from data_matlab/branches/khip_<k>.mat and computed
%   if missing. To parallelise, call khip_branch(k) in separate MATLAB sessions.
%   Set SLOWWALK_KHIP_SUMMARY_ONLY=1 to regenerate summaries from the existing
%   supp_khip_sweep.csv without recomputing or rewriting the branch data.
%   Writes sens_khip_summary.csv and sens_khip_scaling.csv in data_matlab/.
KHIPS = [-0.30, -0.20, -0.16, -0.12, -0.08, -0.04, 0.00, 0.05, 0.10];
here = fileparts(mfilename('fullpath'));
datadir = fullfile(here, '..', 'data_matlab');
if ~exist(datadir, 'dir'), mkdir(datadir); end
branch_csv = fullfile(datadir, 'supp_khip_sweep.csv');
summary_only = strcmp(getenv('SLOWWALK_KHIP_SUMMARY_ONLY'), '1');

tic;
if ~summary_only
    B = cell(1, numel(KHIPS));
    for i = 1:numel(KHIPS)
        f = fullfile(datadir, 'branches', sprintf('khip_%+.3f.mat', KHIPS(i)));
        if ~exist(f, 'file'), khip_branch(KHIPS(i)); end
        B{i} = load(f);
    end
    fh = fopen(branch_csv, 'w');
    assert(fh >= 0, 'Cannot open branch CSV for writing.');
    fprintf(fh, 'k_hip,q,alpha_h,s,v,lam_max,N_half,lam1_im\n');
    for i = 1:numel(KHIPS)
        b = B{i};
        if isfield(b, 'q'), q = b.q; else, q = b.s; end
        s_real = 2 * sin(b.Z(1, :));
        v_real = s_real ./ b.T;
        for j = 1:numel(q)
            fprintf(fh, '%.3f,%.3f,%.10g,%.10g,%.10g,%.10f,%s,%.10f\n', ...
                KHIPS(i), q(j), b.Z(1, j), s_real(j), v_real(j), ...
                b.lam_max(j), fmt_nan(b.N_half(j), '%.4f'), b.lam1_im(j) + 0);
        end
    end
    fclose(fh);
    fprintf('saved: %s\n', branch_csv);
end

assert(isfile(branch_csv), 'The branch CSV is required in summary-only mode.');
d = readtable(branch_csv);
assert(isequal(sort(unique(d.k_hip)), KHIPS(:)), 'Expected all nine hip-spring conditions.');
assert(all(d.s > 0 & d.v > 0), 'Realised step lengths and speeds must be positive.');
assert(all(abs(d.s - 2 * sin(d.alpha_h)) < 1e-9), 'Step lengths do not match realised angles.');
s_min_all = zeros(size(KHIPS));
for i = 1:numel(KHIPS)
    s_min_all(i) = min(d.s(abs(d.k_hip - KHIPS(i)) < 1e-9));
end
s_common = max(s_min_all);
fprintf('\ncommon realised step length s = %.10g\n', s_common);

summary_path = fullfile(datadir, 'sens_khip_summary.csv');
scaling_path = fullfile(datadir, 'sens_khip_scaling.csv');
fh = fopen(summary_path, 'w');
fs = fopen(scaling_path, 'w');
assert(fh >= 0 && fs >= 0, 'Cannot open summary outputs for writing.');
fprintf(fh, 'k_hip,n_stable,q_min,s_min_real,v_min,Nhalf_at_qmin,s_common,Nhalf_at_common_s,Nhalf_max_lowspeed,v_eigtype_transition\n');
fprintf(fs, 'k_hip,s_max,n,exponent,coefficient,r_squared\n');
for i = 1:numel(KHIPS)
    b = sortrows(d(abs(d.k_hip - KHIPS(i)) < 1e-9, :), 'q');
    assert(all(diff(b.s) > 0) && all(diff(b.v) > 0), 'Expected monotonic realised step length and speed.');
    st = b.lam_max > 0 & b.lam_max < 1 & isfinite(b.N_half);
    assert(st(1), 'The shortest-step gait must be stable.');
    nh_common = interp1(b.s(st), b.N_half(st), s_common, 'linear');
    assert(isfinite(nh_common), 'The common realised step length lies outside a stable branch.');
    isc = abs(b.lam1_im) > 1e-6;
    j = find(isc(2:end) & ~isc(1:end-1) & st(2:end) & st(1:end-1), 1) + 1;
    if isempty(j), vt = NaN; else, vt = mean(b.v(j-1:j)); end
    nhlow = max(b.N_half(st & b.v < 0.06));
    if isempty(nhlow), nhlow = NaN; end
    fprintf(fh, '%.3f,%d,%.3f,%.10g,%.10g,%.2f,%.10g,%.2f,%.2f,%.10g\n', ...
        KHIPS(i), sum(st), b.q(1), min(b.s), b.v(1), b.N_half(1), ...
        s_common, nh_common, nhlow, vt);

    m = st & b.s <= 0.05;
    assert(sum(m) >= 5, 'Too few short-step gaits for the scaling fit.');
    x = log(b.s(m));
    y = log(1 - b.lam_max(m));
    p = polyfit(x, y, 1);
    r2 = 1 - sum((y - polyval(p, x)).^2) / sum((y - mean(y)).^2);
    fprintf(fs, '%.3f,0.05,%d,%.15g,%.15g,%.15g\n', ...
        KHIPS(i), sum(m), p(1), exp(p(2)), r2);
    fprintf('  k_hip=%+.3f  n_stable=%3d  N_1/2(s_common)=%8.2f  v*=%.10f  n_fit=%d  p=%.10f  C=%.10f  R2=%.10f\n', ...
        KHIPS(i), sum(st), nh_common, vt, sum(m), p(1), exp(p(2)), r2);
end
fclose(fh);
fclose(fs);
fprintf('saved: %s\nsaved: %s  (%.1f s)\n', summary_path, scaling_path, toc);

function str = fmt_nan(x, fmt)
    if isnan(x), str = 'nan'; else, str = sprintf(fmt, x); end
end
