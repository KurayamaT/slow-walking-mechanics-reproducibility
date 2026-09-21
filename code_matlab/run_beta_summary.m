%RUN_BETA_SUMMARY  Rebuild sens_beta_summary.csv from the MATLAB foot-mass sweep.
%   The per-beta branches were computed by others/code/floquet/floquet_KUO_v2_beta.m
%   (driven by run_beta_sweep_figure4.m) and stored as
%   others/results/csv/floquet_beta<b>_k-0.160.csv, whose columns are
%   alpha, v, lam_max, lam1_re, lam1_im, lam2_re, lam2_im, lam3_re, lam3_im.
%   This script only summarises them, reproducing the schema of
%   data/sens_beta_summary.csv:
%     beta, n_total, n_stable, v_min_stable, Nhalf_max_stable, min_lam_max, status
BETAS = [0.00 0.05 0.10 0.15 0.20 0.25 0.28];
here = fileparts(mfilename('fullpath'));
src = fullfile(here, '..', 'legacy', 'results', 'csv');
outdir = fullfile(here, '..', 'data_matlab');
if ~exist(outdir, 'dir'), mkdir(outdir); end
out = fullfile(outdir, 'sens_beta_summary.csv');

fh = fopen(out, 'w');
fprintf(fh, 'beta,n_total,n_stable,v_min_stable,Nhalf_max_stable,min_lam_max,status\n');
for b = BETAS
    f = fullfile(src, sprintf('floquet_beta%.3f_k-0.160.csv', b));
    T = readtable(f);
    lam = T.lam_max; v = T.v;
    keep = ~isnan(lam);
    n_total = sum(keep);
    st = keep & lam < 1;
    n_stable = sum(st);
    if n_stable > 0
        vmin = min(v(st));
        Nh = -log(2) ./ log(lam(st));
        NhMax = max(Nh);
        status = 'stable branch present';
        fprintf(fh, '%.2f,%d,%d,%.4f,%.2f,%.4f,%s\n', b, n_total, n_stable, vmin, NhMax, min(lam(keep)), status);
        fprintf('  beta=%.2f  n=%d  n_stable=%d  v_min=%.4f  N_1/2(max)=%.2f  min|lam|=%.4f\n', ...
                b, n_total, n_stable, vmin, NhMax, min(lam(keep)));
    else
        status = 'no stable branch';
        fprintf(fh, '%.2f,%d,%d,NaN,NaN,%.4f,%s\n', b, n_total, n_stable, min(lam(keep)), status);
        fprintf('  beta=%.2f  n=%d  n_stable=0  min|lam|=%.4f  (%s)\n', b, n_total, min(lam(keep)), status);
    end
end
fclose(fh);
fprintf('saved: %s\n', out);
