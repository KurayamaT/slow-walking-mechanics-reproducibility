function run_beta_sweep_figure4()
%RUN_BETA_SWEEP_FIGURE4  Reproduce the manuscript foot-mass (beta) sweep.
%
%   Produces the data and overlay behind Figure 4 / Section 4.4 / Supplement
%   S6: the spectral radius |lambda_max|(v) for the seven foot-to-hip mass
%   ratios used in the paper, at the paper's hip-spring stiffness.
%
%   Runs floquet_KUO_v2_beta(gam, k_hip, beta) for each beta, which writes
%   one CSV per beta (floquet_gam0.000_km0.160_b{beta}.csv), then overlays
%   |lambda_max|(v) and saves Figure4_beta_sweep.png.
%
%   Parameters (match the manuscript exactly):
%     gam   = 0           (level ground)
%     k_hip = -0.16       (hip-spring stiffness used throughout the paper)
%     beta  in {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.28}
%
%   Requirements
%   ------------
%     - floquet_KUO_v2_beta.m  (same folder)
%     - Parallel Computing Toolbox is used by floquet_KUO_v2_beta (parfor).
%       Without it, MATLAB executes the loop serially (slower, same result).
%
%   Validation
%   ----------
%   This MATLAB pipeline was cross-validated against an independent Python
%   implementation (../analytical/floquet_beta_python.py):
%     * beta=0 reproduces the published Table 1 multipliers to 3 decimals;
%     * the beta>0 equations of motion agree with an algebraically distinct
%       (symmetric-Lagrangian) formulation to machine precision (~1e-15);
%     * end-to-end Floquet multipliers agree to the integration tolerance.
%   See code/MATLAB_REPRODUCE.md and code/analytical/crossval_lwl_beta.py.

    gam   = 0;
    k_hip = -0.16;
    betas = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.28];

    % --- run the sweep (one CSV per beta) ---
    for b = betas
        fprintf('\n========== beta = %.2f ==========\n', b);
        floquet_KUO_v2_beta(gam, k_hip, b);
    end

    % --- overlay |lambda_max|(v) as Figure 4 ---
    figure('Color', 'w', 'Position', [100 100 720 500]); hold on;
    cols = parula(numel(betas));
    for i = 1:numel(betas)
        b  = betas(i);
        fn = strrep(sprintf('floquet_gam%.3f_k%.3f_b%.3f.csv', gam, k_hip, b), '-', 'm');
        if ~isfile(fn)
            warning('run_beta_sweep_figure4:missingCSV', 'missing %s', fn);
            continue;
        end
        Tb = readtable(fn);
        [v, si] = sort(Tb.v);
        lm = Tb.lam_max(si);
        plot(v, lm, '-', 'LineWidth', 1.6, 'Color', cols(i,:), ...
             'DisplayName', sprintf('\\beta = %.2f', b));
    end
    yline(1, 'k--', 'LineWidth', 0.7);
    xlabel('Dimensionless walking speed  v');
    ylabel('|\lambda_{max}|');
    xlim([0 0.20]); ylim([0 1.30]);
    legend('Location', 'southeast'); box on;
    title('Foot-mass (\beta) sweep: low-speed stability transition');
    saveas(gcf, 'Figure4_beta_sweep.png');
    fprintf('\nSaved Figure4_beta_sweep.png\n');
end
