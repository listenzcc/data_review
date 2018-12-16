close all
clear
clc

figure,
%% for each run
for run_ = 1 : 5
    %% prepare filenames
    tsss_name = sprintf('MultiTraining_%d_raw_tsss', run_);
    mat_name = sprintf('para_guess_%s.mat', tsss_name)
    result_dir = 'results_naive_scale';
    %% load parameters
    load(fullfile(result_dir, mat_name), 'para_guess')
    %% time seriers
    t = linspace(-0.2, 0.8, 1001);
    %% prepare parameters, 306 sensors x 6 orts
    para_A = nan(306, 6);
    para_A0 = nan(306, 6);
    para_t0 = nan(306, 6);
    para_d = nan(306, 6);
    para_w = nan(306, 6);
    para_p = nan(306, 6);
    
    %% for each ort_, fill parameters, plot 306 Gabor
    for ort_ = 1 : 6
        % fill parameters
        for j = 1 : 306
            para_A(j, ort_) = para_guess{j, ort_}(1);
            para_A0(j, ort_) = para_guess{j, ort_}(2);
            para_t0(j, ort_) = para_guess{j, ort_}(3);
            para_d(j, ort_) = para_guess{j, ort_}(4);
            para_w(j, ort_) = para_guess{j, ort_}(5);
            para_p(j, ort_) = para_guess{j, ort_}(6);
        end
        % plot 306 Gabor
        subplot(6, 5, (ort_-1)*5+run_)
        gabor_allch = nan(1001, 306);
        for ch_ = 1 : 306
            para = para_guess{ch_, ort_};
            gabor_allch(:, ch_) = gabor(t, para);
        end
        plot(t, gabor_allch)
        set(gca, 'Box', 'off')
        title(sprintf('run%d, ort%d', run_, ort_))
    end
    
    %% save parameters
    for j = {'A', 'A0', 't0', 'd', 'w', 'p'}
        save(fullfile(result_dir,...
            sprintf('%s_para_%s.txt.mat', tsss_name, j{1})),...
            sprintf('para_%s', j{1}),...
            '-ascii')
    end
end



