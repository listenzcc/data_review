close all
clear
clc

figure,
for run_ = 1 : 5
    tsss_name = sprintf('MultiTraining_%d_raw_tsss', run_);
    
    mat_name = sprintf('para_guess_%s.mat', tsss_name)
    
    load(mat_name)
    
    t = linspace(-0.2, 0.8, 1001);
    
    for ort_ = 1 : 6
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
    
end