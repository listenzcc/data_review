close all
clear
clc

for run_ = 1 : 5
    %% prepare path
    tsss_name = sprintf('MultiTraining_%d_raw_tsss', run_);
    data_dir = fullfile('..', 'pics', tsss_name)
    
    %% load events
    events = load(fullfile(data_dir, 'events.txt'));
    events = events(:, 3);
    [C, IA, IC] = unique(events);
    num_events = size(events, 1)
    num_class = max(IC)
    
    %% load data
    data_ = load(fullfile(data_dir, sprintf('data_%d.txt', 0)));
    sz_ = size(data_);
    
    %% fill data
    data = nan([num_events, sz_]);
    for j = 1 : num_events
        disp(j)
        c = IC(j);
        data_ = load(fullfile(data_dir, sprintf('data_%d.txt', j-1)));
        data(j, :, :) = data_;
    end
    
    %% plot mean data
    % figure,
    % for j = 1 : 6
    %     subplot(2, 3, j)
    %     data_mean = squeeze(mean(data(events==C(j), :, :), 1))';
    %     plot(data_mean)
    %     title(C(j))
    % end
    
    %% estimate Gabor parameters
    t = linspace(-0.2, 0.8, 1001);
    para_guess = estimate_gabormodel(data, events, t);
    
    %% save Gabor parameters
    save(sprintf('para_guess_%s.mat', tsss_name), 'para_guess')
    
end