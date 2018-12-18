close all
clear
clc

plot_debug = false;
result_dir = 'results_naive_scale';

%% prepare gabor model
% para_all, 5 run x 306 sensors x 6 orts x 6 parameters
% gabor_all, 5 run x 306 sensors x 6 orts x 1001 ms
% ts, 1001 ms
para_all = nan(5, 306, 6, 6);
gabor_all = nan(5, 306, 6, 1001);
ts = linspace(-0.2, 0.8, 1001);

% for each run
for run_ = 1 : 5
    % prepare filenames
    tsss_name = sprintf('MultiTraining_%d_raw_tsss', run_);
    mat_name = sprintf('para_guess_%s.mat', tsss_name);
    data_dir = fullfile('..', 'pics', tsss_name);
    % load parameters
    load(fullfile(result_dir, mat_name), 'para_guess')
    % for each ort_, fill parameters
    for ort_ = 1 : 6
        % fill parameters
        for j = 1 : 306
            para_ = para_guess{j, ort_};
            para_all(run_, j, ort_, :) = para_;
            gabor_all(run_, j, ort_, :) = gabor(ts, para_);
        end
    end
end

% plot gabors
if plot_debug
    for run_ = 1 : 5
        figure,
        for ort_ = 1 : 6
            subplot(6, 1, ort_)
            gabor_ = squeeze(gabor_all(run_, :, ort_, :));
            plot(ts, gabor_');
        end
    end
end

% return

%% prepare datas
% data_all, 5 runs x 306 sensors x 72 trails x 1001 ms
% ortidx_all, 5 runs x 72 trails
% ts, 1001 ms
data_all = nan(5, 306, 72, 1001);
ortidx_all = nan(5, 72);
ts = ts;

% for each run
for run_ = 1 : 5
    % prepare filenames
    tsss_name = sprintf('MultiTraining_%d_raw_tsss', run_);
    mat_name = sprintf('para_guess_%s.mat', tsss_name);
    data_dir = fullfile('..', 'pics', tsss_name);
    
    % load events
    events = load(fullfile(data_dir, 'events.txt'));
    events = events(:, 3);
    [C, IA, IC] = unique(events);
    
    % load data
    data_ = load(fullfile(data_dir, sprintf('data_%d.txt', 0)));
    sz_ = size(data_);
    
    % fill data
    parfor j = 1 : 72
        disp(j)
        % c, ort idx
        c = IC(j);
        % data_ size: 306 x 1001
        data_ = load(fullfile(data_dir, sprintf('data_%d.txt', j-1)));
        data_all(run_, :, j, :) = data_;
        ortidx_all(run_, j) = c;
    end
end

data_all = scale(data_all);

%% sort data_all
% data_ort_all, 5 runs x 306 sensors x 6 orts x 1001 ms
data_ort_all = nan(5, 306, 6, 1001);
for run_ = 1 : 5
    for ort_ = 1 : 6
        data_ = data_all(run_, :, ortidx_all(run_, :)==ort_, :);
        data_ort_all(run_, :, ort_, :) = mean(data_, 3);
    end
end

% plot datas
if plot_debug
    for run_ = 1 : 5
        figure,
        for ort_ = 1 : 6
            subplot(6, 1, ort_)
            data_ = squeeze(data_ort_all(run_, :, ort_, :));
            plot(ts, gabor_');
        end
    end
end

%% calculate loss
% loss_train, 5 runs x 306 sensors x 6 orts
loss_train = nan(5, 306, 6);
for run_ = 1 : 5
    for sen_ = 1 : 306
        for ort_ = 1 : 6
            d = squeeze(data_ort_all(run_, sen_, ort_, :));
            g = squeeze(gabor_all(run_, sen_, ort_, :));
            loss_train(run_, sen_, ort_) = cal_loss(d-g);
        end
    end
end

for run_ = 1 : 5
    tsss_name = sprintf('MultiTraining_%d_raw_tsss', run_);
    loss_ = squeeze(loss_train(run_, :, :));
    save(fullfile(result_dir,...
        sprintf('%s_loss_train.txt.mat', tsss_name)),...
        'loss_', '-ascii')
end

figure,
for run_ = 1 : 5
    subplot(5, 1, run_)
    for ort_ = 1 : 6
        loss_ = squeeze(loss_train(run_, :, :));
        plot(loss_);
    end
end


%% estimate
% loss_estimate, 5 runs x 306 sensors x 6 orts x 5 mod_run x 6 mod_ort
loss_estimate = nan(5, 306, 6, 5, 6);
for run_ = 1 : 5
    test_run_ = 1 : 5;
    % test_run_(run_) = '';
    for sen_ = 1 : 306
        for ort_ = 1 : 6
            d = squeeze(data_ort_all(run_, sen_, ort_, :));
            for m_run_ = 1 : 5
                for m_ort_ = 1 : 6
                    g = squeeze(gabor_all(m_run_, sen_, m_ort_, :));
                    loss_estimate(run_, sen_, ort_, m_run_, m_ort_)=...
                        cal_loss(d-g);
                end
            end
        end
    end
end

for run_ = 1 : 5
    figure
    for ort_ = 1 : 6
        loss_ = loss_estimate(run_, :, ort_, :, :);
        loss__ = mean(loss_, 2);
        loss___ = squeeze(loss__);
        subplot(6, 1, ort_)
        plot(loss___', 'o-', 'LineWidth', 2)
        set(gca, 'Box', 'Off')
        set(gca, 'XTick', 1:6)
    end
end
    