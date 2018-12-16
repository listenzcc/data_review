close all
clear
clc
%% prepare parameters, 5 run x 306 sensors x 6 orts x 6 parameters
para_all = nan(5, 306, 6, 6);

%% time seriers
t = linspace(-0.2, 0.8, 1001);

%% for each run
for run_ = 1 : 5
    %% prepare filenames
    tsss_name = sprintf('MultiTraining_%d_raw_tsss', run_);
    mat_name = sprintf('para_guess_%s.mat', tsss_name);
    data_dir = fullfile('..', 'pics', tsss_name);
    result_dir = 'results_naive_scale';
    %% load parameters
    load(fullfile(result_dir, mat_name), 'para_guess')
    
    %% for each ort_, fill parameters
    for ort_ = 1 : 6
        % fill parameters
        for j = 1 : 306
            para_all(run_, j, ort_, :) = para_guess{j, ort_};
        end
    end
end

%% for each run
for run_ = 1 %: 5
    %% prepare filenames
    tsss_name = sprintf('MultiTraining_%d_raw_tsss', run_);
    mat_name = sprintf('para_guess_%s.mat', tsss_name);
    data_dir = fullfile('..', 'pics', tsss_name);
    result_dir = 'results_naive_scale';
    
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
    
    %% sort data as 6 orts
    data_ort = cell(6, 1);
    % for each ort
    for ort_ = 1 : 6
        data_ort{ort_, 1} = data(IC==ort_, :, :);
    end
    
    %% calculate loss, 306 sensors x 6 orts x 6 model_orts x 5 run_models
    loss_this = nan(306, 6, 6, 5);
    for ort_ = 1 : 6
        disp(ort_)
        for sen_ = 1 : 306
            data_ = squeeze(data_ort{ort_}(:, sen_, :));
            data_ = scale(data_)';
            for mod_ort_ = 1 : 6
                for run_model_ = 1 : 5
                    para_ = squeeze(para_all(run_model_, sen_, mod_ort_, :));
                    gabor_ = gabor(t, para_);
                    loss_this(sen_, ort_, mod_ort_, run_model_) =...
                        sum(cal_loss(data_ - repmat(gabor_, 1, 12)));
                end
            end
        end
    end
    figure,
    for ort_ = 1 : 6
        subplot(6, 1, ort_)
        hold on
        a = nan(5, 1);
        for run_model_ = 1 : 5
            loss_ = squeeze(loss_this(:, ort_, :, run_model_));
            a(run_model_) = plot(mean(loss_), 'o-');
        end
        hold off
        legend(a, {'1', '2', '3', '4', '5'})
        set(gca, 'XTick', 1:6)
        set(gca, 'Box', 'off')
        title(ort_)
    end
    
end


