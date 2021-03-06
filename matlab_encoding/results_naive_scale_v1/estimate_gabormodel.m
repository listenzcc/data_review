function para_guess = estimate_gabormodel(data, events, t)
%% data: raw data, events: events info of data, t: time seriers
%% prepare nums
num_sensor = size(data, 2)
C= unique(events);
num_ort = length(C)

%% initial para_guess cell
para_guess = cell(num_sensor, num_ort);

%% estimate Gabor parameter
fig = waitbar(0);
for ort_idx = 1 : num_ort
    % for each ort_
    % prepare data
    data_ = squeeze(data(events==C(ort_idx), :, :));
    
    for j = 1 : num_sensor
        % for each sensor_
        waitbar(j/num_sensor, fig, sprintf('%d | %d', ort_idx, num_ort))
        
        % prepare tmp data for fmincon
        d = squeeze(data_(:, j, :));
        d = scale(d, t)';
        d = mean(d, 2);
        save tmp_data t d
        
        % set para initial, bottom and top boundry
        ma = max(d(:));
        mi = min(d(:));
        m = mean(d(:));
        para_arr = [ma, m, 0.2, 0.3, 50, 1, 50, 1];
        para_bot = [mi, mi, min(t), 0.1, 40, 0, 40, 0];
        para_top = [ma, ma, max(t), 0.5, 60, pi, 60, pi];
        
        % estimate
        para_guess{j, ort_idx} = fmincon(@norm2loss, para_arr, [], [], [], [], para_bot, para_top);
    end
end
close(fig)
end

