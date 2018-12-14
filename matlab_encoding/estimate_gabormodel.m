function para_guess = estimate_gabormodel(data, events, t)
num_sensor = size(data, 2)
C= unique(events);
num_ort = length(C)

para_guess = cell(num_sensor, num_ort);

fig = waitbar(0);
for ort_idx = 1 : num_ort
    data_ = squeeze(data(events==C(ort_idx), :, :));
    
    for j = 1 : num_sensor
        waitbar(j/num_sensor, fig, sprintf('%d | %d', ort_idx, num_ort))
        
        d = squeeze(data_(:, j, :));
        d = scale(d)';
        save tmp_data t d

        ma = max(d(:));
        mi = min(d(:));
        para_arr = [1, 2, 0.2, 0.3, 10, 0.3];
        para_bot = [mi, mi, min(t), 0, 0, -pi];
        para_top = [ma, ma, max(t), 5, 1/max(t), pi];
        
        para_guess{j, ort_idx} = fmincon(@norm2loss, para_arr, [], [], [], [], para_bot, para_top);
    end
end
close(fig)
end

