close all
clear
clc

t = linspace(-0.2, 0.8, 1001);

load('data_example.mat', 'data_example')
figure
subplot(2, 1, 1)
plot(t, data_example)

data_c = triu(ones(1001))' * data_example / 1001;
data_c = scale(data_c', t)';
subplot(2, 1, 2)
plot(t, data_c)

return

gabor_ = nan(1001, 6);
for j = 1 : 6
    % prepare tmp data for fmincon
    d = data_example(:, j);
    save tmp_data t d
    
%     A = para(1);
%     A0 = para(2);
%     t0 = para(3);
%     d = para(4);
%     w = para(5);
%     p = para(6);
%     w0 = para(7);
%     p0 = para(8);
    
    % set para initial, bottom and top boundry
    ma = max(d(:));
    mi = min(d(:));
    m = mean(d(:));
    para_arr = [ma, m, 0.2, 0.3, 50, 1, 50, 1];
    para_bot = [mi, mi, min(t), 0.1, 40, 0, 40, 0];
    para_top = [ma, ma, max(t), 0.5, 60, pi, 60, pi];
    
    % estimate
    para_guess = fmincon(@norm2loss, para_arr, [], [], [], [], para_bot, para_top)
    
    gabor_(:, j) = gabor(t, para_guess);
end

subplot(2, 1, 2)
plot(t, gabor_)
