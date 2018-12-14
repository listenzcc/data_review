close all
clear
clc

load data

[C, IA, IC] = unique(events);
j = 1;
data_ = squeeze(data(events==C(j), :, :));

t = linspace(-0.2, 0.8, 1001);

para = struct;
para.A = 1;
para.A0 = 2;
para.t0 = 0.2;
para.d = 0.3;
para.w = 10;
para.p = 0.3;

j = 150;
d = squeeze(data_(:, j, :));
d = scale(d)';
save tmp_data t d

ma = max(d(:));
mi = min(d(:));
para_arr = [1, 2, 0.2, 0.3, 10, 0.3];
para_bot = [mi, mi, min(t), 0, 0, -pi];
para_top = [ma, ma, max(t), 5, 1/max(t), pi];

para_guess = fmincon(@norm2loss, para_arr, [], [], [], [], para_bot, para_top)

plot_result(t, para_guess, d)

function plot_result(t, para, d)
y = gabor(t, para);
figure,
plot(t, d, 'color', 0.5 + [0, 0, 0])
hold on
plot(t, y, 'linewidth', 3)
hold off
end

