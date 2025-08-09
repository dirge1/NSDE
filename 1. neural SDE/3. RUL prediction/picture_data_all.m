clear; clc; close all
load crack_data.mat

% 字体设置
font_name = 'Times New Roman';
font_size = 12;

sample_indices = [1:10, 12:18, 20:21];  % 指定样本编号
T_min = 0;
T_max = 90;

figure;
% 图例用的 NaN 曲线（提前定义）
% -------- subplot 1 --------
hold on;
plot(NaN, NaN, 'bo-', 'DisplayName', 'Data for model establishment');
plot(NaN, NaN, 'r*-', 'DisplayName', 'Data for degradation prediction');
plot(NaN, NaN, 'kd-', 'DisplayName', 'Data for RUL prediction');
for i = sample_indices
    y = crack_data{i};                
    N = length(y);                    
    time = 0:10:(N-1)*10;             

    in_range = (time >= T_min) & (time <= T_max);
    out_range = (time < T_min) | (time > T_max) | (time == T_max);

    plot(time*1e3, y, 'bo-');   
end

xlabel('Cycles', 'FontName', font_name, 'FontSize', font_size);
ylabel('Crack length (in)', 'FontName', font_name, 'FontSize', font_size);

grid on;
set(gca, 'GridLineStyle', '--', 'FontName', font_name, 'FontSize', font_size);
