clear; clc; close all;

% 加载 .mat 文件
load('sde_surface_data.mat');

% 颜色定义（RGB 值转为 [0,1]）
color = [
    0,   114, 189;
    255, 127, 39;
    128, 128, 128;
    119, 172, 48;
    198, 0,  0
    ] / 255;

% ========== 漂移系数图 f(t, x) ==========
figure;
surf(X, T*1e3, f_masked, 'EdgeColor', 'none', 'FaceAlpha', 0.7, 'HandleVisibility', 'off');
shading interp;
colormap('parula');
xlabel('Crack length (in)', 'FontName', 'Times New Roman', 'FontSize', 14, 'Rotation', 45);
ylabel('Cycles', 'FontName', 'Times New Roman', 'FontSize', 14, 'Rotation', -10);
zlabel('\mu', 'FontName', 'Times New Roman', 'FontSize', 14);
view(-65, 30);
hold on;

% crack data 和 f 系数
crack_data = containers.Map( ...
    {0, 5, 10, 15, 20}, ...
    { ...
    [0.90, 0.95, 1.00, 1.05, 1.12, 1.19, 1.27, 1.35, 1.48, 1.64], ...
    [0.90, 0.94, 0.98, 1.03, 1.07, 1.12, 1.18, 1.23, 1.33, 1.41, 1.51, 1.68], ...
    [0.90, 0.93, 0.96, 1.00, 1.04, 1.08, 1.13, 1.18, 1.24, 1.31, 1.39, 1.49, 1.65], ...
    [0.90, 0.92, 0.95, 0.97, 1.00, 1.03, 1.07, 1.11, 1.16, 1.22, 1.26, 1.33, 1.40], ...
    [0.90, 0.92, 0.94, 0.97, 0.99, 1.02, 1.04, 1.07, 1.11, 1.14, 1.18, 1.22, 1.27] ...
    });

f_data = containers.Map( ...
    {0, 5, 10, 15, 20}, ...
    {f_sample_0, f_sample_5, f_sample_10, f_sample_15, f_sample_20} ...
    );

ids = [0, 5, 10, 15, 20];

for i = 1:length(ids)
    idx = ids(i);
    x_vals = crack_data(idx);
    t_vals = (0:length(x_vals)-1) * 10;
    f_vals = f_data(idx);
    
    % 如果是样本 5, 10, 15，就忽略最后一个点
    % 忽略不同样本尾部点数
    if idx == 10
        x_vals = x_vals(1:end-2);
        t_vals = t_vals(1:end-2);
        f_vals = f_vals(1:end-2); % for f
    elseif ismember(idx, [5, 15])
        x_vals = x_vals(1:end-1);
        t_vals = t_vals(1:end-1);
        f_vals = f_vals(1:end-1); % for f
    end
    
    plot3(x_vals, t_vals*1e3, f_vals, ...
        'o-', 'Color', color(i,:), 'MarkerFaceColor', color(i,:), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('Unit %d', idx+1));
end
legend show;
colorbar
xlim([0.9,1.7])
ylim([0,12e4])
grid on;
ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = 14;  % 自行调节大小，例如 12、14、16
ax.GridLineStyle = '--';


% ========== 扩散系数图 g(t, x) ==========
figure;
surf(X, T*1e3, g_masked, 'EdgeColor', 'none', 'FaceAlpha', 0.7, 'HandleVisibility', 'off');
shading interp;
colormap('hot');
xlabel('Crack length (in)', 'FontName', 'Times New Roman', 'FontSize', 14, 'Rotation', 45);
ylabel('Cycles', 'FontName', 'Times New Roman', 'FontSize', 14, 'Rotation', -10);
zlabel('\sigma', 'FontName', 'Times New Roman', 'FontSize', 14);
view(-65, 30);
hold on;

g_data = containers.Map( ...
    {0, 5, 10, 15, 20}, ...
    {g_sample_0, g_sample_5, g_sample_10, g_sample_15, g_sample_20} ...
    );

for i = 1:length(ids)
    idx = ids(i);
    x_vals = crack_data(idx);
    t_vals = (0:length(x_vals)-1) * 10;
    g_vals = g_data(idx);
    
    % 如果是样本 5, 10, 15，就忽略最后一个点
    % 忽略不同样本尾部点数
    if idx == 10
        x_vals = x_vals(1:end-2);
        t_vals = t_vals(1:end-2);
        g_vals = g_vals(1:end-2); % for g
    elseif ismember(idx, [5, 15])
        x_vals = x_vals(1:end-1);
        t_vals = t_vals(1:end-1);
        g_vals = g_vals(1:end-1); % for g
    end
    
    plot3(x_vals, t_vals*1e3, g_vals, ...
        'o-', 'Color', color(i,:), 'MarkerFaceColor', color(i,:), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('Unit %d', idx+1));
end
legend show;
colorbar
xlim([0.9,1.7])
ylim([0,12e4])
grid on;
ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = 14;  % 自行调节大小，例如 12、14、16
ax.GridLineStyle = '--';

