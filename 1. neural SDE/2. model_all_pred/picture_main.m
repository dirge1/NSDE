clear; clc; close all

%% ==== 参数设置 ====
confidence = 95;  % 自定义置信度，例如 90 / 95 / 99
alpha = (100 - confidence) / 2;
font_size = 14;
%% ==== 加载数据 ====
load('sde_simulations.mat');  % 包含 ts 和 simulations
max_t = 200;
ts = ts(:)';  % [1 x T]
simulations = permute(simulations, [1 2]);  % [N x T]

%% ==== 计算置信区间和均值 ====
mean_sim = mean(simulations, 1);
lower = prctile(simulations, alpha, 1);
upper = prctile(simulations, 100 - alpha, 1);

%% ==== 真实样本数据 ====
raw_samples = {
    [0,10,20,30,40,50,60,70,80,90], [0.90, 0.95, 1.00, 1.05, 1.12, 1.19, 1.27, 1.35, 1.48, 1.64];
    [0,10,20,30,40,50,60,70,80,90], [0.90, 0.94, 0.98, 1.03, 1.08, 1.14, 1.21, 1.28, 1.37, 1.47];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.94, 0.98, 1.03, 1.08, 1.13, 1.19, 1.26, 1.35, 1.46];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.94, 0.98, 1.03, 1.07, 1.12, 1.19, 1.25, 1.34, 1.43];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.94, 0.98, 1.03, 1.07, 1.12, 1.19, 1.24, 1.34, 1.43];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.94, 0.98, 1.03, 1.07, 1.12, 1.18, 1.23, 1.33, 1.41];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.94, 0.98, 1.02, 1.07, 1.11, 1.17, 1.23, 1.32, 1.41];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.93, 0.97, 1.00, 1.06, 1.11, 1.17, 1.23, 1.30, 1.39];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.92, 0.97, 1.01, 1.05, 1.09, 1.15, 1.21, 1.28, 1.36];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.92, 0.96, 1.00, 1.04, 1.08, 1.13, 1.19, 1.26, 1.34];
%     [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.93, 0.96, 1.00, 1.04, 1.08, 1.13, 1.18, 1.24, 1.31];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.93, 0.97, 1.00, 1.03, 1.07, 1.10, 1.16, 1.22, 1.29];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.92, 0.97, 0.99, 1.03, 1.06, 1.10, 1.14, 1.20, 1.26];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.93, 0.96, 1.00, 1.03, 1.07, 1.12, 1.16, 1.20, 1.26];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.92, 0.96, 0.99, 1.03, 1.06, 1.10, 1.16, 1.21, 1.27];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.92, 0.95, 0.97, 1.00, 1.03, 1.07, 1.11, 1.16, 1.22];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.93, 0.96, 0.97, 1.00, 1.05, 1.08, 1.11, 1.16, 1.20];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.92, 0.94, 0.97, 1.01, 1.04, 1.07, 1.09, 1.14, 1.19];
%     [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.92, 0.94, 0.97, 0.99, 1.02, 1.05, 1.08, 1.12, 1.16];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.92, 0.94, 0.97, 0.99, 1.02, 1.05, 1.08, 1.12, 1.16];
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0.90, 0.92, 0.94, 0.97, 0.99, 1.02, 1.04, 1.07, 1.11, 1.14]
};

n_samples = size(raw_samples, 1);
real_curves = cell(n_samples, 1);
for i = 1:n_samples
    ts_real = raw_samples{i,1} / max_t;
    xs_real = raw_samples{i,2};
    real_curves{i} = [ts_real(:)'; xs_real(:)'];
end

%% ==== 计算真实均值 ====
all_real_xs = cell2mat(cellfun(@(c) c(2,:), real_curves, 'UniformOutput', false));
real_mean = mean(all_real_xs, 1);
real_ts = raw_samples{1,1};

%% ==== 绘图 ====
figure;
hold on;
fill([ts * max_t*1e3, fliplr(ts * max_t*1e3)], [lower, fliplr(upper)], [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
plot(ts * max_t*1e3, mean_sim, 'b-', 'LineWidth', 2);

% 使用浅灰色统一绘制所有真实曲线
for i = 1:n_samples
    if i == 1
        % 第一条加入图例
        plot(real_curves{i}(1,:) * max_t*1e3, real_curves{i}(2,:), '-', ...
            'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.2, 'DisplayName', 'Real Samples');
    else
        % 其他不加入图例
        plot(real_curves{i}(1,:) * max_t*1e3, real_curves{i}(2,:), '-', ...
            'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.2, 'HandleVisibility', 'off');
    end
end


plot(real_ts*1e3, real_mean, 'k--', 'LineWidth', 2);

xlabel('Cycles');
ylabel('Crack length (in)');
legend('Boundaries at 95% CL', 'Predicted mean degradation', 'Samples', 'True mean degradation', 'Location', 'NorthWest');
grid on         % 打开网格
set(gca, 'GridLineStyle', '--')  % 设置网格线为虚线

xlim([0 9e4]);
ylim([0.9 1.75]);
hold off;
% ? 统一设置字体
set(gca, 'FontName', 'Times New Roman', 'FontSize', font_size);
set(get(gca, 'XLabel'), 'FontName', 'Times New Roman', 'FontSize', font_size);
set(get(gca, 'YLabel'), 'FontName', 'Times New Roman', 'FontSize', font_size);
set(findall(gcf,'Type','Legend'), 'FontName', 'Times New Roman', 'FontSize', font_size);
%% ==== 误差计算 ====
% 找出每个 real_ts 在 ts 中的最近点
ts_pred = ts * max_t;
indices = arrayfun(@(t) find(abs(ts_pred - t) == min(abs(ts_pred - t)), 1), real_ts);
pred_at_real_ts = mean_sim(indices);

abs_error = abs(pred_at_real_ts - real_mean);
mae = mean(abs_error);

fprintf('\n? 每个时间点的误差:\n');
for i = 1:length(real_ts)
    fprintf('t = %3d: predicted = %.4f, real = %.4f, abs error = %.4f\n', ...
        real_ts(i), pred_at_real_ts(i), real_mean(i), abs_error(i));
end
fprintf('\n? 平均绝对误差 (MAE) = %.4f\n', mae);
