clear; clc; close all;

% ========== 参数 ==========
file_path = 'all_results_full.mat';
crack_mat_file = 'crack_data.mat';
data = load(file_path);
mat = load(crack_mat_file);

max_t = 200;
% threshold = 1.6;
% sample_idx = 0;

threshold = 1.3;
sample_idx = 1;

selected_indices = [11, 19];  % 用于查找 crack_data
time_indices = 5:10;
obs_lengths = time_indices;  % 每个 time_index 对应观测长度
step_size = 10;  % 每个观测点的时间间隔（单位：秒）
used_times = obs_lengths * step_size;

% 构造 crack_data map：sample_idx(0-based) → 裂纹轨迹
raw_data = mat.crack_data(:);
crack_data = containers.Map('KeyType', 'double', 'ValueType', 'any');
for i = 1:length(selected_indices)
    crack_data(i-1) = raw_data{selected_indices(i)};
end

% ========== 计算真实失效时间 ==========
crack_y = crack_data(sample_idx);
crack_x = (0:length(crack_y)-1) * 0.05 * max_t;
crack_y_vals = crack_y(:);
crack_x_vals = crack_x(:);
idx_exceed = find(crack_y_vals > threshold, 1);

if isempty(idx_exceed)
    true_fail_time = NaN;
else
    if idx_exceed == 1
        true_fail_time = crack_x_vals(1);
    else
        x0 = crack_y_vals(idx_exceed - 1);
        x1 = crack_y_vals(idx_exceed);
        t0 = crack_x_vals(idx_exceed - 1);
        t1 = crack_x_vals(idx_exceed);
        frac = (threshold - x0) / (x1 - x0);
        true_fail_time = t0 + frac * (t1 - t0);
    end
end

% ========== 计算真实RUL ==========
true_RULs = true_fail_time - used_times +step_size;

% ========== 预测RUL分布 ==========
rul_grid = linspace(0, 150, 500);
pdf_matrix = NaN(length(time_indices), length(rul_grid));
mean_RULs = NaN(length(time_indices), 1);

for ti = 1:length(time_indices)
    time_index = time_indices(ti);
    prefix = sprintf('t%d_sample_%02d', time_index, sample_idx);

    samples = data.([prefix '_samples']);         % [500, T]
    ts_pred = data.([prefix '_ts']) * max_t;

    num_samples = size(samples, 1);
    RULs = NaN(num_samples, 1);

    for i = 1:num_samples
        traj = samples(i, :);
        idx = find(traj > threshold, 1);

        if isempty(idx)
            continue;
        elseif idx == 1
            RULs(i) = ts_pred(1);
        else
            x0 = traj(idx - 1); x1 = traj(idx);
            t0 = ts_pred(idx - 1); t1 = ts_pred(idx);
            frac = (threshold - x0) / (x1 - x0);
            failure_time = t0 + frac * (t1 - t0);
            RULs(i) = failure_time - ts_pred(1);
        end
    end

    valid_RULs = RULs(~isnan(RULs));
    if ~isempty(valid_RULs)
        [f, ~] = ksdensity(valid_RULs, rul_grid);
        pdf_matrix(ti, :) = f;
        mean_RULs(ti) = mean(valid_RULs);
    end
end

% ========== 绘图 ==========
figure('Position', [100, 100, 1000, 550]); hold on;
h_pdf = [];

for ti = 1:length(time_indices)
    time_val = time_indices(ti);
    rul_vals = rul_grid;
    pdf_vals = pdf_matrix(ti, :);

    if ti == 1
        h_pdf = plot3(ones(size(rul_vals)) * time_val, rul_vals, pdf_vals, ...
            'b-', 'LineWidth', 1.5, 'DisplayName', 'PDF Curves');
    else
        plot3(ones(size(rul_vals)) * time_val, rul_vals, pdf_vals, ...
            'b-', 'LineWidth', 1.5);
    end
end

% ========== 添加均值预测线 ==========
h_mean = plot3(time_indices, mean_RULs, zeros(size(mean_RULs)), ...
    '--or', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Predicted RUL');

% ========== 添加真实RUL线 ==========
h_true = plot3(time_indices, true_RULs, zeros(size(true_RULs)), ...
    '-sg', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'True RUL');

% ========== 图像格式 ==========
xlabel('Observation Length (time index)');
ylabel('Remaining Useful Life (s)');
zlabel('PDF');
title(sprintf('RUL PDF vs Time Index (Threshold = %.1f)', threshold));
% view(45, 30);
view(-70, 35);
grid on;
legend([h_pdf, h_mean, h_true], {'PDF Curves', 'Predicted RUL', 'True RUL'});
