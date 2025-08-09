clear; clc; close all;

% === 加载所有数据文件 ===
load('prediction_all_samples.mat');                         % baseline
data_abl1 = load('prediction_all_samples_2.mat');        % Ablation 1
data_abl2 = load('prediction_all_samples_8.mat');        % Ablation 2
data_abl3 = load('prediction_all_samples_16.mat');       % Ablation 3

% 手动设置颜色（与3D图中一致）
color_map = {
    [0.3010, 0.7450, 0.9330], ...
    [0, 0.4470, 0.7410], ...    % Proposed neural SDE - 蓝色
    [0.8500, 0.3250, 0.0980], ...  % Zhang - 橙色
    [0.9290, 0.6940, 0.1250], ...  % Ablation 1 - 金色
};

% === 设置统一字体 ===
fontName = 'Times New Roman';
fontSize = 12;

figure('Position', [10, 4, 1600, 1000]);  % 控制整体大小
tiledlayout(6, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

sample_indices = [2:10, 12:18, 20:21];  % 指定 sample 编号

for idx = 1:18
    sample_name = sprintf("sample_%02d", idx);
    s = eval(sample_name);
    s_abl1 = data_abl1.(sample_name);
    s_abl2 = data_abl2.(sample_name);
    s_abl3 = data_abl3.(sample_name);

    nexttile;
    hold on;

    % 不确定性带
    ts = s.ts(:);
    lower = s.lower(:);
    upper = s.upper(:);
    fill([ts*1e3; flipud(ts*1e3)], [lower; flipud(upper)], [0.8 0.8 0.8], ...
        'EdgeColor', 'none', 'FaceAlpha', 0.8);

    % 观测点
    plot(s.observed_ts*1e3, s.observed_vals, 'ko-', 'MarkerSize', 4, 'LineWidth', 1);

    % 真实轨迹
    if ~isempty(s.crack_x)
        crack_x = s.crack_x;
        crack_y = s.crack_y;
        plot(crack_x(10:end)*1e3, crack_y(10:end), 'r-', 'LineWidth', 2);
        xlim([70*1e3, max(crack_x)*1e3]);
    else
        xlim([70*1e3, 120*1e3]);
    end

    % 模型预测均值
    plot(s.ts*1e3, s.mean, '--', 'LineWidth', 2, 'Color', color_map{1});        % baseline
    plot(s_abl1.ts*1e3, s_abl1.mean, '-', 'LineWidth', 1, 'Color', color_map{2}); % ablation 1
    plot(s_abl2.ts*1e3, s_abl2.mean, '-', 'LineWidth', 1, 'Color', color_map{3}); % ablation 2
    plot(s_abl3.ts*1e3, s_abl3.mean, '-', 'LineWidth', 1, 'Color', color_map{4}); % ablation 3

    % 坐标轴和标题
    set(gca, 'FontName', fontName, 'FontSize', fontSize);
    title(sprintf("Unit %d", sample_indices(idx)), 'FontSize', fontSize, 'FontName', fontName);
    grid on;
    set(gca, 'GridLineStyle', '--');

    ylims = ylim;
    ymin = floor(ylims(1)*5)/5;
    ymax = ceil(ylims(2)*5)/5;
    yt = ymin:0.2:ymax;
    yt = round(yt, 1);
    yticks(yt);

    xlabel('Cycles', 'FontName', fontName, 'FontSize', fontSize);
    ylabel('Crack Size (in)', 'FontName', fontName, 'FontSize', fontSize);

    % 图例
    if idx == 2
        legend({'Boundaries of the proposed method at 95% CL', ...
            'Observed data', ...
            'True degradation', ...
            'Proposed model', ...
            'Ablation 1: age-dependent drift', ...
            'Ablation 2: age-dependent drift and diffusion', ...
            'Ablation 3: age- and state-dependent drift'}, ...
            'FontSize', fontSize - 1, 'FontName', fontName, ...
            'Orientation', 'horizontal', ...
            'NumColumns', 3, ...
            'Location', 'northoutside');
    end

    % ========== RMSE & CRPS ==========
    y_pred_samples = s.samples';  % baseline
    y_true = s.crack_y(:);

    crack_x_test = crack_x(crack_x > 90);
    ts_pred = 90:2:(90 + size(y_pred_samples, 1) - 1);
    test_indices = arrayfun(@(t) find(ismembertol(ts_pred, t, 1e-6), 1), crack_x_test, 'UniformOutput', false);
    test_indices = test_indices(~cellfun(@isempty, test_indices));
    test_indices = cell2mat(test_indices);

    y_pred_samples_test = y_pred_samples(test_indices, :);  % [T_test × 2000]
    y_true_test = y_true(11:end);

    % === baseline RMSE & CRPS ===
    y_pred_mean = mean(y_pred_samples_test, 2);
    rmse_all(2, idx) = sqrt(mean((y_pred_mean - y_true_test).^2));

    crps_vals = zeros(length(test_indices), 1);
    for t = 1:length(test_indices)
        pred_dist = y_pred_samples_test(t, :);
        obs = y_true_test(t);
        crps_vals(t) = mean(abs(pred_dist - obs)/obs) - 0.5 * mean(mean(abs(pred_dist' - pred_dist)/obs));
    end
    crps_all(2, idx) = mean(crps_vals);

    % === ablation 1 ===
    abl1_samples = s_abl1.samples';                        % [T × 2000]
    abl1_mean = mean(abl1_samples, 2);                     % mean across samples
    abl1_mean_test = abl1_mean(test_indices);              % only test indices
    rmse_all(1, idx) = sqrt(mean((abl1_mean_test - y_true_test).^2));

    crps_vals = zeros(length(test_indices), 1);
    abl1_test_samples = abl1_samples(test_indices, :);
    for t = 1:length(test_indices)
        pred_dist = abl1_test_samples(t, :);
        obs = y_true_test(t);
        crps_vals(t) = mean(abs(pred_dist - obs)/obs) - 0.5 * mean(mean(abs(pred_dist' - pred_dist)/obs));
    end
    crps_all(1, idx) = mean(crps_vals);

    % === ablation 2 ===
    abl2_samples = s_abl2.samples';
    abl2_mean = mean(abl2_samples, 2);
    abl2_mean_test = abl2_mean(test_indices);
    rmse_all(3, idx) = sqrt(mean((abl2_mean_test - y_true_test).^2));

    crps_vals = zeros(length(test_indices), 1);
    abl2_test_samples = abl2_samples(test_indices, :);
    for t = 1:length(test_indices)
        pred_dist = abl2_test_samples(t, :);
        obs = y_true_test(t);
        crps_vals(t) = mean(abs(pred_dist - obs)/obs) - 0.5 * mean(mean(abs(pred_dist' - pred_dist)/obs));
    end
    crps_all(3, idx) = mean(crps_vals);

    % === ablation 3 ===
    abl3_samples = s_abl3.samples';
    abl3_mean = mean(abl3_samples, 2);
    abl3_mean_test = abl3_mean(test_indices);
    rmse_all(4, idx) = sqrt(mean((abl3_mean_test - y_true_test).^2));

    crps_vals = zeros(length(test_indices), 1);
    abl3_test_samples = abl3_samples(test_indices, :);
    for t = 1:length(test_indices)
        pred_dist = abl3_test_samples(t, :);
        obs = y_true_test(t);
        crps_vals(t) = mean(abs(pred_dist - obs)/obs) - 0.5 * mean(mean(abs(pred_dist' - pred_dist)/obs));
    end
    crps_all(4, idx) = mean(crps_vals);
end

set(gcf,'unit','centimeters','position',[10 1 19 40]);

% === RMSE Plot ===
figure; hold on
line_styles = {'--', '--', '--', '--'};
marker_list = {'^', 'o', 'h', 'd'};
method_names = {
    'G = 2', ...
    'G = 4', ...
    'G = 8', ...
    'G = 16'
};
font_size = 14;

for m = 1:length(method_names)
    plot(sample_indices, rmse_all(m,:), ...
        'Color', color_map{m}, ...
        'LineStyle', line_styles{m}, ...
        'Marker', marker_list{m}, ...
        'LineWidth', 1, ...
        'MarkerSize', 6, ...
        'DisplayName', method_names{m});
end

xlabel('Unit number', 'FontName', fontName, 'FontSize', font_size);
ylabel('RMSE', 'FontName', fontName, 'FontSize', font_size);
legend('Location', 'northwest', 'FontName', fontName, 'FontSize', font_size);
grid on
set(gca, 'GridLineStyle', '--')
set(gca, 'FontName', fontName, 'FontSize', font_size);
ylim([0, 0.1])

% === CRPS Plot ===
figure; hold on
for m = 1:length(method_names)
    plot(sample_indices, crps_all(m,:), ...
        'Color', color_map{m}, ...
        'LineStyle', line_styles{m}, ...
        'Marker', marker_list{m}, ...
        'LineWidth', 1, ...
        'MarkerSize', 6, ...
        'DisplayName', method_names{m});
end

xlabel('Unit number', 'FontName', fontName, 'FontSize', font_size);
ylabel('NCRPS', 'FontName', fontName, 'FontSize', font_size);
legend('Location', 'northwest', 'FontName', fontName, 'FontSize', font_size);
grid on
set(gca, 'GridLineStyle', '--')
set(gca, 'FontName', fontName, 'FontSize', font_size);
ylim([0, 0.04])
