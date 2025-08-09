clear; clc; close all;

% === 加载所有数据文件 ===
load('prediction_all_samples.mat');              % baseline
data_exp = load('prediction_all_samples_exp.mat');       % exp模型
data_general = load('prediction_all_samples_general.mat');   % general模型

% 手动设置颜色（与3D图中一致）
color_map = {
    [0, 0.4470, 0.7410], ...    % Proposed neural SDE - 蓝色
    [0.3010, 0.7450, 0.9330], ...  % Iannacone - 浅蓝
    [0.8500, 0.3250, 0.0980], ...  % Zhang - 橙色
    [0.9290, 0.6940, 0.1250], ...  % Ablation 1 - 金色
    [0.4940, 0.1840, 0.5560], ...  % Ablation 2 - 紫红色
    [0.4660, 0.6740, 0.1880]   ...% Ablation 3 - 草绿色
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
    s_exp = data_exp.(sample_name);
    s_general = data_general.(sample_name);
    
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
    
    % 三个模型的预测均值
    plot(s.ts*1e3, s.mean, 'b--', 'LineWidth', 2);                % baseline
    plot(s_exp.ts*1e3, s_exp.mean, 'm-', 'LineWidth', 1);        % exp
    plot(s_general.ts*1e3, s_general.mean, 'k--', 'LineWidth', 1);% general
    
    % 坐标轴和标题
    set(gca, 'FontName', fontName, 'FontSize', fontSize);
    title(sprintf("Unit %d", sample_indices(idx)), 'FontSize', fontSize, 'FontName', fontName);
    grid on;
    set(gca, 'GridLineStyle', '--');
    
    % Y 轴刻度
    ylims = ylim;
    ymin = floor(ylims(1)*5)/5;
    ymax = ceil(ylims(2)*5)/5;
    yt = ymin:0.2:ymax;
    yt = round(yt, 1);
    yticks(yt);
    
    % 标签
    xlabel('Cycles', 'FontName', fontName, 'FontSize', fontSize);
    ylabel('Crack Size (in)', 'FontName', fontName, 'FontSize', fontSize);
    
    % 图例
    if idx == 2
        legend({'Boundaries of the proposed method at 95% CL', ...
            'Observed data', ...
            'True degradation', ...
            'Proposed model', ...
            'Allen''s model', ...
            'Zhang''s model'}, ...
            'FontSize', fontSize - 1, 'FontName', fontName, ...
            'Orientation', 'horizontal', ...
            'NumColumns', 3, ...
            'Location', 'northoutside');
    end
    
    y_pred_samples = s.samples';   % [T × 2000]
    y_true = s.crack_y(:);         % [T × 1]
    
    
    % 假设 crack_x 单位是千 cycles，比如 [85, 90, 95, ..., 120]
    crack_x_test = crack_x(crack_x > 90);  % 只取 >=90 的点
    
    % y_pred_samples 的时间轴：
    ts_pred = 90:2:(90 + size(y_pred_samples, 1) - 1);  % 与行数对齐
    
    % 找出 crack_x_test 中每个时间在 ts_pred 中的索引
    test_indices = arrayfun(@(t) find(ismembertol(ts_pred, t, 1e-6), 1), crack_x_test, 'UniformOutput', false);
    test_indices = test_indices(~cellfun(@isempty, test_indices));  % 去掉空的
    test_indices = cell2mat(test_indices);
    
    
    % === 截取测试区域 ===
    y_pred_samples_test = y_pred_samples(test_indices, :);  % [T_test × 2000]
    y_true_test = y_true(11:end);
    % === baseline 模型 ===
    y_pred_mean = mean(y_pred_samples_test, 2);
    rmse_baseline = mean(abs((y_pred_mean - y_true_test))./y_true_test);
    rmse_all(1,idx) = rmse_baseline;
    
    % === exp 模型 ===
    y_exp_mean = mean(s_exp.samples', 2);  % [T × 1]
    y_exp_test = y_exp_mean(test_indices);
    rmse_exp = mean(abs((y_exp_test - y_true_test))./y_true_test);
    rmse_all(2,idx) = rmse_exp;
    
    % === general 模型 ===
    y_gen_mean = mean(s_general.samples', 2);  % [T × 1]
    y_gen_test = y_gen_mean(test_indices);
    rmse_general = mean(abs((y_gen_test - y_true_test))./y_true_test);
    rmse_all(3,idx) = rmse_general;
    
    
    % === baseline CRPS ===
    for t = 1:length(test_indices)
        pred_dist = y_pred_samples_test(t, :);
        obs = y_true_test(t);
        crps_vals(t) = mean(abs(pred_dist - obs)/obs) - 0.5 * mean(mean(abs(pred_dist' - pred_dist)/obs));
    end
    crps_all(1, idx) = mean(crps_vals);
    
    % === exp CRPS ===
    exp_samples = s_exp.samples';  % [T × 2000]
    exp_samples_test = exp_samples(test_indices, :);
    crps_vals = zeros(length(test_indices), 1);
    for t = 1:length(test_indices)
        pred_dist = exp_samples_test(t, :);
        obs = y_true_test(t);
        crps_vals(t) = mean(abs(pred_dist - obs)/obs) - 0.5 * mean( mean(abs(pred_dist' - pred_dist)/obs));
    end
    crps_all(2, idx) = mean(crps_vals);
    
    % === general CRPS ===
    gen_samples = s_general.samples';  % [T × 2000]
    gen_samples_test = gen_samples(test_indices, :);
    crps_vals = zeros(length(test_indices), 1);
    for t = 1:length(test_indices)
        pred_dist = gen_samples_test(t, :);
        obs = y_true_test(t);
        crps_vals(t) = mean(abs(pred_dist - obs)/obs) - 0.5 * mean(mean(abs(pred_dist' - pred_dist)/obs));
    end
    crps_all(3, idx) = mean(crps_vals);
    
end

set(gcf,'unit','centimeters','position',[10 1 19 40]);

line_styles = {'--', '--', '--'};  % 所有预测方法统一虚线
marker_list_full = {'o', '^', 'h'};   % 和主图保持一致
color_map = {
    [0, 0.4470, 0.7410], ...    % Proposed neural SDE - 蓝色
    [0.3010, 0.7450, 0.9330], ...  % Iannacone - 浅蓝
    [0.8500, 0.3250, 0.0980], ...  % Zhang - 橙色
    };
font_size=14;
% 方法名称（含主方法）
method_names = {
    'Proposed neural SDE', ...
    'Allen''s SDE model for crack growth', ...
    'Zhang''s age- and state-dependent SDE model', ...
    };
figure
hold on
for m = 1:length(method_names)
    plot(sample_indices,rmse_all(m,:), ...
        'Color', color_map{m}, ...
        'LineStyle', line_styles{m}, ...
        'Marker', marker_list_full{m}, ...
        'LineWidth', 1, ...
        'MarkerSize', 6, ...
        'DisplayName', method_names{m});
end

xlabel('Unit number', 'FontName', 'Times New Roman', 'FontSize', font_size);
ylabel('NAE', 'FontName', 'Times New Roman', 'FontSize', font_size);
legend('Location', 'northwest', 'FontName', 'Times New Roman', 'FontSize', font_size);
grid on         % 打开网格
set(gca, 'GridLineStyle', '--')  % 设置网格线为虚线
set(gca, 'FontName', 'Times New Roman', 'FontSize', font_size);

ylim([0,0.11])

figure
hold on
for m = 1:length(method_names)
    plot(sample_indices, crps_all(m,:), ...
        'Color', color_map{m}, ...
        'LineStyle', line_styles{m}, ...
        'Marker', marker_list_full{m}, ...
        'LineWidth', 1, ...
        'MarkerSize', 6, ...
        'DisplayName', method_names{m});
end

xlabel('Unit number', 'FontName', 'Times New Roman', 'FontSize', font_size);
ylabel('NCRPS', 'FontName', 'Times New Roman', 'FontSize', font_size);
legend('Location', 'northwest', 'FontName', 'Times New Roman', 'FontSize', font_size);
grid on
set(gca, 'GridLineStyle', '--')
set(gca, 'FontName', 'Times New Roman', 'FontSize', font_size);
ylim([0,0.085])