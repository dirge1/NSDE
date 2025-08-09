import numpy as np
from scipy.io import savemat
from scipy.io import loadmat

# ========== 参数 ==========
file_path = "all_predictions_t10_all_other_samples.npz"
time_index = 10
max_t = 200

# ========== crack_data 真值 ==========
mat = loadmat("crack_data.mat")
raw_data = mat['crack_data'].squeeze()  # 原始 shape: (n, 1)

# 只保留 selected_indices 中的样本
exclude_indices = {10, 18}
selected_indices = [i for i in range(len(raw_data)) if i not in exclude_indices]

# 构造字典：key 为 0 开始的编号，value 为浮点数组成的列表
crack_data = {
    i: raw_data[idx].flatten().astype(float).tolist()
    for i, idx in enumerate(sorted(selected_indices))
}

# ========== 提取数据 ==========
data = np.load(file_path)
all_samples = {}

for sample_idx in range(1, 19):  # Sample 1~20
    key_prefix = f"t{time_index}_sample_{sample_idx:02d}"
    mean = data[f"{key_prefix}_mean"]
    lower = data[f"{key_prefix}_lower"]
    upper = data[f"{key_prefix}_upper"]
    ts = data[f"{key_prefix}_ts"]
    true = data[f"{key_prefix}_true"]

    ts_full = np.linspace(0, len(true) - 1, len(true)) * 10 / max_t
    observed_ts = ts_full[:time_index] * max_t
    observed_vals = true[:time_index]

    crack_y = crack_data.get(sample_idx, [])
    crack_x = np.arange(len(crack_y)) * 0.05 * max_t if crack_y else []
    samples = data[f"{key_prefix}_samples"]

    all_samples[f"sample_{sample_idx:02d}"] = {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "samples": samples,
        "ts": ts * max_t,
        "true": true,
        "observed_ts": observed_ts,
        "observed_vals": observed_vals,
        "crack_x": crack_x,
        "crack_y": crack_y
    }

# ========== 保存为 .mat ==========
savemat("prediction_all_samples.mat", all_samples)
print("保存成功：prediction_all_samples.mat")
