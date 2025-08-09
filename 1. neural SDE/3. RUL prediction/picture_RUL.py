import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat

# ========== 参数设置 ==========
file_path = "all_predictions_all_t.npz"
time_index = 8
show_samples = False
max_t = 200


# 加载 .mat 文件
mat = loadmat("crack_data.mat")
raw_data = mat['crack_data'].squeeze()  # 原始 shape: (n, 1)

# 只保留 selected_indices 中的样本
selected_indices = {10, 18}

# 构造字典：key 为 0 开始的编号，value 为浮点数组成的列表
crack_data = {
    i: raw_data[idx].flatten().astype(float).tolist()
    for i, idx in enumerate(sorted(selected_indices))
}

# ========== 子图设置 ==========
fig, axes = plt.subplots(1, 2, figsize=(20, 14))
axes = axes.flatten()

data = np.load(file_path)

for i, sample_idx in enumerate(range(0, 2)):
    ax = axes[i]

    key_prefix = f"t{time_index}_sample_{sample_idx:02d}"
    mean = data[f"{key_prefix}_mean"]
    lower = data[f"{key_prefix}_lower"]
    upper = data[f"{key_prefix}_upper"]
    ts = data[f"{key_prefix}_ts"]
    true = data[f"{key_prefix}_true"]

    ts_full = np.linspace(0, len(true) - 1, len(true)) * 10 / max_t
    ts_obs = ts_full[:time_index]
    xs_obs = true[:time_index]

    ax.plot(ts_obs * max_t, xs_obs, 'ko-', label="Observed", markersize=3)
    ax.plot(ts * max_t, mean, 'b-', label="Pred Mean", linewidth=1)
    ax.fill_between(ts * max_t, lower, upper, color='gray', alpha=0.3)

    if sample_idx < len(crack_data):  # ✅ 修改点
        crack_y = crack_data[sample_idx]
        crack_x = np.arange(len(crack_y)) * 0.05 * max_t
        ax.plot(crack_x, crack_y, 'g--', linewidth=1.5, label="True Traj")
        ax.set_xlim(0, crack_x.max())
    else:
        ax.set_xlim(0, 120)

    ax.set_ylim(0.85, 2)
    ax.set_title(f"Sample {sample_idx}", fontsize=9)
    ax.grid(True)
    if i == 0:
        ax.legend(fontsize=7)


# # 隐藏多余子图（如果有）
# for j in range(6, len(axes)):
#     axes[j].axis("off")

plt.tight_layout()
plt.show()
