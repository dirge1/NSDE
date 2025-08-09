import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ==== 设置置信度百分比 ====
confidence = 95
alpha = (100 - confidence) / 2

# ==== 读取模拟结果 ====
data = np.load("sde_simulations.npz")
ts = data["ts"]
simulations = data["simulations"]
mean = simulations.mean(axis=0)
lower = np.percentile(simulations, alpha, axis=0)
upper = np.percentile(simulations, 100 - alpha, axis=0)

print(f"✅ 模拟结果加载成功，时间步数: {len(ts)}，置信度区间: {confidence}%")

# ==== 加载真实退化样本 ====
mat_data = loadmat('crack_data.mat')
crack_data = mat_data['crack_data'].squeeze()

exclude_indices = {10, 18}
raw_samples = []

for idx, sample in enumerate(crack_data):
    if idx in exclude_indices:
        continue
    y_values = sample.flatten().tolist()
    time_values = list(range(0, len(y_values) * 10, 10))
    raw_samples.append((time_values, y_values))

max_t = 200
real_curves = [(np.array(ts_) / max_t, np.array(xs_)) for ts_, xs_ in raw_samples]

# ==== 绘图 ====
plt.figure(figsize=(12, 6))
plt.fill_between(ts * max_t, lower, upper, color='gray', alpha=0.4, label=f'{confidence}% Confidence Band')
plt.plot(ts * max_t, mean, '-', color='blue', label='Mean Prediction')

for i, (ts_real, xs_real) in enumerate(real_curves, start=1):
    plt.plot(np.array(ts_real) * max_t, xs_real, '-o', label=f'Real Sample {i}', markersize=4, alpha=0.6)

plt.xlabel("Time")
plt.ylabel("Degradation")
plt.xlim(0, 90)
plt.ylim(0.9, 1.75)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
