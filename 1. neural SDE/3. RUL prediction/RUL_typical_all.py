import numpy as np
import torch
import torchsde
import torch.nn as nn
import os
import scipy.io
from scipy.io import loadmat

# ========= 模型定义（保持不变） ==========
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class SDEFunc(torchsde.SDEIto):
    def __init__(self):
        super().__init__(noise_type="diagonal")
        self.f_net = MLP(2, 1)
        self.g_net = MLP(2, 1)

    def f(self, t, x):
        if x.dim() == 1: x = x[:, None]
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t[:, None]
        out = self.f_net(torch.cat([x, t], dim=-1))
        return torch.nn.functional.softplus(out) + 1e-4  # softplus ensures positivity

    def g(self, t, x):
        if x.dim() == 1: x = x[:, None]
        if t.dim() == 0: t = t.expand(x.shape[0], 1)
        elif t.dim() == 1: t = t[:, None]
        out = self.g_net(torch.cat([x, t], dim=-1))
        return torch.nn.functional.softplus(out) + 1e-4

class SDEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = SDEFunc()

# ========= 加载模型 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SDEModel()
model.load_state_dict(torch.load("sde_model.pth", map_location=device))
model.to(device)
model.eval()

# ========= 原始轨迹 ==========
# 加载 .mat 文件
mat_data = loadmat('crack_data.mat')
crack_data = mat_data['crack_data'].squeeze()  # 转换为一维数组，元素为列向量

# 仅保留除第 10, 18 个样本以外的所有样本
exclude_indices = {10, 18}
select_indices = [i for i in range(len(crack_data)) if i not in exclude_indices]
raw_samples = []

for idx, sample in enumerate(crack_data):
    if idx in select_indices:
        y_values = sample.flatten().tolist()
        time_values = list(range(0, len(y_values) * 10, 10))
        raw_samples.append((time_values, y_values))

# 固定 time_index 为 10
time_index = 10
max_t = 200
N = 500
dt = 0.01
all_results = {}

# 归一化的真实轨迹
real_curves = [(np.array(ts) / max_t, np.array(xs)) for ts, xs in raw_samples]

for idx, (t_full, x_full) in enumerate(real_curves):
    print(f"\n🔄 正在处理第 {idx} 条轨迹...")

    ts_obs = t_full[:time_index]
    xs_obs = x_full[:time_index]

    x0 = xs_obs[-1]
    t0 = ts_obs[-1]

    ts_future = np.arange(t0, 1, step=dt)

    x0_tensor = torch.tensor([[x0]], dtype=torch.float32).to(device)
    ts_tensor = torch.tensor(ts_future, dtype=torch.float32).to(device)

    samples = []
    with torch.no_grad():
        for _ in range(N):
            xs_sim = torchsde.sdeint(model.func, x0_tensor, ts_tensor, dt=0.01, method="euler")
            samples.append(xs_sim.squeeze().cpu().numpy())

    samples = np.stack(samples)
    mean = samples.mean(axis=0)
    lower = np.percentile(samples, 2.5, axis=0)
    upper = np.percentile(samples, 97.5, axis=0)

    key_prefix = f"t{time_index}_sample_{idx:02d}"
    all_results[f"{key_prefix}_mean"] = mean
    all_results[f"{key_prefix}_lower"] = lower
    all_results[f"{key_prefix}_upper"] = upper
    all_results[f"{key_prefix}_ts"] = ts_future
    all_results[f"{key_prefix}_true"] = x_full
    all_results[f"{key_prefix}_samples"] = samples

# 保存所有结果
output_file = "all_predictions_t10_all_other_samples.npz"
np.savez_compressed(output_file, **all_results)
print(f"\n✅ 所有结果已保存到 {output_file}")

