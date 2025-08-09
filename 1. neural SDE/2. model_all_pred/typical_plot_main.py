import torch
import torch.nn as nn
import torchsde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 所有真实退化样本 ====
# 加载 .mat 文件
mat_data = loadmat('crack_data.mat')
crack_data = mat_data['crack_data'].squeeze()  # 转换为一维数组，元素为列向量

# 构建 raw_samples 列表，排除第 1, 4, 10, 16, 18 个样本
exclude_indices = {10, 18}
raw_samples = []

for idx, sample in enumerate(crack_data):
    if idx in exclude_indices:
        continue
    y_values = sample.flatten().tolist()
    time_values = list(range(0, len(y_values) * 10, 10))

    # 截断到时间 <= 90
    truncated_indices = [i for i, t in enumerate(time_values) if t <= 90]
    time_values = [time_values[i] for i in truncated_indices]
    y_values = [y_values[i] for i in truncated_indices]

    raw_samples.append((time_values, y_values))

max_t = 200
real_curves = [(np.array(ts) / max_t, np.array(xs)) for ts, xs in raw_samples]

# ==== 模型结构 ====
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

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
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t[:, None]
        out = self.g_net(torch.cat([x, t], dim=-1))
        return torch.nn.functional.softplus(out) + 1e-4  # softplus ensures positivity


class SDEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = SDEFunc()

    def simulate(self, ts, x0, N=1):
        ts_tensor = torch.tensor(ts, dtype=torch.float32).to(device)
        x0_tensor = torch.tensor([[x0]], dtype=torch.float32).to(device)
        results = []
        for _ in range(N):
            xs = torchsde.sdeint(self.func, x0_tensor, ts_tensor, dt=0.05, method="euler")
            results.append(xs.squeeze(-1).squeeze(-1).detach().cpu().numpy())
        return np.stack(results)  # shape [N, T]

# ==== 加载模型 ====
model = SDEModel()
model.load_state_dict(torch.load("sde_model.pth", map_location=device))
model.to(device)
model.eval()

# ==== 仿真并绘图 ====
with torch.no_grad():
    T = 101
    ts = np.linspace(0, 1, T)
    x0 = 0.9
    simulations = model.simulate(ts, x0, N=500)
    lower = np.percentile(simulations, 2.5, axis=0)
    upper = np.percentile(simulations, 97.5, axis=0)
    mean = simulations.mean(axis=0)

    # 保存均值+置信区间
    df = pd.DataFrame({
        "time": ts,
        "mean": mean,
        "lower_2.5%": lower,
        "upper_97.5%": upper
    })
    df.to_csv("sde_simulation_summary.csv", index=False)
    print("✅ 均值和置信区间已保存为 sde_simulation_summary.csv")

    # 保存所有模拟轨迹
    np.savez("sde_simulations.npz",
             ts=ts,
             simulations=simulations,
             mean=mean,
             lower=lower,
             upper=upper)
    print("✅ 全部模拟轨迹已保存为 sde_simulations.npz")

    df.to_csv("sde_simulation_summary.csv", index=False)
    print("✅ 均值和置信区间已保存为 sde_simulation_summary.csv")

    plt.figure(figsize=(12, 6))
    plt.fill_between(ts * max_t, lower, upper, color='gray', alpha=0.4, label='95% Confidence Band')
    plt.plot(ts * max_t, mean, '-', color='blue', label='Mean Prediction')

    # 真实数据，原来是按 max_t=100 归一化的，所以乘回 100
    for i, (ts_real, xs_real) in enumerate(real_curves, start=1):
        plt.plot(np.array(ts_real) * max_t, xs_real, '-o', label=f'Real Sample {i}', markersize=4, alpha=0.8)

    plt.title("SDE Simulation with 95% Confidence Interval + All Real Samples")
    plt.xlabel("Time")
    plt.ylabel("Degradation")
    plt.xlim(0, max_t)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




