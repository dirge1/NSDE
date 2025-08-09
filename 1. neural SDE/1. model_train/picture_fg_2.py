import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torchsde
import numpy as np
from scipy.interpolate import interp1d
import scipy.io as sio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 每个样本的 crack 长度随时间变化
crack_data = {
    0:  [0.90, 0.95, 1.00, 1.05, 1.12, 1.19, 1.27, 1.35, 1.48, 1.64],
    1:  [0.90, 0.94, 0.98, 1.03, 1.08, 1.14, 1.21, 1.28, 1.37, 1.47, 1.60],
    2:  [0.90, 0.94, 0.98, 1.03, 1.08, 1.13, 1.19, 1.26, 1.35, 1.46, 1.58, 1.77],
    3:  [0.90, 0.94, 0.98, 1.03, 1.07, 1.12, 1.19, 1.25, 1.34, 1.43, 1.55, 1.73],
    4:  [0.90, 0.94, 0.98, 1.03, 1.07, 1.12, 1.19, 1.24, 1.34, 1.43, 1.55, 1.71],
    5:  [0.90, 0.94, 0.98, 1.03, 1.07, 1.12, 1.18, 1.23, 1.33, 1.41, 1.51, 1.68],
    6:  [0.90, 0.94, 0.98, 1.02, 1.07, 1.11, 1.17, 1.23, 1.32, 1.41, 1.52, 1.66],
    7:  [0.90, 0.93, 0.97, 1.00, 1.06, 1.11, 1.17, 1.23, 1.30, 1.39, 1.49, 1.62],
    8:  [0.90, 0.92, 0.97, 1.01, 1.05, 1.09, 1.15, 1.21, 1.28, 1.36, 1.44, 1.55, 1.72],
    9:  [0.90, 0.92, 0.96, 1.00, 1.04, 1.08, 1.13, 1.19, 1.26, 1.34, 1.42, 1.52, 1.67],
    10: [0.90, 0.93, 0.96, 1.00, 1.04, 1.08, 1.13, 1.18, 1.24, 1.31, 1.39, 1.49, 1.65],
    11: [0.90, 0.93, 0.97, 1.00, 1.03, 1.07, 1.10, 1.16, 1.22, 1.29, 1.37, 1.48, 1.64],
    12: [0.90, 0.92, 0.97, 0.99, 1.03, 1.06, 1.10, 1.14, 1.20, 1.26, 1.31, 1.40, 1.52],
    13: [0.90, 0.93, 0.96, 1.00, 1.03, 1.07, 1.12, 1.16, 1.20, 1.26, 1.30, 1.37, 1.45],
    14: [0.90, 0.92, 0.96, 0.99, 1.03, 1.06, 1.10, 1.16, 1.21, 1.27, 1.33, 1.40, 1.49],
    15: [0.90, 0.92, 0.95, 0.97, 1.00, 1.03, 1.07, 1.11, 1.16, 1.22, 1.26, 1.33, 1.40],
    16: [0.90, 0.93, 0.96, 0.97, 1.00, 1.05, 1.08, 1.14, 1.16, 1.20, 1.24, 1.32, 1.38],
    17: [0.90, 0.92, 0.94, 0.97, 1.01, 1.04, 1.07, 1.09, 1.14, 1.19, 1.23, 1.28, 1.35],
    18: [0.90, 0.92, 0.94, 0.97, 0.99, 1.02, 1.05, 1.08, 1.12, 1.16, 1.20, 1.25, 1.31],
    19: [0.90, 0.92, 0.94, 0.97, 0.99, 1.02, 1.05, 1.08, 1.12, 1.16, 1.19, 1.24, 1.29],
    20: [0.90, 0.92, 0.94, 0.97, 0.99, 1.02, 1.04, 1.07, 1.11, 1.14, 1.18, 1.22, 1.27],
}

def build_mask_function_from_data(crack_data):
    num_points = max(len(v) for v in crack_data.values())
    t_array = np.arange(num_points) * 10

    x_matrix = []
    for i in range(num_points):
        xi = [v[i] for v in crack_data.values() if len(v) > i]
        x_matrix.append(xi)

    x_matrix = np.array(x_matrix, dtype=object)
    x_mins = np.array([min(xs) for xs in x_matrix])
    x_maxs = np.array([max(xs) for xs in x_matrix])

    f_min = interp1d(t_array, x_mins, kind='linear', fill_value='extrapolate')
    f_max = interp1d(t_array, x_maxs, kind='linear', fill_value='extrapolate')

    return lambda t, x: (x >= f_min(t)) & (x <= f_max(t))

def plot_sde_drift_diffusion(model):
    model.eval()
    model.to(device)
    func = model.func

    t_real_max = 150
    t_vals = torch.linspace(0, t_real_max / 200, 1000).to(device)
    x_vals = torch.linspace(0.85, 1.8, 1000).to(device)

    T, X = torch.meshgrid(t_vals, x_vals, indexing='ij')
    T_flat = T.reshape(-1)
    X_flat = X.reshape(-1)

    with torch.no_grad():
        f_vals = func.f(T_flat, X_flat).reshape(T.shape).cpu().numpy()
        g_vals = func.g(T_flat, X_flat).reshape(T.shape).cpu().numpy()

    T_real = (T * 200).cpu().numpy()
    X_np = X.cpu().numpy()

    # 原始 mask
    mask_func = build_mask_function_from_data(crack_data)
    mask = mask_func(T_real, X_np)

    # === 提取 sample 0 和 sample 20 的边界曲线 ===
    sample_ids = [0, 5, 10, 15, 20]
    sample_data = {}
    boundary_points = []
    for key in sample_ids:
        x_seq = crack_data[key]
        t_seq = np.arange(len(x_seq)) * 10
        x_tensor = torch.tensor(x_seq, dtype=torch.float32).to(device)
        t_tensor = torch.tensor(t_seq / 200, dtype=torch.float32).to(device)
        with torch.no_grad():
            f_seq = func.f(t_tensor, x_tensor).squeeze().cpu().numpy()
            g_seq = func.g(t_tensor, x_tensor).squeeze().cpu().numpy()
        sample_data[f'f_sample_{key}'] = f_seq
        sample_data[f'g_sample_{key}'] = g_seq
        sample_data[f'x_sample_{key}'] = np.array(x_seq)
        sample_data[f't_sample_{key}'] = t_seq

        if key in [0, 20]:
            boundary_points.append((t_seq, x_seq, f_seq))

    # 构建平滑边界曲面（上边界包络面）
    t_all = np.concatenate([boundary_points[0][0], boundary_points[1][0]])
    x_all = np.concatenate([boundary_points[0][1], boundary_points[1][1]])
    f_all = np.concatenate([boundary_points[0][2], boundary_points[1][2]])

    from scipy.interpolate import griddata
    upper_surface = griddata(
        points=(t_all, x_all),
        values=f_all,
        xi=(T_real, X_np),
        method='cubic'
    )

    # 构建截断：mask & 小于边界面
    smooth_mask = (f_vals <= upper_surface)
    final_mask = mask & smooth_mask

    f_vals_masked = np.where(final_mask, f_vals, np.nan)
    g_vals_masked = np.where(final_mask, g_vals, np.nan)

    # 保存所有数据
    sio.savemat("sde_surface_data.mat", {
        'T': T_real,
        'X': X_np,
        'f': f_vals,
        'g': g_vals,
        'f_masked': f_vals_masked,
        'g_masked': g_vals_masked,
        'mask': final_mask.astype(np.uint8),
        **sample_data,
    })


# ===== 模型定义 =====
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
        if x.dim() == 1:
            x = x[:, None]
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t[:, None]
        out = self.f_net(torch.cat([x, t], dim=-1))
        return torch.nn.functional.softplus(out) + 1e-4

    def g(self, t, x):
        if x.dim() == 1:
            x = x[:, None]
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t[:, None]
        out = self.g_net(torch.cat([x, t], dim=-1))
        return torch.nn.functional.softplus(out) + 1e-4

class SDEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = SDEFunc()

    def forward(self, batch):
        total_nll = 0
        for ts, xs in batch:
            ts, xs = ts.to(device), xs.to(device)
            dt = ts[1:] - ts[:-1]
            dx = xs[1:] - xs[:-1]
            x_prev = xs[:-1]
            t_prev = ts[:-1]

            mu = self.func.f(t_prev, x_prev).squeeze(-1) * dt
            std = self.func.g(t_prev, x_prev).squeeze(-1) * dt.sqrt() + 1e-6

            dist = torch.distributions.Normal(mu, std)
            log_prob = dist.log_prob(dx)
            total_nll += -log_prob.sum()
        return total_nll / len(batch)

if __name__ == "__main__":
    model = SDEModel()
    model.load_state_dict(torch.load("sde_model.pth", map_location=device))
    plot_sde_drift_diffusion(model)
