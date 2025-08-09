import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchsde
import numpy as np
import numpy.ma as ma

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_sde_drift_diffusion(model):
    model.eval()
    model.to(device)
    func = model.func

    # 创建时间和状态变量的网格
    t_vals = torch.linspace(0, 1, 100).to(device)  # 归一化时间 [0, 1]
    x_vals = torch.linspace(0.9, 1.7, 100).to(device)  # 状态范围

    T, X = torch.meshgrid(t_vals, x_vals, indexing='ij')
    T_flat = T.reshape(-1)
    X_flat = X.reshape(-1)

    with torch.no_grad():
        f_vals = func.f(T_flat, X_flat).reshape(T.shape).cpu().numpy()
        g_vals = func.g(T_flat, X_flat).reshape(T.shape).cpu().numpy()

    # 将归一化时间轴还原为真实时间（乘以200）
    T_real = (T * 200).cpu()
    X_cpu = X.cpu()

    # ---------- 1. 构造遮罩区域 ----------
    # 示例逻辑：在早期时间段，仅允许 x >= 0.95 + 0.4 * t（你应改为真实逻辑）
    mask = (X_cpu >= (0.95 + 0.4 * T_real / 200))

    # ---------- 2. 应用遮罩 ----------
    f_masked = ma.masked_where(~mask, f_vals)
    g_masked = ma.masked_where(~mask, g_vals)

    # ---------- 3. 绘图 ----------
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    cf1 = axs[0].contourf(X_cpu, T_real, f_masked, levels=50)
    axs[0].set_title("Drift f(t, x)")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("t (real time)")
    fig.colorbar(cf1, ax=axs[0])

    cf2 = axs[1].contourf(X_cpu, T_real, g_masked, levels=50)
    axs[1].set_title("Diffusion g(t, x)")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("t (real time)")
    fig.colorbar(cf2, ax=axs[1])

    plt.tight_layout()
    plt.show()


# ----- 模型定义 -----
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
        return torch.nn.functional.softplus(out) + 1e-4  # 保证为正

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

    # 可视化 f(t, x) 和 g(t, x)
    plot_sde_drift_diffusion(model)
