# sde_train_nll.py
import os, random, torch, torch.nn as nn, torch.optim as optim, numpy as np
from torch.utils.data import Dataset, DataLoader
import torchsde
import scipy.io
from scipy.io import loadmat

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(1107)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class MyDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ts, xs = self.samples[idx]
        return torch.tensor(ts)/max_t, torch.tensor(xs, dtype=torch.float32)

def collate_fn(batch):
    return batch  # return as list

dataset = MyDataset(raw_samples)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# ----- Model -----
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

# ----- Train -----
def train_model(model, dataloader, save_path="sde_model_nll.pth"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')  # 初始化最优损失

    for epoch in range(10000):
        model.train()
        total_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: NLL = {avg_loss:.6f}")

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

    print(f"✅ Final best model saved to {save_path}")


def compute_aic(model, dataloader):
    model.eval()
    model.to(device)
    total_nll = 0
    with torch.no_grad():
        for batch in dataloader:
            total_nll += model(batch).item()

    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_likelihood = -total_nll
    aic = 2 * k - 2 * log_likelihood

    print(f"📊 AIC = {aic:.2f} (k = {k}, logL = {log_likelihood:.2f})")
    return aic

# ----- Run -----
if __name__ == "__main__":
    model = SDEModel()
    train_model(model, loader, save_path="sde_model.pth")

    # 加载最佳模型后计算 AIC
    model.load_state_dict(torch.load("sde_model.pth"))
    compute_aic(model, loader)
