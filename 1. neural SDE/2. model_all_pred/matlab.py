# Python 中执行一次，生成 .mat 文件供 MATLAB 使用
import numpy as np
import scipy.io

data = np.load("sde_simulations.npz")
scipy.io.savemat("sde_simulations.mat", {
    "ts": data["ts"],
    "simulations": data["simulations"],
})
print("✅ 已保存为 sde_simulations.mat")
