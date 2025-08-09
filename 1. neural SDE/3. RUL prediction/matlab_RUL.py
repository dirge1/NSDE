import numpy as np
import scipy.io

# 加载 .npz 文件
data = np.load("all_predictions_all_t.npz", allow_pickle=True)

threshold = 0.8
matlab_data = {}

for key in data.files:
    matlab_data[key] = data[key]

# 保存成 .mat 文件
scipy.io.savemat("all_results_full.mat", matlab_data)
print("✅ 已保存为 all_results_full.mat")
