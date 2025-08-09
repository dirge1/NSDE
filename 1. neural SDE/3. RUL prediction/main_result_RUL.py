import numpy as np

# ==== 参数设置 ====
file_path = "all_predictions_all_t4.npz"
time_index = 13
max_t = 5000

# ==== Ground Truth（单位：秒）====
true_times_by_threshold = {
    9: {
        0: 3285.0877, 1: 3810.8108, 5: 3180.5556, 9: 3028.2258,
    },

    6: {
        2: 3484.8485, 3: 3815.7895, 4: 3226.1905, 6: 3699.0741,
        7: 3860.4651, 8: 3071.4286, 10: 3157.4074, 11: 3037.0370,
        13: 3407.8947, 14: 3625.0000,
    },


}

# ==== 加载预测数据 ====
data = np.load(file_path)
results = []

for threshold, samples_dict in true_times_by_threshold.items():
    for sample_idx, true_time_s in samples_dict.items():
        key_prefix = f"t{time_index}_sample_{sample_idx:02d}"
        mean = data[f"{key_prefix}_mean"]
        ts = data[f"{key_prefix}_ts"] * max_t
        samples = data[f"{key_prefix}_samples"]

        # 1. 插值预测时间
        pred_time = None
        for i in range(1, len(mean)):
            if mean[i - 1] < threshold <= mean[i]:
                x0, x1 = mean[i - 1], mean[i]
                t0, t1 = ts[i - 1], ts[i]
                pred_time = t0 + (threshold - x0) * (t1 - t0) / (x1 - x0)
                break

        # 2. CRPS计算
        threshold_times = []
        for curve in samples:
            for j in range(1, len(curve)):
                if curve[j - 1] < threshold <= curve[j]:
                    y0, y1 = curve[j - 1], curve[j]
                    t0, t1 = ts[j - 1], ts[j]
                    t_interp = t0 + (threshold - y0) * (t1 - t0) / (y1 - y0)
                    threshold_times.append(t_interp)
                    break
            else:
                threshold_times.append(ts[-1])

        threshold_times = np.array(threshold_times)
        true_time_ms = true_time_s

        term1 = np.mean(np.abs(threshold_times - true_time_ms))
        term2 = 0.5 * np.mean(np.abs(threshold_times[:, None] - threshold_times[None, :]))
        crps = term1 - term2

        results.append({
            "Sample Index": sample_idx,
            "Threshold": threshold,
            "Predicted Time": round(pred_time, 4) if pred_time else None,
            "True Time": round(true_time_ms, 4),
            "Error": round(abs(pred_time - true_time_ms), 4) if pred_time else None,
            "CRPS": round(crps, 4),
        })

# ==== 打印结果 ====
print(f"{'Sample':>6} | {'Thresh':>6} | {'Pred(ms)':>10} | {'True(ms)':>10} | {'Error':>8} | {'CRPS':>7}")
print("-" * 60)
for r in results:
    if r["Predicted Time"] is not None:
        print(f"{r['Sample Index']:>6} | {r['Threshold']:>6.1f} | {r['Predicted Time']:>10.4f} | {r['True Time']:>10.4f} | {r['Error']:>+8.4f} | {r['CRPS']:>7.4f}")
    else:
        print(f"{r['Sample Index']:>6} | {r['Threshold']:>6.1f} | {'Not reached':>10} | {r['True Time']:>10.4f} | {'   N/A':>8} | {r['CRPS']:>7.4f}")

import pandas as pd

# ==== 保存为CSV ====
df = pd.DataFrame(results)
df.to_csv("prediction_results.csv", index=False)
print("\n结果已保存为 'prediction_results.csv'")

