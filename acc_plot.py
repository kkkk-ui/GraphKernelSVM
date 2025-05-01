import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ログデータの読み込み
log_data = pd.read_csv("log.txt", header=0, names=["h", "c", "acc"])

# データ型を数値に変換
log_data["h"] = log_data["h"].astype(int)    # h は整数
log_data["c"] = log_data["c"].astype(float)  # C は実数
log_data["acc"] = log_data["acc"].astype(float)  # acc も実数に変換

fig, axs = plt.subplots(1, 1, figsize=(8, 5),dpi=300)  # x行x列のサブプロット

# h ごとに c ごとの分類精度の平均を計算
mean_acc = log_data.groupby(["h", "c"])["acc"].mean().reset_index()

# h ごとにプロット
for h_value in mean_acc["h"].unique():
    subset = mean_acc[mean_acc["h"] == h_value]
    axs.plot(subset["c"], subset["acc"], marker='o', linestyle='-', label=f"h={h_value}")
    break

axs.set_xscale('log')
axs.tick_params(labelsize=13)
axs.set_ylim(50,100)
axs.set_xlim(10e-4,10e2)
axs.set_xlabel("Regularization Parameter C")
axs.set_ylabel("Average Classification Accuracy (%)")
axs.set_title("Classification Accuracy")
if len(mean_acc["h"].unique()) == 1:
    print("")
else:
    axs.legend()

