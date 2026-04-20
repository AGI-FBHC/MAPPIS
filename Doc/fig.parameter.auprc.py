import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('Doc/times/times.ttf')
plt.rc('font',family='Times New Roman')

point_text_size=16
tick_size=18
xlabel_size=20
ylabel_size=18

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# ==========================
# 数据（保持你原始数据）
# ==========================
data = {
    "Train_335-1": {
        "node feature dim": ([64, 128, 256, 512], [0.548, 0.563, 0.556, 0.549]),
        "GCNII layers": ([2, 4, 8, 16], [0.543, 0.555, 0.562, 0.559]),
        "shallow layers": ([1,2,3], [0.557, 0.567, 0.552]),
        "middle layers": ([2, 3, 4], [0.523, 0.539, 0.530]),
        "deep layers": ([2, 3, 4,5], [0.555, 0.563, 0.554, 0.552]),
        "channel attention hops": ([6, 8, 10, 12], [0.556, 0.569, 0.556, 0.553]),
        "gamma": ([ 0.01, 0.02, 0.03, 0.04],[0.560, 0.561, 0.544, 0.527]),
        "alpha": ([0.5, 0.7, 0.9], [0.559, 0.563, 0.558]),
        "lambda": ([1, 1.5, 2], [0.555, 0.569, 0.560]),
    },
    "DNA_Train_573": {
        "node feature dim": ([64,128, 256, 512], [0.600, 0.608, 0.603, 0.590]),
        "GCNII layers": ([2, 4, 8, 16], [0.559, 0.591, 0.608, 0.608]),
        "shallow layers": ([1, 2, 3], [0.584, 0.608, 0.599]),
        "middle layers": ([2,3,4], [0.554, 0.580, 0.571]),
        "deep layers": ([2, 3, 4, 5], [0.601, 0.608, 0.606, 0.604]),
        "channel attention hops": ([6, 8, 10, 12], [0.592, 0.608, 0.601, 0.598]),
        "gamma": ([0.01, 0.02, 0.03, 0.04], [0.608, 0.617, 0.579, 0.565]),
        "alpha": ([0.5, 0.7, 0.9], [0.601, 0.608, 0.603]),
        "lambda": ([1, 1.5, 2], [0.606, 0.608, 0.604]),
    },
    "RNA_Train_495": {
        "node feature dim": ([64, 128, 256,512], [0.541, 0.544, 0.542, 0.540]),
        "GCNII layers": ([2, 4, 8, 16], [0.532, 0.539, 0.547, 0.546]),
        "shallow layers": ([1, 2, 3], [0.543, 0.544, 0.539]),
        "middle layers": ([2, 3, 4], [0.514, 0.532, 0.523]),
        "deep layers": ([2, 3, 4, 5], [0.546, 0.547, 0.544, 0.542]),
        "channel attention hops": ([6, 8, 10, 12], [0.546, 0.546, 0.544, 0.542]),
        "gamma": ([0.01, 0.02, 0.03, 0.04], [0.547, 0.554, 0.510, 0.485]),
        "alpha": ([0.5, 0.7, 0.9], [0.542, 0.551, 0.546]),
        "lambda": ([1, 1.5, 2], [0.542, 0.547, 0.542]),
    },
}


metric="auprc"
param_list = list(next(iter(data.values())).keys())
n_params = len(param_list)
nrows, ncols = 3, 3

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12))
axes = axes.flatten()

colors = ["tab:blue", "tab:orange", "tab:green"]
labels = list(data.keys())

# 第一行
# hidden dim
ax = axes[0]
for color, label in zip(colors, labels):
    x, auprc = data[label]["node feature dim"]
    y =auprc
    ax.plot(x, y, marker="o", linestyle="-", linewidth=3, markersize=8, color=color, label=label, alpha=0.5)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi+20, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 3:
                ax.text(xi-20, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi + 20, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 3:
                ax.text(xi-20, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        else:
            if i == 0:
                ax.text(xi + 20, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 3:
                ax.text(xi-20, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.axvline(x=128, linestyle='--', color='red', linewidth=1.5,alpha=0.5)
ax.set_xticks(x)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel("(a) node feature dim",fontsize=xlabel_size)
ax.set_ylabel(metric.upper(),fontsize=ylabel_size)


# gcnii layers
ax = axes[1]
for color, label in zip(colors, labels):
    x, auprc = data[label]["GCNII layers"]
    y =auprc
    ax.plot(x, y, marker="o", linestyle="-", linewidth=3, markersize=8, color=color, label=label, alpha=0.5)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi+1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 3:
                ax.text(xi-1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi+1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 3:
                ax.text(xi-1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        else:
            if i == 0:
                ax.text(xi + 1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 3:
                ax.text(xi - 1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.axvline(x=8, linestyle='--', color='red', linewidth=1.5,alpha=0.5)
ax.set_xticks(x)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel("(b) GCNII layers", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size)


# shallow layers
ax = axes[2]
for color, label in zip(colors, labels):
    x,  auprc = data[label]["shallow layers"]
    y =auprc
    ax.plot(x, y, marker="o", linestyle="-", linewidth=3, markersize=8, color=color, label=label, alpha=0.5)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi + 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 1:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 2:
                ax.text(xi - 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi + 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 2:
                ax.text(xi - 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        else:
            if i == 0:
                ax.text(xi + 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 2:
                ax.text(xi - 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
ax.axvline(x=2, linestyle='--', color='red', linewidth=1.5,alpha=0.5)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel("(c) shallow layers", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size)


#第二行

# middle layers
ax = axes[3]
for color, label in zip(colors, labels):
    x,  auprc = data[label]["middle layers"]
    y =auprc
    # 把x字符串转为数值用于绘图
    ax.plot(x, y, marker="o", linestyle="-", linewidth=3, markersize=8, color=color, label=label, alpha=0.5)
    
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                 ax.text(xi + 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 1:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom") 
            else:
                ax.text(xi - 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi + 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 1:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi-0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
        else:
            if i == 0:
                ax.text(xi + 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 1:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi -0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.axvline(x=3, linestyle='--', color='red', linewidth=1.5,alpha=0.5)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel("(d) middle layers", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size)

# deep layers
ax = axes[4]
for color, label in zip(colors, labels):
    x,  auprc = data[label]["deep layers"]
    y =auprc
    ax.plot(x, y, marker="o", linestyle="-", linewidth=3, markersize=8, color=color, label=label, alpha=0.5)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi + 0.2, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 1:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 3:
                ax.text(xi - 0.2, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi + 0.2, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 3:
                ax.text(xi - 0.2, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        else:
            if i == 0:
                ax.text(xi + 0.2, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 3:
                ax.text(xi - 0.2, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.axvline(x=3, linestyle='--', color='red', linewidth=1.5,alpha=0.5)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel("(e) deep layers", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size)

# channel attention hops
ax = axes[5]
for color, label in zip(colors, labels):
    x,  auprc = data[label]["channel attention hops"]
    y =auprc
    ax.plot(x, y, marker="o", linestyle="-", linewidth=3, markersize=8, color=color, label=label, alpha=0.5)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi+0.3,yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 3:
                ax.text(xi-0.3, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi + 0.3, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 3:
                ax.text(xi-0.3, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        else:
            if i == 0:
                ax.text(xi + 1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 3:
                ax.text(xi - 0.4, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.axvline(x=8, linestyle='--', color='red', linewidth=1.5,alpha=0.5)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel("(f) channel attention hops", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size)

#第三行



# alpha
ax = axes[6]
for color, label in zip(colors, labels):
    x,  auprc = data[label]["alpha"]
    y =auprc
    ax.plot(x, y, marker="o", linestyle="-", linewidth=3, markersize=8, color=color, label=label, alpha=0.5)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi + 0.02, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 1:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi - 0.02, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi + 0.02, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi - 0.02, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        else:
            if i == 0:
                ax.text(xi + 0.02, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi - 0.02, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.axvline(x=0.7, linestyle='--', color='red', linewidth=1.5,alpha=0.5)
ax.set_xticks(x)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel(r"(h) $\alpha$", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size  )

# lambda
ax = axes[7]
for color, label in zip(colors, labels):
    x,  auprc = data[label]["lambda"]
    y =auprc
    ax.plot(x, y, marker="o", linestyle="-", linewidth=3, markersize=8, color=color, label=label, alpha=0.5)
    
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi + 0.04, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 1:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi - 0.04, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi + 0.04, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi - 0.04, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        else:
            if i == 0:
                ax.text(xi + 0.04, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi - 0.04, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.axvline(x=1.5, linestyle='--', color='red', linewidth=1.5,alpha=0.5)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel(r"(i) $\lambda$", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size  )

# gamma
ax = axes[8]
for color, label in zip(colors, labels):
    x,  auprc = data[label]["gamma"]
    y =auprc
    ax.plot(x, y, marker="o", linestyle="-", linewidth=3, markersize=8, color=color, label=label, alpha=0.5)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi+0.002, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 3:
                ax.text(xi-0.002, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi + 0.002, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 3:
                ax.text(xi - 0.002, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
        else:
            if i == 0:
                ax.text(xi + 0.002, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 3:
                ax.text(xi - 0.002, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.axvline(x=0.02, linestyle='--', color='red', linewidth=1.5,alpha=0.5)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel(r"(g) $\gamma$", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size)


handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04),ncol=3, fontsize=point_text_size)


# 删除多余子图
for j in range(n_params, nrows*ncols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(f"Doc/fig/fig.parameter.{metric}.png", dpi=600, transparent=True, bbox_inches="tight")
plt.savefig(f"Doc/fig/fig.parameter.{metric}.svg", dpi=600, transparent=True, bbox_inches="tight")
plt.show()