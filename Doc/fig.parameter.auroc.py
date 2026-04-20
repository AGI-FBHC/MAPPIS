import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('Doc/times/times.ttf')
plt.rc('font',family='Times New Roman')

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

point_text_size=18
tick_size=18
xlabel_size=20
ylabel_size=18

# ==========================
# 数据（保持你原始数据）
# ==========================
data = {
    "Train_335-1": {
        "hidden dim": ([64, 128, 256, 512], [0.853, 0.860, 0.855, 0.850], [0.548, 0.561, 0.556, 0.549]),
        "gcnii layers": ([2, 4, 8, 16], [0.850, 0.855, 0.860, 0.859], [0.543, 0.555, 0.561, 0.559]),
        "ha layers": ([1, 2, 3], [0.852, 0.860, 0.858], [0.544, 0.561, 0.552]),
        "pseudo-pos": (["CA", "C", "SC"], [0.842, 0.851, 0.860], [0.523, 0.539, 0.561]),
        "context r": ([10, 15, 20,25], [0.858, 0.860, 0.858, 0.857], [0.555, 0.561, 0.554, 0.552]),
        "map cutoff": ([12, 14, 16, 18], [0.857, 0.860, 0.856, 0.854], [0.556, 0.561, 0.556, 0.553]),
        "dropout": ([ 0.1, 0.2, 0.3, 0.4],[0.858, 0.860, 0.855, 0.844],[0.561, 0.561, 0.544, 0.527]),
        "α": ([0.5, 0.7, 0.9], [0.859, 0.860, 0.859], [0.559, 0.561, 0.558]),
        "λ": ([1, 1.5, 2], [0.857, 0.860, 0.860], [0.555, 0.561, 0.560]),
    },
    "DNA_Train_573": {
        "hidden dim": ([64,128, 256, 512], [0.901, 0.921, 0.919,0.911], [0.600, 0.608, 0.603, 0.590]),
        "gcnii layers": ([2, 4, 8, 16], [0.902, 0.914, 0.921, 0.920], [0.559, 0.591, 0.608, 0.608]),
        "ha layers": ([1, 2, 3], [0.914, 0.921, 0.920], [0.584, 0.608, 0.599]),
        "pseudo-pos": (["CA", "C", "SC"], [0.903, 0.913, 0.921], [0.554, 0.580, 0.608]),
        "context r": ([10, 15, 20, 25], [0.918, 0.921, 0.919, 0.917], [0.601, 0.608, 0.606, 0.604]),
        "map cutoff": ([12, 14, 16, 18], [0.912, 0.921, 0.916, 0.914], [0.592, 0.608, 0.601, 0.598]),
        "dropout": ([0.1, 0.2, 0.3, 0.4], [0.921, 0.917, 0.908, 0.904], [0.608, 0.597, 0.579, 0.561]),
        "α": ([0.5, 0.7, 0.9], [0.919, 0.921, 0.920], [0.601, 0.608, 0.603]),
        "λ": ([1, 1.5, 2], [0.920, 0.921, 0.921], [0.606, 0.608, 0.604]),
    },
    "RNA_Train_495": {
        "hidden dim": ([64, 128, 256,512], [0.884, 0.885, 0.882, 0.880], [0.547, 0.544, 0.542, 0.540]),
        "gcnii layers": ([2, 4, 8, 16], [0.873, 0.878, 0.884, 0.884], [0.532, 0.539, 0.547, 0.546]),
        "ha layers": ([1, 2, 3], [0.884, 0.885, 0.881], [0.547, 0.544, 0.539]),
        "pseudo-pos": (["CA", "C", "SC"], [0.859, 0.877, 0.884], [0.514, 0.532, 0.547]),
        "context r": ([10, 15, 20,25], [0.884, 0.884, 0.882, 0.880], [0.546, 0.547, 0.544, 0.542]),
        "map cutoff": ([12, 14, 16, 18], [0.884, 0.884, 0.881, 0.882], [0.547, 0.546, 0.544, 0.542]),
        "dropout": ([0.1, 0.2, 0.3, 0.4], [0.884, 0.878, 0.866, 0.854], [0.547, 0.534, 0.510, 0.485]),
        "α": ([0.5, 0.7, 0.9], [0.882, 0.884, 0.884], [0.542, 0.547, 0.546]),
        "λ": ([1, 1.5, 2], [0.879, 0.884, 0.880], [0.542, 0.547, 0.542]),
    },
}


metric="auroc"
param_list = list(next(iter(data.values())).keys())
n_params = len(param_list)
nrows, ncols = 3, 3

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12))
axes = axes.flatten()

colors = ["tab:blue", "tab:orange", "tab:green"]
labels = list(data.keys())

# hidden dim
ax = axes[0]
for color, label in zip(colors, labels):
    x, auroc, auprc = data[label]["hidden dim"]
    y = auroc if metric == "auroc" else auprc
    ax.plot(x, y, marker="o", linestyle="-", color=color, label=label)
    for i,(xi, yi) in enumerate(zip(x, y)):
        if color == "tab:green":
            if i==1:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i==0:
                ax.text(xi+20, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        elif color == "tab:orange":
            if i == 2:
                ax.text(xi - 20, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
        else:
            if i == 0:
                ax.text(xi + 20, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.set_xlim(10,560)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel("(a) hidden dim",fontsize=xlabel_size)
ax.set_ylabel(metric.upper(),fontsize=ylabel_size)


# gcnii layers
ax = axes[1]
for color, label in zip(colors, labels):
    x, auroc, auprc = data[label]["gcnii layers"]
    y = auroc if metric == "auroc" else auprc
    ax.plot(x, y, marker="o", linestyle="-", color=color, label=label)
    for i,(xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi+0.5, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i== 2:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i==3:
                ax.text(xi-0.5, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi+0.5, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i==3:
                ax.text(xi-0.5, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        else:
            if i == 0:
                ax.text(xi+0.5, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i==3:
                ax.text(xi-0.5, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.set_xlim(1,18)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel("(b) gcnii layers", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size)

# ha layers
ax = axes[2]
for color, label in zip(colors, labels):
    x, auroc, auprc = data[label]["ha layers"]
    y = auroc if metric == "auroc" else auprc
    ax.plot(x, y, marker="o", linestyle="-", color=color, label=label)
    for i,(xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi + 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 1:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 2:
                ax.text(xi - 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi + 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 2:
                ax.text(xi - 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        else:
            if i == 0:
                ax.text(xi + 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 2:
                ax.text(xi - 0.1, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.tick_params(labelsize=tick_size)
ax.set_xlabel("(c) ha layers", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size)

# pseudo-pos
ax = axes[3]
for color, label in zip(colors, labels):
    x, auroc, auprc = data[label]["pseudo-pos"]
    y = auroc if metric == "auroc" else auprc

    x_map = {"CA": 0, "C": 1, "SC": 2}
    inv_map = {v: k for k, v in x_map.items()}  # 反向映射回标签

    # 把x字符串转为数值用于绘图
    x_numeric = [x_map[xi] for xi in x]
    ax.plot(x_numeric, y, marker="o", linestyle="-", color=color, label=label)

    for i, (xi_str, yi) in enumerate(zip(x, y)):
        xi = x_map[xi_str]  # 数值横坐标
        x_offset = 0  # 默认不偏移
        va="bottom"

        if color == "tab:orange":
            if xi_str == "CA":   # 第一个点向右偏移
                x_offset = 0.1
            elif xi_str == "SC":  # 第三个点向左偏移
                x_offset = -0.1
                va = "top"
        elif color == "tab:green":
            if xi_str == "CA":   # 第一个点向右偏移
                x_offset = 0.1
            elif xi_str == "SC":  # 第三个点向左偏移
                x_offset = -0.1
            else:
                va = "top"
        else:
            if xi_str == "CA":   # 第一个点向右偏移
                x_offset = 0.1
            elif xi_str == "SC":  # 第三个点向左偏移
                x_offset = -0.1

        ax.text(xi + x_offset, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va=va)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel("(d) pseudo-pos", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size)
ax.set_xticks(list(x_map.values()))
ax.set_xticklabels(list(x_map.keys()))

# context r
ax = axes[4]
for color, label in zip(colors, labels):
    x, auroc, auprc = data[label]["context r"]
    y = auroc if metric == "auroc" else auprc
    ax.plot(x, y, marker="o", linestyle="-", color=color, label=label)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi + 0.7, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 1:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 3:
                ax.text(xi - 0.7, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi + 0.7, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        else:
            if i == 0:
                ax.text(xi + 0.7, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.set_xlim(9,27)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel("(e) context r", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size)


# map cutoff
ax = axes[5]
for color, label in zip(colors, labels):
    x, auroc, auprc = data[label]["map cutoff"]
    y = auroc if metric == "auroc" else auprc
    ax.plot(x, y, marker="o", linestyle="-", color=color, label=label)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 1:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi + 0.5, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            elif i == 2:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 5:
                ax.text(xi - 0.5, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        else:
            ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.set_xlim(11,19)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel("(f) map cutoff", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size)


# dropout
ax = axes[6]
for color, label in zip(colors, labels):
    x, auroc, auprc = data[label]["dropout"]
    y = auroc if metric == "auroc" else auprc
    ax.plot(x, y, marker="o", linestyle="-", color=color, label=label)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        elif color == "tab:green":
            ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        else:
            if i == 0:
                ax.text(xi+0.02, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 5:
                ax.text(xi-0.02, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.set_xlim(0.05,0.45)
ax.tick_params(labelsize=tick_size)
ax.set_xlabel("(g) dropout", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size)


# α
ax = axes[7]
for color, label in zip(colors, labels):
    x, auroc, auprc = data[label]["α"]
    y = auroc if metric == "auroc" else auprc
    ax.plot(x, y, marker="o", linestyle="-", color=color, label=label)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi+0.02, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 1:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi-0.02, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
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
ax.tick_params(labelsize=tick_size)
ax.set_xlabel(r"(h) $\alpha$", fontsize=xlabel_size)
ax.set_ylabel(metric.upper(), fontsize=ylabel_size)


# λ
ax = axes[8]
for color, label in zip(colors, labels):
    x, auroc, auprc = data[label]["λ"]
    y = auroc if metric == "auroc" else auprc
    ax.plot(x, y, marker="o", linestyle="-", color=color, label=label)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if color == "tab:orange":
            if i == 0:
                ax.text(xi + 0.05, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            elif i == 1:
                ax.text(xi, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
            else:
                ax.text(xi - 0.05, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="top")
        elif color == "tab:green":
            if i == 0:
                ax.text(xi + 0.05, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi - 0.05, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
        else:
            if i == 0:
                ax.text(xi + 0.05, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
            else:
                ax.text(xi - 0.05, yi, f"{yi:.3f}", fontsize=point_text_size, ha="center", va="bottom")
ax.tick_params(labelsize=tick_size)    
ax.set_xlabel(r"(i) $\lambda$", fontsize=xlabel_size)
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