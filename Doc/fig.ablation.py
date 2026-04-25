import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('Doc/times/times.ttf')
plt.rc('font',family='Times New Roman')

# 生成一行三列的子图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plt.rcParams.update({'font.size': 18})


# 第一个子图是柱状图，消融，悬空的，四个柱子
x = ['Basic', '+DWG', '+HGCN', '+MA', "+HSF"]
y = np.array([0.421,0.480, 0.630, 0.840 , 0.873 ])
bottom = np.array([0.0, y[0], y[1], y[2], y[3]])
delta= y-bottom
errors= np.array([0.006, 0.008, 0.007, 0.010, 0.006])
bar_labels =[f'+{v:.3f}' for v in (delta)]
bar_labels[0] = f'{y[0]:.3f}'

bars = axs[0].bar(x,delta, color='#4e79a7',edgecolor='black', alpha=0.8, yerr=errors, capsize=5, bottom=bottom)
axs[0].bar_label(bars, labels=bar_labels, padding=3, fontsize=16)
axs[0].tick_params(axis='both', labelsize=16)
axs[0].set_ylim(0.2, 1.0)
axs[0].set_ylabel('AUROC', fontsize=16)
axs[0].set_xlabel('(a)', fontsize=18)

# 第二个子图是柱状图，消融，悬空的，三个柱子
x = ['Basic', '+Shallow', '+Middle', '+Deep']
y = np.array([0.301, 0.476, 0.560, 0.595])
bottom = np.array([0.0, y[0], y[1], y[2]])
delta= y-bottom 
errors= np.array([0.008, 0.009, 0.008, 0.006])
bar_labels =[f'+{v:.3f}' for v in (delta)]
bar_labels[0] = f'{y[0]:.3f}'

bars = axs[ 1].bar(x,delta, color='#4e79a7',edgecolor='black', alpha=0.8, yerr=errors, capsize=5, bottom=bottom)
axs[1].bar_label(bars, labels=bar_labels, padding=3, fontsize=16)
axs[1].tick_params(axis='both', labelsize=16)
axs[1].set_ylim(0, 0.8)
axs[1].set_ylabel('AUPRC', fontsize=16)
axs[1].set_xlabel('(b)', fontsize=18)

# 第三个子图
x = ['Basic','+EA', '+CA', '+LA']
y = np.array([0.351, 0.390, 0.511, 0.578])
bottom = np.array([0.0, y[0], y[1], y[2]])
delta= y-bottom
errors= np.array([0.007, 0.008, 0.012, 0.011])
bar_labels =[f'+{v:.3f}' for v in (delta)]
bar_labels[0] = f'{y[0]:.3f}'

bars = axs[2].bar(x,delta, color='#4e79a7',edgecolor='black', alpha=0.8, yerr=errors, capsize=5, bottom=bottom)
axs[2].bar_label(bars, labels=bar_labels, padding=3, fontsize=16)
axs[2].tick_params(axis='both', labelsize=16)
axs[2].set_ylim(0, 0.7)
axs[2].set_ylabel('F1', fontsize=16)
axs[2].set_xlabel('(c)', fontsize=18)


plt.tight_layout()#调整整体空白
plt.savefig("Doc/fig/fig.ablation.svg",dpi=600,transparent=True)
plt.savefig("Doc/fig/fig.ablation.png",dpi=600,transparent=True)
plt.savefig("Doc/fig/fig.ablation.jpg",dpi=600,transparent=True)
plt.show()