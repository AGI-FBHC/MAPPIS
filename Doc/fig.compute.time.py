import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('Doc/times/times.ttf')
plt.rc('font',family='Times New Roman')

datasets =["(a) Train_355-1","(b) Test_60","(c) Test_315-28","(d) BTest_31-6","(e) UTest_31-6","(f) DNA-Train_573", "(g) DNA-Test_129", "(h) RNA-Train_495", "(i) RNA-Test_117"]
# methods = ["DeepPPISP",  "MaSIF-site", "GraphBind","GraphPPIS", "RGN", "AGAT-PPIS", "Spatom", "RGCNPPIS", "MAPPIS"]
methods = ["DeepPPISP",  "GraphPPIS", "AGAT-PPIS","RGCNPPIS", "GTE-PPIS", "ASCE-PPIS", "MEG-PPIS", "RCLG-PPIS", "MAPPIS"]
color =["blue", "skyblue", "royalblue", "lightgreen", "springgreen", "cyan", "deepskyblue", "orange", "red"]
colors = [
    '#3B5BA5',
    '#5F7DB8',
    '#8FAADC',
    '#4C9F70',
    '#7BC8A4',
    '#4CA1A3',
    '#9C8ADE',
    '#D4A72C',
    '#D9534F',  # MAPPIS
]
ylim = [340, 50, 200, 30, 30, 600, 100, 550, 80]
width=0.5
group_width=6
bar_width = 0.8
x = np.arange(0,len(datasets)*group_width,group_width)
dx = [x, x+width, x+2*width, x+3*width, x+4*width, x+5*width, x+6*width, x+7*width, x+8*width]
data=[
    [55.14, 15.46,  63.47,11.45,10.55,120.01, 32.21, 140.02,20.25], # DeepPPISP
    [140.85,20.85,  98.35,12.07,12.34,280.65, 58.45, 297.25,39.42], # MaSIF-site
    [154.44,21.75,  104.58,13.02,14.24,311.45, 50.10,314.78,43.75], # GraphBind
    [201.45,27.46, 120.45,17.51,17.08,400.25, 60.65, 328.86,46.71], # GraphPPIS
    [170.35,25.27, 104.11,13.24,13.40,339.45, 52.85, 284.96,40.25], # RGN
    [284.23,38.12, 143.96,21.08,21.19,512.29, 74.12, 462.14,67.42], # AGAT-PPIS
    [211.92,30.79, 122.07,18.76,18.34,422.23, 61.47, 322.78,50.88], # Spatom
    [199.72,29.50, 115.05,15.34,15.21,402.73, 55.22, 301.58,47.36], # RGCNPPIS
    [110.17,17.15,  90.76, 6.93, 6.97,150.15, 37.94, 148.08,27.68],# MAPPIS
]
errs=[
    [3.14, 1.46,  6.47, 0.45, 1.55, 10.01, 3.21, 8.02, 1.25], # DeepPPISP
    [5.85, 1.85,  4.35, 0.87, 1.34, 15.65, 1.45, 5.25, 1.42], # MaSIF-site
    [5.44, 1.75,  4.58, 1.02, 1.24, 10.45, 2.10, 8.78, 1.75], # GraphBind
    [3.45, 1.46,  5.45, 0.51, 1.08, 9.25, 3.65, 10.86, 2.71], # GraphPPIS
    [4.35, 1.27,  4.11, 0.24, 1.40, 9.45, 2.85, 8.96, 2.25], # RGN
    [5.23, 2.12,  3.96, 1.08, 1.19, 10.29, 2.12, 8.14, 2.42], # AGAT-PPIS
    [7.92, 2.79,  2.07, 0.76, 1.34, 12.23, 1.47, 8.78, 2.88], # Spatom
    [7.72, 2.50,  5.05, 0.34, 1.21, 12.73, 1.22, 11.58, 1.36], # RGCNPPIS
    [8.17, 2.15,  2.76, 0.93, 1.07, 12.15, 3.94, 10.08, 2.68],# MAPPIS
]
data = np.array(data)
errs= np.array(errs)
error_attri = dict(elinewidth = 1, ecolor="black", capsize = 3)

plt.rcParams.update({'font.size': 18})
plt.subplots(3,3, figsize=(15,15), dpi=600)
for i in range(9):
    plt.subplot(3, 3, i+1)
    bar_labels =[f'{v:.0f}' for v in (data[:,i])]
    bars = plt.bar(methods, data[:,i], width=bar_width,color=colors,alpha=0.5)
    plt.bar_label(bars, labels=bar_labels, padding=3, fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Time (s)', fontsize=18)
    plt.ylim(0,ylim[i])
    plt.xlabel(datasets[i], fontsize=20)
    


plt.tight_layout()#调整整体空白
plt.savefig("Doc/fig/fig.compute.time.svg", dpi=600,transparent=True)
plt.savefig("Doc/fig/fig.compute.time.jpg", dpi=600,transparent=True)
plt.savefig("Doc/fig/fig.compute.time.png", dpi=600,transparent=True)
plt.show()