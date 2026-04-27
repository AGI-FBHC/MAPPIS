import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/times.ttf')
plt.rc('font',family='Times New Roman')

datasets =["Test_60","DNA-Test_129","RNA-Test_117"]
methods = [ "AGAT-PPIS","RGCNPPIS ", "GTE-PPIS", "ASCE-PPIS", "MEG-PPIS", "RCLG-PPIS", "MAPPIS"]
indexs = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)", "(l)", "(m)", "(n)", "(o)", "(p)", "(q)", "(r)"]
color =["blue", "skyblue", "royalblue", "lightgreen", "springgreen", "cyan", "deepskyblue", "orange", "red"]
colors = [
    '#8FAADC',
    '#4C9F70',
    '#7BC8A4',
    '#4CA1A3',
    '#9C8ADE',
    '#D4A72C',
    '#D9534F',  # MAPPIS
]
ylim = [60,500,1000,2000]
x = np.arange(0,20,1)
bar_width=0.4
width=0.5
group_width=6
bar_width = 0.8
title_size = 48
lable_size = 18
stick_size=18
x = np.arange(0,len(datasets)*group_width,group_width)
dx = [x, x+width, x+2*width, x+3*width, x+4*width, x+5*width, x+6*width, x+7*width, x+8*width]

cpu_occ=[
    [30.272,33.812,45.885,14.338,12.551,13.325,6.203], #Test_60
    [27.511,29.931,40.928,12.642,11.124,12.251,5.495],  # DNA-Test_129
    [21.564,24.159,30.813,9.904,8.596,9.449,4.261],#RNA-Test_117
]


RAM=[
    [313.033,337.449,334.455,359.19,366.706,370.877,122.747],#Test_60
    [272.775,307.849,292.526,316.439,334.607,339.749,108.874], # DNA-Test_129
    [217.221,241.143,233.334,247.436,255.856,265.97,85.526], #RNA-Test_117

]


VRAM=[
    [807.665,837.629,840.085,868.523,836.095,864.33,800.756], #Test_60
    [753.088,756.455,759.27,789.148,763.931,795.155,750.069], #DNA-Test_129
    [546.506,551.291,560.767,569.692,560.71,578.894,542.05], #RNA-Test_117
]

FLOPs=[
    [ 697.331, 746.28, 723.833, 806.826, 915.552, 1534.036, 510.475], #Test_60
    [ 642.399,674.638,655.123,742.156,834.785,1367.295,499.528], #DNA-Test_129
    [502.424,508.071,497.053,572.268,658.814,1059.305,360.54], #RNA-Test_117
]


plt.rcParams.update({'font.size': 18})
plt.subplots(4,3, figsize=(15,20))

for i in range(3):
    plt.subplot(4,3, i+1)
    bars = plt.bar(methods, cpu_occ[i], width=bar_width, color=colors,alpha=0.5)
    bar_labels =[f'{v:.1f}' for v in (cpu_occ[i])]
    plt.bar_label(bars, labels=bar_labels, padding=3, fontsize=18)
    plt.axhline(y=cpu_occ[i][-1], color='red', linestyle='--', linewidth=1)  # 添加竖线分隔
    plt.ylim(0, ylim[0])    
    plt.ylabel('CPU Occupancy (%)', fontsize=lable_size)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(indexs[i]+" "+datasets[i], fontsize=20)
    plt.xticks(fontsize=stick_size)
    plt.yticks(fontsize=stick_size)

for i in range(3):
    plt.subplot(4,3, 3+i+1)
    bars = plt.bar(methods, RAM[i],  width=bar_width, color=colors,alpha=0.5)
    bar_labels =[f'{v:.0f}' for v in (RAM[i])]
    plt.bar_label(bars, labels=bar_labels, padding=3, fontsize=18)
    plt.axhline(y=RAM[i][-1], color='red', linestyle='--', linewidth=1)  # 添加竖线分隔
    plt.ylim(0, ylim[1])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('RAM Usage (MB)', fontsize=lable_size)
    plt.xlabel(indexs[i+3]+" "+datasets[i], fontsize=20)
    plt.xticks(fontsize=stick_size)
    plt.yticks(fontsize=stick_size)

for i in range(3):
    plt.subplot(4,3, 6+i+1)
    bars = plt.bar(methods, VRAM[i],  width=bar_width, color=colors,alpha=0.5)
    bar_labels =[f'{v:.0f}' for v in (VRAM[i])]
    plt.bar_label(bars, labels=bar_labels, padding=3, fontsize=18)
    plt.axhline(y=VRAM[i][-1], color='red', linestyle='--', linewidth=1)  # 添加竖线分隔
    plt.ylim(400, ylim[2])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('GPU VRAM Usage (MB)', fontsize=lable_size)
    plt.xlabel(indexs[i+6]+" "+datasets[i], fontsize=20)
    plt.yticks(fontsize=stick_size)
    
for i in range(3):
    plt.subplot(4,3, 9+i+1)
    bars = plt.bar(methods, FLOPs[i],  width=bar_width, color=colors,alpha=0.5)
    bar_labels =[f'{v:.0f}' for v in (FLOPs[i])]
    plt.bar_label(bars, labels=bar_labels, padding=3, fontsize=18)
    plt.axhline(y=FLOPs[i][-1], color='red', linestyle='--', linewidth=1)  # 添加竖线分隔
    plt.ylim(0, ylim[3])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('FLOPs (MFLOPs)', fontsize=lable_size)
    plt.xlabel(indexs[i+9]+" "+datasets[i], fontsize=20)
    plt.xticks(fontsize=stick_size)
    plt.yticks(fontsize=stick_size)



plt.tight_layout()#调整整体空白
plt.savefig("Doc/fig/fig.compute.space.png",dpi=600,transparent=True)
plt.savefig("Doc/fig/fig.compute.space.svg",dpi=600,transparent=True)
plt.savefig("Doc/fig/fig.compute.space.jpg",dpi=600,transparent=True)
plt.show()