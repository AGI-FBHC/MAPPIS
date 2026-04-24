import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/times.ttf')
plt.rc('font',family='Times New Roman')

datasets =["Test_60","DNA-Test_129","RNA-Test_117"]
# methods = ["DeepPPISP",  "MaSIF-site", "GraphBind","GraphPPIS", "RGN", "AGAT-PPIS", "Spatom", "RGCNPPIS", "MAPPIS"]
methods = ["DeepPPISP",  "GraphPPIS", "AGAT-PPIS","RGCNPPIS ", "GTE-PPIS", "ASCE-PPIS", "MEG-PPIS", "RCLG-PPIS", "MAPPIS"]
indexs = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)", "(l)", "(m)", "(n)", "(o)", "(p)", "(q)", "(r)"]
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
ylim = [50,500,1000,2000]
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
    [40.324,26.186,30.272,33.812,45.885,14.338,12.551,13.325,6.203], #Test_60
    [36.181,23.011,27.511,29.931,40.928,12.642,11.124,12.251,5.495],  # DNA-Test_129
    [29.169,17.709,21.564,24.159,30.813,9.904,8.596,9.449,4.261],#RNA-Test_117
]


RAM=[
    [206.095,251.271,313.033,337.449,334.455,359.19,366.706,370.877,122.747],#Test_60
    [184.642,233.316,272.775,307.849,292.526,316.439,334.607,339.749,108.874], # DNA-Test_129
    [140.315,181.161,217.221,241.143,233.334,247.436,255.856,265.97,85.526], #RNA-Test_117

]


VRAM=[
    [522.677,565.584,607.665,837.629,727.085,768.523,836.095,864.33,800.756], #Test_60
    [458.243,514.105,553.088,754.455,654.27,679.148,738.931,785.155,750.069], #DNA-Test_129
    [364.878,403.907,426.506,579.291,494.767,526.692,584.71,598.894,542.05], #RNA-Test_117
]

FLOPs=[
    [ 542.568, 729.563, 697.331, 746.28, 723.833, 806.826, 915.552, 1534.036, 510.475], #Test_60
    [ 505.86,636.699,642.399,674.638,655.123,742.156,834.785,1367.295,499.528], #DNA-Test_129
    [394.568,508.28,502.424,508.071,497.053,572.268,658.814,1059.305,360.54], #RNA-Test_117
]


plt.rcParams.update({'font.size': 18})
plt.subplots(4,3, figsize=(15,20))

for i in range(3):
    plt.subplot(4,3, i+1)
    plt.bar(methods, cpu_occ[i], width=bar_width, color=colors,alpha=0.5)
    plt.ylim(0, ylim[0])
    for j, value in enumerate(cpu_occ[i]):
        plt.text(j-0.1,value + 5/200*ylim[0], str(round(value,1)), ha='center')  # 添加文本，调整垂直位置
    plt.ylabel('CPU Occupancy (%)', fontsize=lable_size)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(indexs[i]+" "+datasets[i], fontsize=20)
    plt.xticks(fontsize=stick_size)
    plt.yticks(fontsize=stick_size)

for i in range(3):
    plt.subplot(4,3, 3+i+1)
    plt.bar(methods, RAM[i],  width=bar_width, color=colors,alpha=0.5)
    for j, value in enumerate(RAM[i]):
        plt.text(j-0.1,value + 5/200*ylim[1], str(round(value)), ha='center')  # 添加文本，调整垂直位置
    plt.ylim(0, ylim[1])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('RAM Usage (MB)', fontsize=lable_size)
    plt.xlabel(indexs[i+3]+" "+datasets[i], fontsize=20)
    plt.xticks(fontsize=stick_size)
    plt.yticks(fontsize=stick_size)

for i in range(3):
    plt.subplot(4,3, 6+i+1)
    plt.bar(methods, VRAM[i],  width=bar_width, color=colors,alpha=0.5)
    for j, value in enumerate(VRAM[i]):
         plt.text(j-0.1,value + 5/200*ylim[2], str(round(value)), ha='center')  # 添加文本，调整垂直位置
    plt.ylim(0, ylim[2])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('GPU VRAM Usage (MB)', fontsize=lable_size)
    plt.xlabel(indexs[i+6]+" "+datasets[i], fontsize=20)
    plt.yticks(fontsize=stick_size)
    
for i in range(3):
    plt.subplot(4,3, 9+i+1)
    plt.bar(methods, FLOPs[i],  width=bar_width, color=colors,alpha=0.5)
    for j, value in enumerate(FLOPs[i]):
        plt.text(j-0.1,value + 5/200*ylim[3], str(round(value)),ha='center')  # 添加文本，调整垂直位置
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