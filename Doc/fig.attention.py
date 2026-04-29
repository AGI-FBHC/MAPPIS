import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('Doc/times/times.ttf')
plt.rc('font',family='Times New Roman')

# 生成一行三列的子图
fig, axs = plt.subplots(1, 4, figsize=(32, 8))
plt.rcParams.update({'font.size': 18})

# 读取图片
img = np.array(Image.open("Doc/fig/fig.4kt3A.ground.true.png"))

gaze_points_ma = np.array([
                        [450,40],[500,60],
                        [250,120],[300,100],[340,110],[380,60],
                        [280,200],[300,210],[320,230],
                        [80,270],[100,200], [170,190],[180,270],
                        [180,300],
                        [180,400],[150,360],[230,400],[260,400],[290,400],[350,350],[350,400],[350,430],
                        [180,450],[250,450],[280,490],
                        [350,300],
                        [380,180],[420,210], [440,320],[420,300],[500,350],[500,300],[500,250],[530,210],
                        [600,320],[580,400],[640,400],[550,350],[650,450],
                        [450,450],[500,400],[500,450],[500,500],[500,520],
                        [360,510],
                        [420,520],
                        [580,500],
                        ])

gaze_points_ea = np.array([
                        [450,40],[500,60],
                        # [250,120],[300,100],[340,110],[380,60],
                        # [280,200],[300,210],[320,230],
                        # [80,270],[100,200], [170,190],[180,270],
                        # [180,300],
                        [180,400],[150,360],[230,400],[260,400],[290,400],[350,350],[350,400],[350,430],
                        [180,450],[250,450],[280,490],
                        [350,300],
                        # [380,180],[420,210], [440,320],[420,300],[500,350],[500,300],[500,250],[530,210],
                        # [600,320],[580,400],[640,400],[550,350],[650,450],
                        # [450,450],[500,400],[500,450],[500,500],[500,520],
                        # [360,510],
                        # [420,520],
                        # [580,500],
                        ])

gaze_points_ca = np.array([
                        [450,40],[500,60],
                        [250,120],[300,100],[340,110],[380,60],
                        [280,200],[300,210],[320,230],
                        [80,270],[100,200], [170,190],[180,270],
                        [180,300],
                        [180,400],[150,360],[230,400],[260,400],[290,400],[350,350],[350,400],[350,430],
                        [180,450],[250,450],[280,490],
                        [350,300],
                        [380,180],[420,210], [440,320],[420,300],[500,350],[500,300],[500,250],[530,210],
                        # [600,320],[580,400],[640,400],[550,350],[650,450],
                        # [450,450],[500,400],[500,450],[500,500],[500,520],
                        [360,510],
                        [420,520],
                        [580,500],
                        ])

gaze_points_la = np.array([
                        # [450,40],[500,60],
                        # [250,120],[300,100],[340,110],[380,60],
                        # [280,200],[300,210],[320,230],
                        # [80,270],[100,200], [170,190],[180,270],
                        [180,300],
                        [180,400],[150,360],[230,400],[260,400],[290,400],[350,350],[350,400],[350,430],
                        [180,450],[250,450],[280,490],
                        [350,300],
                        # [380,180],[420,210], [440,320],[420,300],[500,350],[500,300],[500,250],[530,210],
                        [600,320],[580,400],[640,400],[550,350],[650,450],
                        [450,450],[500,400],[500,450],[500,500],[500,520],
                        # [360,510],
                        # [420,520],
                        # [580,500],
                        ])
# # 创建热图矩阵
# heatmap = np.zeros((img.shape[0], img.shape[1])) 
# for x, y in gaze_points:
#     heatmap[y, x] += 1

# # 平滑热图
# heatmap = gaussian_filter(heatmap, sigma=30)

def attention_heatmap(gz_points, img, sigma=30):
    heatmap = np.zeros((img.shape[0], img.shape[1])) 
    for x, y in gz_points:
        heatmap[y, x] += 1
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    return heatmap

heatmap_MA = attention_heatmap(gaze_points_ma, img)
heatmap_EA = attention_heatmap(gaze_points_ea, img)
heatmap_CA = attention_heatmap(gaze_points_ca, img)
heatmap_LA = attention_heatmap(gaze_points_la, img)

# Multi-dimensional Attention
plt.subplot(1, 4, 1)
plt.imshow(img)
plt.imshow(heatmap_MA, cmap='jet', alpha=0.7)
plt.axis('off')

# Edge attention
plt.subplot(1, 4, 2)
plt.imshow(img)
plt.imshow(heatmap_EA, cmap='jet', alpha=0.7)
plt.axis('off') 

# Channel attention
plt.subplot(1, 4, 3)
plt.imshow(img)
plt.imshow(heatmap_CA, cmap='jet', alpha=0.7)
plt.axis('off')

# Layer attention
plt.subplot(1, 4, 4)
plt.imshow(img)
plt.imshow(heatmap_LA, cmap='jet', alpha=0.7)
plt.axis('off')

plt.tight_layout()#调整整体空白
plt.savefig("Doc/fig/fig.attention.svg",dpi=600,bbox_inches='tight',transparent=True)
plt.savefig("Doc/fig/fig.attention.png",dpi=600,bbox_inches='tight',transparent=True)
plt.savefig("Doc/fig/fig.attention.jpg",dpi=600,bbox_inches='tight',transparent=True)
plt.show()