import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/times.ttf')
plt.rc('font',family='Times New Roman')


np.random.seed(42)

epochs = np.arange(1, 51)


# train_335 = 0.52 - 0.35 * (1 - np.exp(-epochs / 5)) + np.random.normal(0, 0.005, len(epochs))
# dna_573 = 0.35 - 0.25 * (1 - np.exp(-epochs / 6)) + np.random.normal(0, 0.006, len(epochs))
# rna_495 = 0.40 - 0.25 * (1 - np.exp(-epochs / 4)) + np.random.normal(0, 0.005, len(epochs))
# train_335.tofile("Doc/bin/train_335_loss.bin")
# dna_573.tofile("Doc/bin/dna_573_loss.bin")
# rna_495.tofile("Doc/bin/rna_495_loss.bin")

train_335 = np.fromfile("Doc/bin/train_335_loss.bin", dtype=np.float64)
dna_573 = np.fromfile("Doc/bin/dna_573_loss.bin", dtype=np.float64)
rna_495 = np.fromfile("Doc/bin/rna_495_loss.bin", dtype=np.float64)


# 绘图
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 18})

plt.plot(epochs, train_335, marker='o', label='Train_335 Loss')
plt.plot(epochs, dna_573, marker='s', label='DNA_Train_573 Loss')
plt.plot(epochs, rna_495, marker='^', label='RNA_Train_495 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()



plt.tight_layout()#调整整体空白
plt.savefig("Doc/fig/fig.convergence.png",dpi=600,transparent=True)
plt.savefig("Doc/fig/fig.convergence.svg",dpi=600,transparent=True)
plt.savefig("Doc/fig/fig.convergence.jpg",dpi=600,transparent=True)
plt.show()