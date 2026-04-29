<div align="center">
  <h1>MAPPIS：Efficient Multi-dimension Attention Framework for Protein-Protein Interaction Site Prediction</h1>
</div>


## Introduction
<p align="justify">
Protein–Protein Interaction Site (PPIS) prediction plays a crucial role in understanding protein functions, elucidating disease mechanisms, and guiding drug discovery. Although traditional experimental methods can provide high-resolution structural information, they are costly and time-consuming, making them unsuitable for large-scale analysis. In recent years, deep learning approaches—particularly stacked Graph Neural Networks (GNNs) and attention mechanisms—have demonstrated superior performance in PPIS prediction. However, existing methods still face challenges such as fragmented multi-scale and hierarchical semantic modeling, as well as efficiency bottlenecks and representation degradation caused by structural complexity. To address these issues, we propose MAPPIS, a multi-dimensional attention-enhanced hierarchical graph neural network framework. MAPPIS jointly models intra-layer, inter-layer and layer-group attention to construct a unified multi-dimensional attention mechanism, enabling effective integration of multi-scale biochemical and structural features. In addition, a hierarchical deep graph architecture is introduced to enhance representation capability while alleviating over-smoothing, and to reduce computational complexity and memory overhead. Experimental results on multiple benchmark datasets demonstrate that MAPPIS consistently achieves state-of-the-art accuracy while exhibiting remarkable computational time and space efficiency compared to leading methods, highlighting its potential in large-scale biological applications.
</p>

<img src="Doc/fig/fig.mappis.png" width="100%"> 

## Dependency
```markdown
python                    3.10.18
dgl                       2.2.1
freesasa                  2.2.1
matplotlib                3.10.0
numpy                     2.1.2
pandas                    2.3.1
scikit-learn              1.6.1
torch                     2.3.0
torch-cluster             1.6.3
torch-geometric           2.5.0
torch-scatter             2.1.2
torch-sparse              0.6.18
torch-spline-conv         1.2.2
torchaudio                2.3.0
torchdata                 0.8.0
torchvision               0.18.0
```
### Environment and Preprocess
<p align="justify">
All experiments were repeated ten times, and the reported metrics correspond to the mean and variance computed over these ten runs. In each run, we use a different random seed to initialize the model. The datasets are divided into training and testing sets with a ratio of 8:1. All the models in the study were implemented under the same software and hardware environment. The software environment involved Ubuntu 22.04, Python 3.8, Torch 2.0 and CUDA 11.8. The hardware environment involved an Intel i7 CPU (8 cores, 3.0 GHz), 96 GB of RAM, and an RTX 3090 GPU with 24 GB of VRAM. The protein three-dimensional structures used in this study were predicted from amino acid sequences using AlphaFold 3.0, with the resulting structures saved in PDB format. Visualization was performed using PyMOL 2.0.
</p>

## Train and Test

### Train
start training
if use AttPreSite_model.py
```markdown
python AttPreSite_model.py
```
output
```markdown
./Model/fold1_best_model.pkl
./Model/fold2_best_model.pkl
...
./Model/full_model_30.pkl
```

if use AttPreSite-Ligand_model.py
```markdown
python AttPreSite-Ligand_model.py --ligand RNA --trans
```
output
```markdown
./Model/fold1_best_model.pkl
./Model/fold2_best_model.pkl
...
./Model/full_model_30.pkl
```

### Test
start testing
if use AttPreSite_model.py
```markdown
python AttPreSite_model.py
```
output
```markdown
Test_60:
Test loss:  0.35188861563801765
Test binary acc:  0.8532410225197808
Test precision: 0.5292
Test recall:  0.6375903614457832
Test f1:  0.5783606557377049
Test AUC:  0.8730095882672436
Test AUPRC:  0.5954371390159193
Test mcc:  0.49356660367384997
Threshold:  0.29

Test_315-28:
Test loss:  0.3352209431109528
Test binary acc:  0.8613687557970054
Test precision: 0.50909428359317
Test recall:  0.6404389446649544
Test f1:  0.5672629510908903
Test AUC:  0.881639916372179
Test AUPRC:  0.5846846871636862
Test mcc:  0.4905450545869246
Threshold:  0.4

BTest_31:
Test loss:  0.3097736287501551
Test binary acc:  0.8623839244938347
Test precision: 0.48627450980392156
Test recall:  0.713463751438435
Test f1:  0.5783582089552238
Test AUC:  0.8915595663497061
Test AUPRC:  0.6005335030075699
Test mcc:  0.5127453818159148
Threshold:  0.17

BTest_31-6:
Test loss:  0.29347263753414154
Test binary acc:  0.8743178717598908
Test precision: 0.5009505703422054
Test recall:  0.713125845737483
Test f1:  0.588498045784478
Test AUC:  0.8971802369715172
Test AUPRC:  0.60762077430879
Test mcc:  0.5282226538371553
Threshold:  0.17

UBtest_31-6:
Test loss:  0.3668563061952591
Test binary acc:  0.8299814094980564
Test precision: 0.36176194939081535
Test recall:  0.5428973277074542
Test f1:  0.4341957255343082
Test AUC:  0.8118437397506826
Test AUPRC:  0.4116572608571926
Test mcc:  0.3485157642515198
Threshold:  0.18
```

if use AttPreSite-Ligand_model.py
```markdown
python AttPreSite-Ligand_model.py --ligand RNA --trans
```
output
```markdown
DNA-Test_129:
Test loss:  0.17306546022205851
Test binary acc:  0.9235505797680927
Test precision: 0.41608765366114375
Test recall:  0.6950892857142857
Test f1:  0.5205616850551655
Test AUC:  0.932489375569505
Test AUPRC:  0.5233944364174145
Test mcc:  0.5006401050722077
Threshold:  0.32
```

## Visual Results
<div>
<img src="Doc/fig/fig.parameter.auprc.png" style="width: 48%; height: auto;" />
<p align="justify">Figure 1: Comparison of 5-fold CV AUPRC across hyperparameter settings for Train 335-1, DNA-Train 573, and RNA-Train 495 datasets.</p>
</div>

<div>
<img src="Doc/fig/fig.convergence.png" style="width: 32%; height: auto;" />
<p align="justify">Figure 2: Convergence Curve on Train_335, DNA_Train_573, and RNA_Train_495.</p>
</div>

<div>
<img src="Doc/fig/fig.case.png" style="width: 48%; height: auto;" />
<p align="justify">
Figure 3: Predicted interaction sites for protein 4kt3 chain A: correctly predicted interaction sites (green); interaction sites incorrectly predicted as non-sites (red); non-sites misclassified as interaction sites (yellow); correctly predicted non-interaction sites (gray).
</p>
</div>

<div>
<img src="Doc/fig/3.jpg" style="width: 40%; height: auto;" />
<p align="justify"> Figure 4: Protein 4kt3 chain A contains 138 residues, with the interaction sites highlighted in green: (a) four structural representations of the chain: surface, stick, mesh, and cartoon; (b) 3D surface views of the chain rotated at 0°, 90°, 180°, and 270°, providing a more intuitive visualization of its spatial conformation.</p>
</div>
