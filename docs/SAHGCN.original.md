

# SAHGCN

模块化 VCA 双分支交互、图边权更新及分组消融命令见 [INTERACTION_ABLATIONS.md](INTERACTION_ABLATIONS.md)。新增模块默认关闭，可分别验证对齐、直接/门控交互、单向交互和图更新。

This is the official repository for “**Scene-Adaptive Hypergraph Convolutional Network for Hyperspectral Image Classification(SAHGCN)**”. The repo is based on Pytorch.

## Abstract

*The complex structure of land covers in remote sensing scenes* *can be considered to be composed of a mixture of clustered and striped structure communities. However, existing hyperspectral image (HSI) classification methods do not effectively utilize the community structure information. Moreover, since different community structures require different processing strategies, existing methods are typically constrained to particular structural types. In this work, we propose a scene-adaptive hypergraph convolutional network (SAHGCN), which effectively represents and utilizes the community structures contained in HSI. Specifically, the scene-adaptive hypergraph is constructed through a sampling strategy that strikes a balance between depth-first search (DFS) and breadth-first search (BFS) strategies. The striped structure can be efficiently* *mined by a second-order biased random walk strategy with higher p-values and lower q-values. In contrast, clustered structure can be efficiently* *mined by lower p-values and higher q-values. Additionally, we employ multi-kernel matrices to determine the node connections, mitigating the issue of weakly fitting high-dimensional nonlinear spectral-spatial information in manually defined distance functions. Finally, learning and inference are performed through hypergraph convolution (HGC) and multi-scale local convolution (MSC).* *Experimental results demonstrate that the proposed SHAGCN can effectively extract the community structure information contained inHSI, enhancing classification performance.*



## Getting Started

**Train(set data_set_name=="paviaU" or data_set_name="xuzhou",then run main file**

```
python main.py
```

## End-to-end learned superpixel graph

`train_vca_multidataset.py` supports two graph paths:

- `--graph-mode static` keeps the original label-guided LDA-SLIC preprocessing.
- `--graph-mode learned` skips LDA-SLIC and learns a local sparse pixel-to-token
  assignment together with a dynamic graph from the VCA-normalized features.

Example:

```powershell
python train_vca_multidataset.py --datasets paviaU --graph-mode learned --epochs 500
```

The learned path adds compactness, assignment-entropy, and token-balance
regularizers. The static mode remains the default so existing commands and
checkpoints keep their original behavior.


