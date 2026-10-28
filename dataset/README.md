# 数据文件布局

本仓库不包含原始数据或标签。请自行获取相应数据集，将 MATLAB 文件放在以下位置；文件内变量名必须与表中一致。

| 训练名称 | 相对 `dataset/` 的文件 | MATLAB 变量名 |
| --- | --- | --- |
| `paviaU` | `PaviaU/PaviaU.mat`、`PaviaU/PaviaU_gt.mat` | `paviaU`、`paviaU_gt` |
| `xuzhou` | `Xuzhou/xuzhou.mat`、`Xuzhou/xuzhou_gt.mat` | `xuzhou`、`xuzhou_gt` |
| `whu-honghu` | `WHU-Hi/WHU-Hi-HongHu/WHU_Hi_HongHu.mat`、`WHU-Hi/WHU-Hi-HongHu/WHU_Hi_HongHu_gt.mat` | `WHU_Hi_HongHu`、`WHU_Hi_HongHu_gt` |
| `WHUL` | `WHU-Hi/WHU-Hi-LongKou/WHU_Hi_LongKou.mat`、`WHU-Hi/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat` | `WHU_Hi_LongKou`、`WHU_Hi_LongKou_gt` |
| `whu-hanchuan` | `WHU-Hi/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat`、`WHU-Hi/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat` | `WHU_Hi_HanChuan`、`WHU_Hi_HanChuan_gt` |
| `indian` | `Indian/Indian_pines_corrected.mat`、`Indian/Indian_pines_gt.mat` | `indian_pines_corrected`、`indian_pines_gt` |

图像形状应为 `H × W × B`，标签为 `H × W`，背景标记为 0。读取和波长元数据定义见 `loadData/data_read.py`。

`WHU` 是历史兼容名称，会裁剪 HongHu 并修改标签；完整 HongHu 使用 `whu-honghu`。其他数据集分支可能仍指向历史本地路径，使用前需按自身文件位置修改。
