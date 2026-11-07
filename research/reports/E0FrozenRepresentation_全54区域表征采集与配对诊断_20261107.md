# E0-R全54冻结区域表征采集与配对诊断

Issue243 / PR244。对应研究主清单§12.5和实施计划§13、§16.1的冻结表征诊断，承接239的接受边界和241的负结果复盘。对Issue184的54个目标模型各执行一次冻结前向，在相同Q区域上比较CNN和图分支的两个阶段。

## 实现与验收

`scripts/export_frozen_representation_diagnostics.py --all` 先认证完整54网格和分析版本，再逐例复用原阶段观察器。每例统计写入独立JSON，完整回执在所有输入和输出复核通过后生成。每景用种子64抽取1024个区域，匹配组使用同一观察、区域索引及权重。保存协方差谱、参与率有效维、PCA分量和分支相关性，采集阶段张量只用于内存计算。原单例参数和文件格式继续由测试覆盖。

授权LXC checkout `/home/lbx/hsi-pioneer271828/checkouts/issue243-representation01`，分支 `codex/e0-full54-representation`，实际测试提交 `f090ca302192a065de852c43e07e4f3beaeea6b4`。Python 3.10.20，Torch 2.7.1+cu128。GPU验证和冻结前向均用物理GPU1，启动时空闲21748MiB，利用率分别52%和54%。

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES= REPRESENTATION_CAPTURE_DEVICE=cpu python -m unittest discover -s tests -p 'test_*representation*.py' -v
python scripts/check_repository.py --expected-commit f090ca302192a065de852c43e07e4f3beaeea6b4
CUDA_VISIBLE_DEVICES=1 REPRESENTATION_CAPTURE_DEVICE=cuda:0 python -m unittest discover -s tests -p test_frozen_representation_capture.py -v
CUDA_VISIBLE_DEVICES=1 python scripts/export_frozen_representation_diagnostics.py \
  --expected-commit f090ca302192a065de852c43e07e4f3beaeea6b4 \
  --freeze "$RUNS/g4-matched-target-issue184-frozen01/freeze.json" \
  --freeze-sha256 7dd7e87334766d74005d14345e994fc2e700e109f6d8141b462004831496b48c \
  --analysis "$RUNS/g4-matched-target-issue184-analysis01/receipt.json" \
  --analysis-sha256 9b9d2e1390fa54915cc34bac87e6826cc5cb138cc746ae1c5012bead8fec441b \
  --all --out "$RUNS/g4-e0-issue243-full54-representation01" --device cuda:0 \
  --maximum-regions 1024 --seed 64 --variance-floor 1e-24
```

`python` 为 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，`RUNS` 为同根目录的 `runs`。

| 项目 | 实际结果 |
|---|---|
| CPU表征专项 | 11/11，36.190秒 |
| GPU冻结采集专项 | 5/5，35.123秒 |
| 仓库检查 | 407个Python文件、128个JSON文件通过 |
| 实际54模型 | 54次前向，530.708秒，初次身份认证288.257秒，14205个保护文件前后核对通过 |
| 独立复核 | 54个摘要哈希、108个权重与侧车文件哈希、216个由已存特征值重算的参与率通过 |

合成全批次用每景一个真实构造的合成权重覆盖54个结构角色，验证批次范围、同样本、已存在输出拒绝和最后一个权重侧车篡改时的前置拒绝。GPU合成仍覆盖旧入口真实六角色的原始阶段与精确预测保护。全量模型回归未重跑。实际全54优化步、BN校准更新、GT解码和新增query释放均为0。独立复核的数值范围为保存的特征值及汇总，原始特征的重新推理和SVD未纳入独立复核。真实前向日志保留PyTorch的稀疏CSR测试阶段警告。

## 终态配对结果

以下是残差模式第二阶段的三模型种子均值，左侧为目标从头训练，右侧为完整源全量微调。

| 目标 | CNN有效维 | 图分支有效维 | 分支平均绝对相关 |
|---|---:|---:|---:|
| PaviaU | 5.959 → 6.346 | 5.890 → 6.025 | 0.219 → 0.244 |
| Xuzhou | 5.143 → 5.834 | 5.181 → 5.360 | 0.249 → 0.285 |
| HongHu | 6.162 → 6.816 | 8.668 → 7.579 | 0.212 → 0.234 |

HongHu残差全量微调终态的图分支有效维下降，CNN有效维及跨分支相关上升。其他两景残差组图分支有效维均上升。图分支变化具有场景差异。这里比较同一支持种子下的两个训练终态，整幅目标场景只读参与前向。源初始化阶段的特征、独立空间块、功能冗余和裁剪收益需要分别验证，不能仅凭终态协方差推断遗忘或决定缩宽。

## 下一模块的实测准备

用Issue188的K10/V5上下文逐景对比Issue184冻结输入，三个场景的有序标签及索引张量、完整观测、图与家族身份均一致，28.793秒。仅头与VCA加三头共54个角色均与各自对应的Issue184全量微调初态一致，89个保护文件通过，8.720秒。仅头可训练参数量为九类1755和22类4290，VCA加三头分别12220和14755。此项是现有12步入口的初始化核对，优化、BN校准、模型前向及新增query释放均为0。首次复核脚本的常量导入路径错误发生在数据加载前，纠正导入后复核通过，生产模块无须因此修改。

后续Issue245实现独立登记的完整预算轻量适配入口，验证恢复和选模，再进行真实对照。真实长训练先报预算和预计耗时，使用2～3张满足门槛的GPU运行独立案例队列。训练启动后vca只值守训练，普通搭建、改进和测试继续在正常工作回合进行。此次冻结表征采集完成，轻量适配真实精度与源初态表征比较尚待验收。

## 证据位置

- 采集回执 `runs/g4-e0-issue243-full54-representation01/receipt.json` SHA256 `541f015f8b24c01da7342356469c11588770dc7ead6e9fd9697eeb333a9289c9`
- 独立复核 `runs/g4-e0-issue243-independent01.json` SHA256 `72651b34be4addfdfdcaa3612b338d3d6f0071aa2f203a70a9e65269778c8b59`
- 上下文核对 `runs/g4-light-context-parity-precheck01.json` SHA256 `b0067aaefea1daefadf69d921ed5e4fb4c43eaf676d43b92170e8f356469db91`
- 初态核对 `runs/g4-light-initial-parity-precheck01.json` SHA256 `c5e81d0b8f41346ce94500d42034d0f6d7e0148ff153ff9b12f620b0e119db86`
- 本机完整副本 `F:/E2E_VCA_worktrees/issue243-export01`，归档SHA256 `bbf74161d9371da8797c44f9d50bf19d4cb09da39fbb52c532477486e3be4e3e`
