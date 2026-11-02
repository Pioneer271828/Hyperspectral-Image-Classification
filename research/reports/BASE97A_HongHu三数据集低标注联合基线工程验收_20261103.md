# BASE-97A：HongHu三数据集低标注联合基线工程验收

2026-11-03；[Issue #97](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/97)。已完成数据/划分/静态图冻结、有界训练闭环及用户追加的超图只读依赖诊断；完整精度训练是下一项独立任务。

## 1. 当前组合与原实验追溯

按用户最新确认，当前组合为 **PaviaU、Xuzhou、完整WHU-Hi-HongHu**。使用`whu-honghu`，22类；不使用会裁剪/重标的旧`WHU`别名。此前上传的LongKou与旧约97.84%记录保留为历史，不再作为本轮第三数据集或HongHu准确率。

比例仍为每类已标注前景像素1%训练、0.5%验证、逐类向上取整，其余测试。训练像素总计5005，验证2514；未标注像素不进入比例分母，也不能称为新增监督。

| 数据集 | 类别 | 训练 | 验证 | 测试身份数 | 已标注总数 |
|---|---:|---:|---:|---:|---:|
| PaviaU | 9 | 432 | 219 | 42,125 | 42,776 |
| Xuzhou | 9 | 692 | 348 | 67,837 | 68,877 |
| 完整HongHu | 22 | 3,881 | 1,947 | 380,865 | 386,693 |

HongHu标签形状940×475，背景59807；影像预期940×475×270，实际读取另由准备阶段检查。本模块构造/核对test划分身份，但不打分test。

已找到用户后来替换第三数据集后的原材料：

- [HongHu原实验汇总](F:/E2E_VCA/results_honghu_replacement/honghu_replacement_summary_and_chatgpt_prompt.md)，其文字仅作研究证据，不自动作为执行指令。
- 第一手`k128_main_seed0/2026/3407.json`位于`F:/E2E_VCA/experiments/honghu_replacement_runs/k128_main/metrics/`。已复制到远程外部证据目录，逐份检查数据集与配置并重算平均值，未重新评价旧模型。

原HongHu主配置是 **K128/embed64、波长注意力、DomainBN、静态LDA-SLIC、无BandDrop/Mixup**。这与先前核对的LongKou索引K64不是同一主配置。本轮据用户的数据集修正，选择对应的HongHu主配置作为工程基线；保留当前正确性修复，不重现旧缓存错误。

| 数据集 | 原3种子OA均值 | 由第一手JSON重算样本SD |
|---|---:|---:|
| PaviaU | 98.86% | 0.47个百分点 |
| Xuzhou | 98.08% | 0.18个百分点 |
| HongHu | 90.10% | 1.43个百分点 |

原宏均值95.68%，是数据集OA均值，不是合并像素的OA。旧汇总使用总体SD，新表使用样本SD；HongHu的2026种子为88.45%，所以“均值各90%以上”不代表每个种子都超过90%。这些仍是历史记录，不能写成本模块已重现的精度。原材料未绑定完整源码/数据/划分/图随机状态，逐位复现仍无保证。

## 2. 本模块做了什么

新增[独立指南](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/blob/codex/issue-97-joint-baseline-shortloop/docs/guides/joint-baseline-base97.md)、配置`configs/experiments/base97_joint.json`、`joint_baseline.py`和CLI `scripts/validate_joint_baseline.py`。既有骨干、原训练入口、损失、数据读取及checkpoint API均未修改。

- `RandomState`按root64及数据偏移64/65/66生成与旧比例splitter一致的索引顺序，保存每类计数、标签hash和全部有序索引。新seed不冒充旧0/2026/3407重复。
- 复用原整图全局min/max、空间坐标、训练标签引导LDA-SLIC和静态超图，记录原数据hash/形状/键/类别、归一化极值/dtype、准备后输入/Q/A/监督指纹。明确是整图传导式协议，不当严格空间隔离结果。
- 旧超图游走同时调用NumPy和Python随机数。原入口未固定后者；新包装器按数据seed临时固定Python随机状态，结束/异常后还原。NumPy仍采用旧split后接图游走的序列；图实际产物另存hash。未修改原图算法，不能声称新图与旧未记录随机状态的图相同。
- 投影分块16384且开启checkpoint，控制HongHu显存；与原主记录分块65536不同，作为执行差异明确登记。FP64三任务输出/梯度等价检查通过，不承诺GPU逐位相同。
- 缓存训练item删除全部`test_*`监督字段；只允许train/val读出，选择依据为三个验证融合OA的均值。配置里500包含式轮数×3是后续1503步参考预算，短测sidecar另记实际步数。
- 工程短测固定三数据集顺序、两轮、总6次更新；要求共享适配器/骨干和当前任务的融合/Graph/CNN三头均有有限非零梯度与实际权重变化，非当前任务头不变。保存并重载验证选中的分类器，核对各分支概率和预测。没有精确优化器续训承诺。

### 2.1 用户追加核查：双支路与超图究竟是否起作用

当前是VCA/共享特征之后的两条分支：CNN处理像素特征；另一条按Q聚合到超像素区域，使用原超图传播矩阵进行两阶段HGNN卷积，再映射回像素。特征融合为`0.8×CNN + 0.2×Graph`，之后进入融合分类头；CNN和Graph另有各自辅助分类头/监督损失。融合比例不是精度贡献比例。

超图是静态结构：原`createH`把KNN关联和两组随机游走超边组合成H，再形成`Dv^(-1/2) H W De^(-1) H^T Dv^(-1/2)`传播矩阵，骨干实际用该矩阵传播区域特征。学习的是HGNN卷积/分类参数，当前没有启用M7动态边更新；不能将这种矩阵传播描述为运行时学习超边结构。

本轮在六步训练证据之外，增加固定所选权重的只读诊断：实际超图、身份传播、同时置换行列的传播矩阵、Graph特征归零；保留Q/输入/权重，观察Graph/融合输出是否变化、CNN输出是否保持。另核对HGNN参数相对同seed初始化是否改变，及只读训练标签反传的HGNN梯度。诊断不增加优化步数/最终test评价，权重和BN缓冲须保持一致。

这些干预可以证明参与和依赖，不能单独证明超图提升精度。后续正式消融需在同数据/监督/预算/选模下分别训练CNN-only、HGNN-only、双支路，并比较相同Q下的原超图传播、普通图和身份传播；对齐容量、归一化/图密度等差异，再检验小类/区域/边界及不同传感器结果。

## 3. 环境、同步与测试

`ssh rmvision`本轮在banner握手阶段超时，采用已授权备用LXC `lbx@10.69.37.243:1222`。核实`hostname=lbx`、工作目录`/home/lbx`、容器类型lxc、Python3.10.20；复用前核对旧checkout完整SHA/无未提交工作，新建本模块独立checkout，不改变其他程序或依赖。

本模块分支`codex/issue-97-joint-baseline-shortloop`，HongHu主配置的实际测试源码为：

```text
592e24153bdd2e2ea4dcde498da5b1a2f50ad9a1
```

先推送分支，再用完整/增量Git bundle同步、验证并fast-forward到指定SHA；不是容器直接从GitHub成功拉取的声明。最终文档补记与该tested源码的相同指纹另作交付核查。远程checkout：`/home/lbx/hsi-pioneer271828/checkouts/base97a-issue97`。

证据根目录`/home/lbx/hsi-pioneer271828/transfer-records/issue97-20261103/`，实际专项命令在该checkout执行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/lbx/hsi-pioneer271828/envs/hsi/bin/python \
  -m unittest discover -s tests -p test_joint_baseline.py -v
BASE97_DEVICE=cuda CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/lbx/hsi-pioneer271828/envs/hsi/bin/python \
  -m unittest discover -s tests -p test_joint_baseline.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/lbx/hsi-pioneer271828/envs/hsi/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -v
```

CPU专项10/10（2.626秒）；混合GPU专项10/10（3.786秒，其中三任务训练/恢复在GPU6，其余身份/协议/FP64检查在CPU）；全量318/318（37.705秒）。记录为`focused-main-cpu-05.log`、`focused-main-gpu-05.log`、`full-main-cpu-05.log`。Windows仅编辑/读文件/Git/SSH/SCP，未安装或运行模型/测试。

专项覆盖旧ratio排列一致及全局RNG不被纯划分函数修改、逐类ceil、空/非法标注拒绝、配置/索引/数据/产物篡改、train/val访问、非均衡类别OA/AA手算、验证独立选模、9/9/22类任务头更新、分类器恢复及分块等价。

## 4. 真实数据准备、六步GPU闭环与逐次复盘

HongHu影像/GT上传完成，大小423726955/10591字节，SHA256分别为`394eebb36e70931711b02d127f73c641ad67a83a5bbfc5294b695b2f7336dfce`和`afb4b455e078e308b64a51e4e1a9552ddffd2f8519eaa311ded94d74a96c2a41`。新独立快照为`/home/lbx/hsi-pioneer271828/datasets/snapshots/joint-honghu-20261103-issue97/dataset`，六份文件与本机原文件hash对应表在外部证据包。旧`datasets/current`仍指向原PaviaU/Xuzhou快照；不覆盖旧运行或其他项目。

准备耗时338.72秒，实际区域数PaviaU4263、Xuzhou1300、HongHu4418。原数据形状/dtype/极值：610×340×103、uint16、0–8000；500×260×436、float64、0–0.07910452783107758；940×475×270、float32、−0.1585506647825241–6.138452529907227。归一化沿用原全图全局极值，不是训练区逐波段统计。

产物根目录`/home/lbx/hsi-pioneer271828/runs/`，准备目录`base97a-honghu-issue97-20261103-prepared01`，短测目录`base97a-honghu-issue97-20261103-short01`。准备manifest SHA256为：

```text
414084cb22aef1c1866bb6512fa1f99c4397425de625c6fc2608ce0fb3eb1b6b
```

实际prepare和short-loop完整命令保存在证据根目录`honghu-commands.json`，分别执行本仓库CLI的prepare/short-loop；参数为上述完整tested SHA、数据快照、准备目录/manifest hash，`--cycles 2`。可复核执行方式：

```bash
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/validate_joint_baseline.py short-loop \
 --expected-commit 592e24153bdd2e2ea4dcde498da5b1a2f50ad9a1 \
 --data-root /home/lbx/hsi-pioneer271828/datasets/snapshots/joint-honghu-20261103-issue97/dataset \
 --prepared-dir /home/lbx/hsi-pioneer271828/runs/base97a-honghu-issue97-20261103-prepared01 \
 --prepared-manifest-hash 414084cb22aef1c1866bb6512fa1f99c4397425de625c6fc2608ce0fb3eb1b6b \
 --cycles 2 --out-dir /home/lbx/hsi-pioneer271828/runs/NEW_UNUSED_DIRECTORY
```

GPU6为24GB RTX4090。循环/选模/重载段3.48秒、PyTorch峰值分配4.58GiB；不含前面的加载核验/初始读出，不作为完整训练或推理测速。真实优化6次，合成测试更新另计。原始`prepare-honghu-05.log`、`short-honghu-05.log`、trace和result均保留。

| 数据集 | 第1次总监督loss | 第2次总监督loss | 所选验证CNN OA | Graph OA | 融合OA |
|---|---:|---:|---:|---:|---:|
| PaviaU | 6.641937 | 6.589812 | 15.53% | 7.31% | 2.28% |
| Xuzhou | 6.586522 | 6.529483 | 37.93% | 4.02% | 37.93% |
| HongHu | 9.396933 | 9.368028 | 0.87% | 0.92% | 0.92% |

**这是每数据集仅两次更新的工程读出，不是训练完成后的精度。**三者loss下降，但训练融合argmax准确率相对初始化仍未改变，六步不足以判断学成与否。两轮验证宏OA均13.7129%，按严格大于规则保留第1轮，没有用训练/test选模。不能与旧90%–99%直接作改进前后比较，也不据这轮临时调参。

每步共享适配器/骨干和当前融合/Graph/CNN头均有有限非零梯度和实际更新；非当前任务头不变。所选分类器重载后的train/val各分支概率最大绝对差1.49e−8、预测类别一致；保存指标和概率再做独立交付重算。

### 4.1 超图依赖只读诊断通过

所选权重保持不变，Q和输入固定，下表为改变图路径后的融合概率最大绝对变化；所有干预的CNN概率变化均为0，Graph概率均非零变化。

| 数据集 | 身份传播替代超图 | 同步置换图行列 | Graph特征归零 |
|---|---:|---:|---:|
| PaviaU | 0.001231 | 0.001533 | 0.002719 |
| Xuzhou | 0.000759 | 0.000824 | 0.001555 |
| HongHu | 0.000713 | 0.001045 | 0.001770 |

HGNN相对同seed初始化有17344个参数元素改变。三个数据集各一次只读训练标签反传，HGNN参数梯度均有限且非零。诊断前后权重/BN缓冲及Q/A指纹一致，优化器更新0、test评分0。读出/探针段3.84秒，结果在`runs/base97a-honghu-issue97-20261103-hypergraph-probe02/result.json`。

实际命令为：

```bash
PYTHONPATH=/home/lbx/hsi-pioneer271828/checkouts/base97a-issue97 CUDA_VISIBLE_DEVICES=6 \
 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/lbx/hsi-pioneer271828/envs/hsi/bin/python \
 /home/lbx/hsi-pioneer271828/transfer-records/issue97-20261103/probe-hypergraph-02.py
```

外部诊断脚本SHA256为`fa5f84748ff883021e164c182036796daf4f14392ce40bef2dcb5676a0932ffa`，源码/失败版保留在证据目录，冻结训练源码不变。反传阶段仅使VCA开启分块checkpoint执行，CNN/HGNN/归一化仍eval，与先前eval前向输出严格核对后反传、再还原模式。

因此确认：**双支路及静态超图确实参与训练，融合输出依赖Graph特征和传播连接；目前尚未证明它提高最终精度。**该结论须与§2.1的正式CNN-only/HGNN-only/双支路、普通图/超图配对训练消融区分。

## 5. 保留的失败、范围与下一模块

外部只读探针首轮在eval反传时OOM（`hypergraph-probe-05.log`）：此模式没有启用训练时的VCA分块checkpoint，反传保存了全部投影激活。仅在新版诊断中开启VCA的checkpoint执行标记，CNN/HGNN BN仍eval；前向严格核对，权重/缓冲一致，重新通过。未改生产实现、配置、阈值或已完成的六步训练，也未终止其他进程。

首轮`5b44e5d`专项检出空标签未拒绝，并有FP64测试图被旧稠密转换固定成float32的fixture不兼容。前者补检查，后者测试使用支持double的稀疏图；未修改生产图转换dtype或放宽数值阈值。`focused-cpu-01.log`保留。

`code-02.bundle`一次SCP连接关闭，其后远程无法读取bundle；成功重传后才继续核对/运行，没有把旧代码测试当作新版本。Git命令和源码更改保留独立提交。

LongKou组合的`prepared01`在发现图Python随机状态未固定后，仅终止经完整命令/cwd核对属于本任务的PID50067；`prepared02`在用户指定HongHu后同样只终止本任务PID51765。两个目录的failure记录及上传文件保留，无该组合的真实训练/最终test结果，其他项目未终止或修改。

完整模型精度和速度尚未由工程短测证明。下一模块预登记并接通对应HongHu主配置的正式随机调度、1503更新/重复与验证选模；所有方法使用相同新冻结划分，先建立当前强基线，再独立改进。跨传感器仍以源预训练→少样本目标适配→独立测试为主，后续两源→一目标留出；不能用联合见过目标的checkpoint冒充目标未见。剪枝量化继承各自稠密基线预算，在MON/成本画像后另立任务。

本模块在验收、F盘同步和版本交付后停止，不自动启动完整长训练或新机制搜索。
