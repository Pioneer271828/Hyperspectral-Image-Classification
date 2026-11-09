# BN-ALIGN1：DomainBN统计校准与保存恢复验收

2026-11-03；[Issue #103](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/103)、[PR #104](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/104)。用户批准继续处理BASE-97C发现的统计一致性问题。**本轮完成默认关闭的校准模块、保存恢复及三种子冻结验证；没有训练权重更新，没有新正式test评价。**

## 范围与实现

新增`model/bn_calibration.py`、`scripts/validate_bn_calibration.py`、`tests/test_bn_calibration.py`。原207个源码/配置未修改，模型、旧训练/恢复入口、损失、采样、数据及默认配置不变。

校准默认`none`，返回原对象且不读取观测。显式开启`scene_population_denoise_bn_v1`和`acc1p_full_scene_transductive`上下文后，生成单独的eval模型副本；每个传感器分别重算前端两层denoise DomainBN的running_mean/var。先完成第一层统计并使用它前向，再计算第二层，避免将过时上游特征的统计写入后层。以整幅场景为总体，方差correction0，与该场景批统计归一化一致；不是PyTorch默认EMA的correction1更新，也不跨传感器混合平均。

全部权重、BN仿射参数/eps/计数器、其他分支/未选域缓冲、源模型均保持不变。校准API只接受net_input/wavelengths/fwhm，不接收标签。源checkpoint SHA256、场景/元数据身份、允许上下文、改变的缓冲键和新state摘要写入既有checkpoint侧车；普通加载后显式`verify_calibration_binding`核验。错误场景、来源、上下文或被篡改的状态拒绝使用。原通用loader不自动强制该新契约，调用方必须验证。

现有checkpoint门槛为FP32；FP64仅用于数值公式对照，不通过原FP32 loader导出debug模型。本模块没有接入训练期验证选模；校准模型不是原优化器训练状态，不能替代精确续训快照。

## 实现前的文献与开源核查

已阅读[原BatchNorm](https://arxiv.org/html/1502.03167)§§3–3.2、[Rethinking Batch in BatchNorm](https://arxiv.org/html/2105.07576v1)§§3/5/A.2/A.3、[PyTorch2.7官方契约](https://docs.pytorch.org/docs/2.7/generated/torch.nn.BatchNorm2d.html)。官方fvcore的`precise_bn.py`及实际根Apache-2.0许可证固定于`8363d8d9e048e04a528112a15a50696fe125f97f`，已核查总体矩与批方差、域与数据访问边界。

采用最小独立适配；没有复制、安装或运行外部训练工程/代码，也没有引入额外依赖。方法和许可范围登记在[参考核查](../../docs/references/bn-calibration-review.json)，使用方式见[模块指南](../../docs/guides/bn-calibration.md)。论文中的通用视觉结果不当成本项目收益证据。

## 实际版本、环境与测试

分支`codex/issue-103-bn-calibration`；核心源码实际验收完整SHA：

```text
dc5da4a9318f9e77febe848ac7cc28c4c9d19885
```

首版`54d1a86e3e26530672c0426981d084eeed6b2330`的CPU8/GPU8/全量336通过，真实验收因过紧的FP32参考概率容差失败，详见下文。随后只加强验收中的数值控制，没有修改生产校准算法或精度默认。

首选rmvision握手超时；使用授权lbx LXC `10.69.37.243:1222`，Python3.10.20、PyTorch2.7.1+cu128。独立checkout `/home/lbx/hsi-pioneer271828/checkouts/bn-align1-issue103`，复用源checkout前检查其干净且SHA正确，再clone，不改动旧运行目录。先推送GitHub，再用Git bundle同步、核对完整SHA和干净工作树；不冒充容器直接GitHub拉取。GPU4开始时约21GiB空余、低负载，未操作其他进程或安装依赖。全部执行在远程，Windows只文件/Git/SSH操作。

以下命令在该checkout执行，`python`实际为`/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，统一前缀`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4`：

```bash
python -m unittest discover -s tests -p test_bn_calibration.py -v
BNALIGN_DEVICE=cuda python -m unittest discover -s tests -p test_bn_calibration.py -v
python -m unittest discover -s tests -p 'test_*.py' -v
python -u scripts/validate_bn_calibration.py \
 --expected-commit dc5da4a9318f9e77febe848ac7cc28c4c9d19885 \
 --campaign-dir /home/lbx/hsi-pioneer271828/runs/base97c-issue101-20261103-full01 \
 --prepared-dir /home/lbx/hsi-pioneer271828/runs/base97a-honghu-issue97-20261103-prepared01 \
 --data-root /home/lbx/hsi-pioneer271828/datasets/snapshots/joint-honghu-20261103-issue97/dataset \
 --out-dir /home/lbx/hsi-pioneer271828/runs/bn-align1-issue103-20261103-real02
```

最终CPU9/GPU9、全量337通过（全量44.585秒），181个Python AST及51个JSON解析通过，原207源码/配置对象逐项不变。专项覆盖默认恒等、逐层矩、参考一致性、源/参数/其他域/计数器保护、域顺序/幂等、零方差、标签/非有限/错误上下文拒绝、保存重载/来源/场景/状态篡改，以及严格FP64控制。

## 冻结真实验证与结果

继续使用BASE-97C原primary：种子64/512/2048，选择位置1503/993/1503不变，既有1503步预算不变；没有重新选checkpoint、优化器更新或额外标签。准备manifest仍为`414084cb22aef1c1866bb6512fa1f99c4397425de625c6fc2608ce0fb3eb1b6b`，数据/图root64固定。比较事先登记为原保存统计、校准并重载、运行时denoise批统计参考、循环交换传感器统计负对照；保存前的校准模型另作重载对照。

所有统计在评分前冻结，校准函数不读取标签。校准使用既定整图无标签上下文，包括未标注像素；准备验证器仍检查完整标签文件/划分身份，训练/验证标签只用于后续readout，不参与统计拟合。没有test前向、评分或改写旧正式成绩。

| 种子 | 原验证宏OA | 校准重载验证宏OA | 配对差值 |
|---|---:|---:|---:|
| 64 | 96.06% | 96.13% | +0.07个百分点 |
| 512 | 83.85% | 92.18% | +8.33个百分点 |
| 2048 | 85.42% | 95.48% | +10.06个百分点 |
| 均值±样本SD | **88.44±6.64%** | **94.60±2.12%** | **均值+6.15个百分点** |

宏OA是三个数据集等权，±采用ddof=1。保存前、重载后、运行时参考的各种子监督预测/指标一致。训练支持点宏OA89.91±6.18→95.74±2.26%；没有新增监督训练。

| 数据集，验证指标 | 原OA | 校准OA | 原AA均值 | 校准AA均值 |
|---|---:|---:|---:|---:|
| PaviaU | 95.89±3.65% | 96.80±2.74% | 95.43% | 96.91% |
| Xuzhou | 84.58±13.53% | **98.56±0.00%** | 77.91% | 98.83% |
| HongHu | 84.87±7.96% | 88.43±3.92% | 86.93% | 89.16% |

Xuzhou三个种子OA计数相同，不代表各自逐类错误完全相同。HongHu仍是主要短板；512仍使用原uncalibrated验证选出的epoch330模型，不能据本轮结果宣布原选模已经解决。

错误交换传感器统计后的验证宏OA为5.19±0.74%，且绑定核验拒绝其作为合法校准模型；说明正确域统计及绑定不能省略。这是刻意错误的机制负对照，不是训练失败。

三种子均值不降低、3/3配对为正，各数据集平均AA不降低，预登记验证候选规则通过。**这支持统计差异是当前下降/波动的重要来源，并证明校准结果能够正确保存使用；数据仍是已反复查看的同一验证集，不能写成独立精度确认、正式test提升或跨传感器泛化。**原BASE-97C正式test宏88.66±6.10%保持原记录，不能替换为本表94.60%。

## 数值差异、失败与保护核验

首轮真实验收在FP32校准与原生批归一化参考的概率比较失败：一个值差4.72e-5，超过原rtol1e-4/atol1e-5。固定权重重放发现native FP32计算差异最高约7.09e-5，但监督预测未变；同场景FP64严格重放差约9.27e-15。初次FP64调试遇到旧稠密图转换固定FP32的类型冲突；另一次外置诊断导入路径错误亦保留。调试时将邻接表示转稀疏并逐值核对不变，未改生产图实现/默认精度。

最终三个种子都增加FP64严格控制，rtol1e-9/atol1e-10，最大差8.49e-14。生产仍FP32；原生参考rtol1e-4/atol1e-4，实际最大概率差1.28e-4（组合相对/绝对容差内），所有train/val硬预测严格一致。校准保存重载state精确相同，概率按原分类器容差rtol1e-5/atol1e-6通过，最大差5.07e-7。没有将GPU原生结果说成逐位等价，也没有通过修改训练或正式分数掩盖失败。

独立外置核查用原prepared训练/验证标签与保存概率重算270组混淆/OA/AA/NLL及预测计数，重新载入三个校准模型校验state摘要，通过。183个历史训练文件、13个原始/准备文件不变；源模型/输入/权重均保留。新校准模型的12个running mean/var缓冲更新，其他所有state受逐项保护。

最终真实入口约62.88秒，包含输入核验、五组读出、FP64控制和文件hash，不作为生产校准或推理测速。没有梯度训练和依赖安装。原始失败目录`real01`及相关日志保留，新成功目录`real02`独立，未覆盖重试。

## 产物、决定与下一步

远程专属根`/home/lbx/hsi-pioneer271828/`：`runs/bn-align1-issue103-20261103-real02/`含plan/result、每种子独立calibrated.pt/侧车、监督概率、review和独立回执；`transfer-records/issue103-20261103/`含实际命令、CPU/GPU/全量日志、数值诊断及失败记录。权重、数据与原始日志不提交Git。本机核查归档在`C:/Users/asus/.codex/artifacts/issue103-bn-align1/`。

决定：保留为默认关闭、显式绑定当前整图协议的可用校准模块。下一BN-ALIGN2应统一验证、选模与最终推理的统计规则；优先复用现有51个已保存状态做预登记的只读开发检查，避免不必要的重训。若进行正式确认，重新冻结选模/评价与独立确认条件，不能在本次test上反复选择方案。训练期接入不应把临时校准统计写回优化器的训练模型。

三主线保持不变，当前不新增MON/剪枝/新机制作为前置；稳定强基线后继续独立单支路/图对照及M4/跨传感器机制。完成文档/F同步、远程最终核对和版本管理后，本模块停止。
