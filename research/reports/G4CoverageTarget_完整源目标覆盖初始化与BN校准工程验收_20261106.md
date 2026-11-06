# E1完整源目标初始化与BN校准组件工程验收

Issue #199 / Draft PR #201；开发分支`codex/issue-199-coverage-target`，工作树`F:/E2E_VCA_worktrees/issue-199-coverage-target`。实际测试/生产组件SHA **`42a2f463ecdb4223d78537c4178a28191a50ad5e`**。本项完成初始化和校准推理组件，**不等于E1目标训练、完整实验或真实收益已完成**。

## 对应计划、用途和实现

对应主清单P4 §10.1–10.3、实施计划§14 E1及§19的真实迁移接入，前置为188/190/194/196和已验收E1算子。需要把完整none源权重、fresh目标头/域与E1的实际观测元数据绑定，同时保留原覆盖关闭训练/校准合同。

新增`g4_coverage_target.py`、只读CPU入口`scripts/audit_g4_coverage_target_initialization.py`、`tests/test_g4_coverage_target.py`及参考核查JSON。没有改原331的模型、训练、校准、数据或配置文件。新的总体统计BN策略单独命名；原`calibrate_target`仍拒绝E1模型，不用撤销原保护接通新功能。

工厂支持scratch和完整源全量微调，none/continuous/anchor_only/binary/shuffled五条件。none参数不增，四启用条件各新增194参数；所有原共享权重、fresh三头和目标域状态逐张量复制，与冻结184基础初态一致。新增E1构建使用局部/隔离RNG，启用E1后的全局RNG与同角色原工厂相同；没有声称原完整源读取过程完全不消耗调用者RNG。194是参数登记数，不是194个标量均已变化。实际基线潜在通道数128，此194参数量不随K线性增加，未将其改成64。

显式前向只接收无标签OBS_KEYS；每个子集必须是绑定全观测的有序带号/波长/FWHM/质量子集，不可升级为verified或更换元数据来源。元数据进入VCA和覆盖调制；none保留原前向返回合同。全观测总体BN以无标签像素/区域总体均值和correction0方差更新，只允许该目标的running_mean/var改变，其他参数、仿射系数、计数器和原模型不变。receipt绑定观测、元数据、架构、实际调制器/几何配置、源状态及校准后完整状态；权重/sidecar均需预期哈希后才能重载。

校准/FP32评价使用已验收ordered行归约。CPUFP64仅作独立数值参考，必须显式稀疏邻接；旧骨干的稠密图缓存仍为FP32，未改它来支持FP64，也未声称原checkpoint构造器支持FP64 dtype往返。

## 原文与复用核查

已重新阅读[FiLM原论文v2 §2.1–2.2](https://arxiv.org/html/1709.07871v2)，核对作者仓库`ethanjperez/film`固定`fe43ddf8a22b339dcca2efa07091ce9d498955cf`的`FiLM.forward`和[实际LICENSE](https://github.com/ethanjperez/film/blob/fe43ddf8a22b339dcca2efa07091ce9d498955cf/LICENSE)。条件仿射关系及视觉推理实验不能当作HSI覆盖迁移收益证据。作者代码根许可CC-BY-NC-4.0及其CLEVR继承范围保留，没有复制外部代码、训练器、数据协议、权重或依赖，没有运行上游实验。

本项复用项目原E1覆盖几何、调制器和同容量条件；总体BN代码为项目原校准函数的独立适配。DOFA仍仅是相关波长条件化文献，不作为本机制或VCA的来源证明。固定核查见`docs/references/g4-coverage-target-reference-review.json`。

## 实际环境、命令和结果

授权LXC根`/home/lbx/hsi-pioneer271828`，Python3.10.20、Torch2.7.1+cu128/CUDA12.8，GPU0 RTX4090。实际测试checkout `checkouts/g4-coverage-target-issue199-engineering02`，Linux分支`codex/issue-199-coverage-target-final`，干净完整SHA同上。真实初始化和独立核验分别为`checkouts/g4-coverage-target-issue199-real01`、`checkouts/g4-coverage-target-issue199-independent01`，同一SHA。原始日志在`runs/issue199-{cpu,gpu,full,checker,help}02.log`。

公共环境为`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`，在checkout执行：

```bash
CUDA_VISIBLE_DEVICES= COVERAGE_TARGET_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_g4_coverage_target.py -v
CUDA_VISIBLE_DEVICES=0 COVERAGE_TARGET_DEVICE=cuda /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_g4_coverage_target.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -v
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/check_repository.py --expected-commit 42a2f463ecdb4223d78537c4178a28191a50ad5e
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/audit_g4_coverage_target_initialization.py --help
```

CPU6/6，35.539s；GPU6/6，40.226s；全599项598通过+1既有GPU专用CLI跳过，602.595s；checker378Python/91JSON及help通过。GPU专项开始门槛19621MiB/60%，全量开始20484MiB/9%，均符合free≥12288MiB、util≤60%。全量仍让GPU0可见以满足旧source/M4 CUDA钩子，不能称纯CPU回归；无自动CI已运行的声明。

覆盖全部五条件/两种初始化的基础状态与RNG控制，full/75%实际B/连续缺口/范围视图及坐标/元数据一致，零初始化的ordered概率精确一致，index-only和范围外无支撑在非零偏置下仍保护。四启用条件各合成3步确认194参数登记、整体正梯度和实际参数变化；未把部分条件常零特征对应权重描述为全标量更新。校准参数保护、5个目标BN的总体统计单位/样本数、独立CPUFP64函数总体统计oracle、全部五条件3次精确重载、状态/质量/来源/配置/标签/子集校准/权重对/输出/CLI拒绝通过。GPU测试中的FP64 oracle明确在CPU运行。没有对独立GPU优化轨迹作精确等价结论。

## 真实初始化和独立核验

真实CPU核验为三场景×scratch/完整源FT×五条件，共30初始化，仅model64/head1064；基础6个角色复用184冻结初态。所有不属于覆盖调制的状态哈希均与184一致，四启用条件仅新增194参数。67个源/观察/上下文/元数据/证据文件不变，6.259387s。只做初始化和参数检查，没有真实网络前向、BN校准或优化。

另一干净checkout不调用本项`initialize`，使用原none工厂6个基础模型和已验收`CoverageModulation`构造预期状态字典，独立复算30个完整状态/架构/参数数量与源初态；68文件不变，3.468071s。它是状态重建核验，不是30个模型的概率审计。两侧均用Module forward/Adam拒绝断言并确认没有GPU上下文；受保护文件哈希会读字节，但不反序列化支持/验证覆盖层或家族标签。不访问原GT/query。

producer为`runs/g4-coverage-target-issue199-preflight01/receipt.json`，SHA `778b0633836a7fa3ec33436cba3dcf31823dfd56a8df8c8c05ebb3a5ffe7dc5c`；独立为`runs/g4-coverage-target-issue199-independent01/receipt.json`，SHA `496034be102cfe54a9c5bce0b0f2b84b350006f10641e6599e2d53aaa30eddd5`。工程证据同字节副本`docs/architecture/g4-coverage-target-engineering-results.json`，SHA `fe418cd63e21297a6bb97e9a4a869f8fc318de476d6ecf76669e9f2af81797db`。实际真实更新0、forward0、BN0、query0；没有精度或速度结果。

## 保留的失败、限制和下一依赖

首版`09884202808e683a76cff7cb55d5c556a58bc63d` CPU6项为4通过、1失败、1错误，35.047s：断言误写潜在K64而真实128；FP64 oracle使用稠密邻接触发原图缓存Float/Double不符。42版本使用实际K，FP64参考显式稀疏图，并提前拒绝不支持的FP64邻接，没有修改原骨干。原日志`runs/issue199-cpu01.log`保留。外部真实审计脚本01因`else194`语法错误，在运行前拒绝，未读输入或写receipt；02修正，原脚本/日志保留。

GitHub TLS/连接器传输故障、初次REST路径404及超时不冒充成功；不确定POST后先GET检查，未重复创建PR。42提交最终已推送并核远程完整SHA，PR201保持Draft直至本组件交付收尾。原生CUDA独立优化轨迹等价仍不在已通过范围，196原失败继续保留。

当前仍是少样本整景传导式上下文。三组中心仍approximate、FWHM/SRF缺失；400–2500nm是原全局参考，不是重新认证仪器范围或逐像素可靠性。空间隔离、传感器家族排除、更多模型/支持集种子与真实同条件结果仍需计划后续实验。

验收结论仅为**E1完整源目标初始化/元数据前向/总体BN校准与重载组件工程通过**。下一项是有界E1目标训练、恢复和公平控制接入，之后才可能登记真实E1对照。真实实验继续遵守184完整接受→独立审计→拟合逐类成本复盘→全冻结ordered query的既定顺序。本任务未停启184控制器或运行其审计/query，资源排程198由用户侧会话负责。

主入口只同步报告、参考核查、证据、current-status及远程生成投影；候选源码不合入primary/main。已交付组件在current-status记录聊天报告位置，避免重复报告。
