# E0-R同区域冻结特征统计组件工程验收

Issue218 / PR219；对应研究主清单§12.5/§16，模块计划§13/§16.1及20261030宽度论文评审中的E0-R待办。实际tested SHA **528865a631eb7c1d2620113ae8e623cbe5948fbb**，分支 `codex/representation-redundancy`，工作树 `F:/E2E_VCA_worktrees/representation-redundancy`。本项为合成特征统计/读取组件工程验收，不是实际模型冗余、裁剪或加速实验。

## 实现和文献复用

只新增 `evaluation/representation_redundancy.py`、显式统计CLI、六项测试及 `docs/references/representation-redundancy-review.json`。原331 Python/config不改，无新依赖、模型或训练器改动。218代码不依赖216新Python；仅同步已验收216文档/证据/current-status，防止状态回退。

本次回读指定[v2原文§3.1/§3.2](https://arxiv.org/html/2601.22841v2)：中心化SVD、协方差参与率和通道相关用于描述表示。作者[机构页面](https://www.bifold.berlin/people/leonard-hackel.html)及论文页、标题/作者代码限定搜索未确认可采用的作者代码；状态为unconfirmed，不推断任何渠道均无源码。无上游代码复制、执行、SHA或代码许可复用声明。arXiv的论文CC-BY-4.0不当代码许可；HTML后半未完整渲染，旧PDF哈希只作为继承记录，本轮未重核全文PDF。

独立实现标准统计，未移植论文ViT缩宽、训练或下游任务协议。输入CNN[N,C]、Graph[R,D]、非负有限Q[N,R]；CNN按Q转置加权池化并除区域列质量，零质量区域排除。两路共用同一组局部RNG抽样并排序的区域索引，区域等权，cap1..8192、通道1..512，记录全部实际选中索引/质量/种子。密集、COO及CSR均支持；不将邻近区域计作独立统计重复。

统计为CPU FP64中心化、样本协方差ddof1、SVD特征值/解释方差/累计解释方差、数值秩、80/90/95%维数及参与率。参与率用归一化特征值计算，避免直接平方trace溢出。N<D时显式记录未列出的零特征值维数。Pearson均值排除对角和方差不高于声明floor的列；跨路相关单列全部有效通道对，不假定两路第i通道语义相同。空/单样本、常数/近零协方差返回null和原因，禁止非有限输入/溢出及NaN JSON。floor默认1e-24，单位为特征平方；不把数值秩等同任务所需宽度。

## 实际Linux环境、命令和验收

授权LXC `/home/lbx/hsi-pioneer271828`，干净 `checkouts/issue218-redundancy-engineering02`，分支 `codex/issue218-redundancy-final02`，完整SHA同上。Python3.10.20、Torch2.7.1+cu128；本组件统计CPU FP64，新增GPU专项 **0**。完整回归沿用GPU0 RTX4090供既有CUDA钩子，启动门槛free≥12288MiB/util≤60%，记录 `runs/issue218-full02-gate.txt`。Windows仅编辑/读文件/Git/SSH/SCP，无rmvision复跑或自动CI声明。

公共环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`；Python为 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。

```bash
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -p test_representation_redundancy.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu EDGE_TARGET_DEVICE=cpu EDGE_RUNTIME_DEVICE=cpu FUSION_TARGET_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit 528865a631eb7c1d2620113ae8e623cbe5948fbb
python scripts/analyze_representation_redundancy.py --help
```

CPU **6/6，11.428s**；完整回归 **651项，650通过+1既有GPU专用CLI跳过，1068.499s**；checker **402Python/103JSON**、help通过。完整回归不称纯CPU；新统计组件没有GPU速度主张。

手算重复通道rank1/deff1、正交rank2/deff2、常数/空/单样本通过；独立NumPy covariance/eigvalsh/corrcoef对照、全局尺度不变与非均匀通道尺度敏感性、floor敏感性通过。加权Q稀疏/密集/CSR、零质量区域、跨路共同索引、局部抽样复现与Torch/NumPy全局RNG及输入不变通过。实际合成CLI、原checkpoint仅哈希不反序列化、无NN/optimizer调用、错误split/stage/hash/映射键/输出与训练/query/width参数拒绝通过。

CLI仅接受精确schema和CNN/Graph/Q三键weights_only CPU archive；checkpoint仅哈希，使用已认证的规范路径加载并记录解析路径，符号链接用例通过。输入文件在计算前后核哈希，输出新建在checkout外，不覆盖完成结果。stage1/2分别约束128/64维。

## 证据、失败和限制

工程原件 `runs/representation-redundancy-issue218-engineering01/receipt.json` SHA **097521d063d944f56ffd0a57777ba77f7ae297af5d937cfa68c9de033762999f**，同字节仓库副本 `docs/architecture/representation-redundancy-engineering-results.json`。日志 `runs/issue218-{cpu,full,checker,help}02.log`；所有运行产物在checkout外。

前版bd66fe42ab082aac4eb45658db28bf41d2006650 CPU6/6、12.151s通过，随后在代码审阅中补规范输入路径、隐含零维度和相关null原因；最终528重新专项及全量验收，未以旧CPU通过替代新版本。未发生统计测试失败；连接器/TLS失败及恢复记录保留。新增220阶段采集候选的首次fixture失败另属220，不能混作218失败或通过。

真实特征archive读取 **0**，真实NN/BN/优化/query均 **0**；没有训练模型数、更新预算、精度、小类改善或速度结果。CLI的train/validation是调用方声明，哈希仅证明文件身份，`extraction_provenance_verified=false`；不证明层激活来源或独立划分。额外archive键在weights_only解码后拒绝，本CLI不是原GT提前释放边界。CPU FP64统计不等于全网FP64。

原协方差谱受通道单位/非均匀缩放影响；低有效维度或高相关不证明任意通道可无损删除，也不证明分支功能重复。实际E0-R的冻结提取、全部角色覆盖及容量/精度/硬件成本对照仍未完成；未删去计划必测范围。

工程结论为同区域冻结特征统计和显式读取组件通过。下一 **Issue220 / PR221** 原始stage1/2只读采集和已完成全54/204分析的身份边界工程已登记推进；它使用显式整图传导观察支持，不伪称独立train/validation块。184全54接受→独立331审计→186真实复盘→原331全冻结ordered query→204预登记结论后才执行真实模型采集。本项不接管198资源交接、改冻结184或合并main；按用户要求简报进度后连续推进。
