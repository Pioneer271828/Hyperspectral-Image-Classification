# E0-R原阶段只读采集与全54身份工程验收

Issue220 / PR221；对应研究主清单§12.5/§16、模块计划§13/§16.1，衔接218冻结特征统计。实际tested SHA **87bc0835020c02067ffa0be9e2d6875d03d5d1f8**，开发分支 `codex/frozen-representation-capture`，工作树 `F:/E2E_VCA_worktrees/frozen-representation-capture`。本项通过合成模型阶段采集、原状态保护及已完成54/204身份消费工程；没有执行真实184采集或完整E0-R效果实验。

## 实现、用途与边界

仅新增 `evaluation/frozen_representation_capture.py`、显式导出CLI、四项测试和 `docs/references/frozen-representation-capture-reuse.json`。复用原331骨干阶段方法、OrderedValidation、无标签观察协议及all54冻结验证，204既定源/计划/分析身份，218区域统计；原331、204、218 Python/config没有改动。无新增外部源码、依赖或许可声明，沿用218论文核查范围。

在隔离的单目标eval模型内临时观察原始四个bound stage方法，调用原方法本身：stage1 CNN[N,128]/Graph[R,128]为交换前状态，stage2 CNN[N,64]/Graph[R,64]在图广播前记录。不能将广播后图像素特征再池化冒充原区域表示。三头普通预测仍由原前向执行，在原ordered协议下核对；不复制另一条骨干计算、不训练、不重校准。

临时观察属性在成功及异常时finally删除，原classmethod恢复。保护全部registered参数/缓冲、train flags、观察哈希、Torch CPU及实际使用的一张CUDA设备RNG；构造/采集置于局部fork中，不调用全卡RNG接口。严格拒绝训练态、残留梯度、标签字段、被替换的stage方法和非法形状。普通瞬态非Tensor图缓存可能被原前向使用，不承诺全部Python内存不变；调用必须在隔离进程/模型内串行执行。

CLI在任何Tensor/model反序列化前验证：原331全54冻结/独立接受数量与全保护文件；已完成204分析的固定commit和source hash、原预登记plan、54模型/162头/18组、更新0/不新释放query/消费已有指标标志、受保护输入及analysis.json/report.md输出哈希。主要收益失败不阻止诊断，不能选择性删除负结果。随后从固定launch核唯一case摘要和冻结模型对，先认证label-free观察record与11键archive协议，再解码；只用六OBS字段，不调用family或query/原GT读取器。

所有输入、原权重/sidecar、观察和原保护文件在前后核对；输出必须为全训练目录同级的新目录，避开冻结/分析/输入树且不覆盖已完成输出。两stage特征文件和统计摘要显式标记整图目标传导上下文，不伪称独立train/validation空间块，不生成218 CLI的虚假块来源声明。共同区域cap/seed/floor事前显式给定，无通道排序、缩宽、模型选择或新标签释放。

## 实际Linux验收

授权LXC `/home/lbx/hsi-pioneer271828`，干净 `checkouts/issue220-capture-engineering03`，分支 `codex/issue220-capture-final03`，完整SHA同上。Python3.10.20、Torch2.7.1+cu128、CUDA12.8、GPU0 RTX4090。Windows仅编辑/读文件/Git/SSH/SCP；无rmvision复跑或自动CI声明。

公共环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`，Python为 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。

```bash
CUDA_VISIBLE_DEVICES= REPRESENTATION_CAPTURE_DEVICE=cpu python -m unittest discover -s tests -p test_frozen_representation_capture.py -v
CUDA_VISIBLE_DEVICES=0 REPRESENTATION_CAPTURE_DEVICE=cuda python -m unittest discover -s tests -p test_frozen_representation_capture.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu EDGE_TARGET_DEVICE=cpu EDGE_RUNTIME_DEVICE=cpu FUSION_TARGET_DEVICE=cpu REPRESENTATION_CAPTURE_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit 87bc0835020c02067ffa0be9e2d6875d03d5d1f8
python scripts/export_frozen_representation_diagnostics.py --help
```

CPU **4/4，18.810s**；GPU **4/4，21.993s**；完整回归 **655项，654通过+1既有GPU专用CLI跳过，1040.544s**；checker **405Python/104JSON**、help通过。GPU专项与全量各自核free≥12288MiB/util≤60%，记录 `runs/issue220-{gpu,full}03-gate.txt`，全量保留GPU0可见供旧钩子。

实际合成三M4模式none/capacity_control/x_residual×scratch/完整none源或活动源FT共六角色。两stage原始形状、每次四个原方法各一次、三头预测CPU/GPU均逐位匹配无观察原前向，registered state、输入及RNG不变。注入中途异常后属性清理并能再次正常采集；训练态/label拒绝通过。

结构54冻结fixture中，53个其他权重是文字占位，一例为真实合成checkpoint/观察。它用于验证接受数量、来源哈希、未完成或旧边界拒绝在torch.load前发生；不冒充54份真实训练/独立审计/概率重算。错误分析计数/源码hash/更新/query标志、缺模型、输出所有权及重复输出拒绝通过。主要收益false的正例仍可消费；模拟family/query读取器被禁止，实际合成CLI成功导出两stage Tensor和摘要。一个case只输出一个case证据，不自动声称全54诊断完成。

## 证据、失败和未完成项

工程原件 `runs/frozen-representation-capture-issue220-engineering01/receipt.json` SHA **bf907ff7cd329d5f1d83a99ffea0ef691fbedcfec29ce8e1ba30ee3ff4bd46cd**，同字节副本 `docs/architecture/frozen-representation-capture-engineering-results.json`。日志 `runs/issue220-{cpu,gpu,full,checker,help}03.log`；产物在checkout外。

首版b3e2f294fafed05b6138c8787d899774a6ec1be2 CPU **1通过/3错误，5.555s**：活动模式合成fixture漏显式eval，原采集守卫正确拒绝；另两例将freeze/analysis放在输出父目录，原输出隔离正确拒绝。3d632bd只修用例eval与同级证据目录；87bc进一步从已提交204真实工程回执固定实际source hash并加负例，未削弱守卫或改原骨干。首次错误参考调用涉及合成训练态前向，不是实际184训练/原冻结模型失败。旧clone/日志、连接器/TLS及SCP认证连接关闭记录保留；未因网络不确定重复创建Issue/PR。

随后仅metadata合并f23f549535ee4b581b70e803562aed954882831c，接纳已验收218及216文档/回执/current-status；Python/config与tested87bc无差异。最终文档提交不能替代实际测试SHA。

真实模型/特征archive消费、真实NN/BN/优化/query均 **0**。本项没有真实精度、收敛、逐类改善或硬件提速结论；整图传导区域不算独立重复，统计相关也不能推断通道可无损删除。真实全54特征覆盖与容量对照仍未完成；不将单个case或工程通过当E0-R完整研究完成。

184全54接受→独立331审计→186真实拟合/逐类/成本复盘→原331全冻结ordered query→204预登记结论实际完成后，才允许真实采集。冻结入口不由本组件重建或提前释放；不接管198多卡交接。下一 **Issue222 / PR223** M7仅先验行归一化初始化/BN对照工程已登记继续，原始数值门槛、预算、损失及源合同保持。主入口只同步报告/证据/参考/current-status与远程生成投影，main未合并；按用户要求简报进度后连续搭建。
