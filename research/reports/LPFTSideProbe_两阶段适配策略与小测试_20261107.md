# Issue250：隔离LP-FT两阶段策略候选与小测试

对应主清单P4 §10.3–10.4及实施计划§19。用户在独立侧聊授权优化与小测试，并明确要求不干扰主对话分支。本候选基于已交付 `f1bb584a3430088438500a3473aedda40e253447`，分支 `codex/side-lpft-probe-20261107`，工作树 `F:/E2E_VCA_worktrees/side-lpft-probe-20261107`。主规划 `HIC项目搭建与测试2` 已收到分工与Issue250通知。

## 问题、假设和实现

Issue184完整54模型支持点已拟合，HongHu残差全量微调相对从头约下降3.10pp。新随机头与源骨干同时更新可能造成有用特征偏移；该假设尚未被因果对照确认。本候选先训练新头，再用小学习率更新骨干，供后续与仅头、适配器加头、原全量微调进行同预算比较。

新增 `lpft_probe` 包，未接入任何已有训练器、CLI、配置或检查点路径。`LPFTPlan` 显式绑定总预算、头训练步数、两组学习率和weight decay。头训练预算计入总预算，两个阶段必须非空；骨干学习率不得大于头学习率。模型只允许 `backbone.` 和 `heads.` / `hgnn_heads.` / `cnn_heads.` 参数，拒绝未知参数、别名共享和缺失分组。

`start_phase` 只允许在0和probe_steps边界调用。阶段切换保留所有模型张量，重建Adam并明确清空全部优化器历史。两阶段均设整网eval，BN统计及dropout行为冻结；BN仿射参数只在第二阶段训练。这是本项目的独立候选合同，不是声称原论文或原G4全量微调采用同一BN策略。调用者必须保持该模式、维护实际更新游标并弃用旧优化器。函数本身不是防重放、故障恢复或完整运行时。

`phase_binding` 为策略、精确参数名/形状/dtype和训练权限生成指纹；不认证源权重、数据、实际执行步数或完整检查点。生产接入仍需主规划另行验收。

## 文献与代码复用边界

- 原论文：Kumar等，*Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*，arXiv:2202.10054v1，2022-02-21，§4.2 LP-FT；https://arxiv.org/abs/2202.10054v1 。已读取原论文HTML的摘要、方法动机和理论设定，§4.2完整正文提取受间歇TLS失败影响，仍需补核；不冒充全文核验完成。
- 论文认为在较好预训练特征和较大分布差异下，同时更新随机头与特征可能损伤OOD表现；并非所有任务上微调都更差。本文不将其图像基准提升幅度外推为HSI收益。
- 原文HTML可见fastai相关示例链接，但不是本论文作者代码官方性证据。直连API曾TLS失败、主页候选返回404；随后GitHub连接器成功读取作者仓库 `https://github.com/AnanyaKumar/transfer_learning`，README明确称为本论文代码。已固定SHA `13c3e2bbb8a34bbb781e28cee96ddf20bed5f76f`，读取README及 `unlabeled_extrapolation/baseline_train.py` 的头初始化、linear_probe冻结、线性头权重加载、eval模式和优化器分组部分。
- 固定版本完整树只有 `CLIP/LICENSE` 许可文件，根API license为null。该嵌套许可不能覆盖本次查阅的训练器；作者训练器复用许可仍未确认，因此不复制源码。上游线性探针可用提取特征加sklearn逻辑回归，接入头系数后再微调；本候选采用项目三头Adam，明确是思想适配而非上游数值复现。
- 当前组件完全独立实现，无上游源文件复制、模型下载、依赖安装或外部训练协议引入。仅使用项目已有PyTorch。

## 小测试范围与当前执行状态

新增 `tests/test_lpft_probe.py` 共10项：非法预算/学习率、游标与总预算、冻结骨干/BN和三头实际变化、阶段切换保留权重/清空Adam/分组学习率、12步对独立手写策略精确一致、边界序列化重启一致、非边界无副作用拒绝、未知/别名参数拒绝、只读策略指纹、现有G4网络6步合成支持点训练。

合成边界恢复只验证两个阶段之间重启，阶段内部恢复、RNG/调度/生产接受和数据身份绑定未实现。最后一项用现有G4构造器及合成标签，未使用真实源权重或真实场景，因此不能称真实迁移测试。无GPU、真实GT、query访问或长训练。

候选提交时本侧聊尚未执行Python测试。独立SSH连接LXC被拒绝（publickey,password），备用rmvision连接在banner阶段超时。没有读取主对话认证材料或复用其活动会话。随后主规划经已有授权连接，在独立远程检出执行并于本聊天可读记录确认 **10项CPU小测试通过**，含阶段切换、边界恢复一致性和合成网络接入。侧聊已核读主规划的确认消息，未将它写为本侧独立重跑。

候选源码提交 `f46a1ec92733a3b517193a4db69e5d4c4a77e150`，Git空白检查通过，已推送专属分支并建立 [Draft PR251](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/251)。首次push TLS失败、第二次正常成功，未强推。完整SHA及增量bundle `F:/E2E_VCA_worktrees/side-lpft-issue250-probe01.bundle` 已发主规划。主规划已核本地SHA并将同一bundle传入授权LXC；后续仅报告修改，三份Python源码未变。实际日志、耗时、环境清单及执行回执已请求导出至独立 `F:/E2E_VCA_worktrees/side-lpft-issue250-evidence01/`，本报告当前尚未收齐这些原始记录，PR继续Draft。

建议主规划使用独立 `checkouts/side-lpft-issue250-probe01`，认证本分支实际完整SHA并保持干净，产物放checkout外，使用既有CPU环境：

```bash
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_lpft_probe.py -v
```

主线程负责Issue249既有双卡批次、半小时值守和current-status.json；本候选不改动这些资源。正式训练须另行登记源身份、支持集/预算、BN策略对照、验证频次和选模，先补完整恢复再执行。比较BN冻结与原校准策略时应单独控制，不能将复合改变全部归因于LP-FT。真实收益要用独立支持种子/空间评价确认，保留184负结果。

## 交付边界

### 后续搭建接手与阶段内恢复扩展（2026-11-07，待验）

用户在本侧聊明确要求接手后续模块搭建，并通知主对话。已发送接手通知；主对话正在收口Issue249显存探针，双方仍各自独立工作树，未交接的公共文件/current-status不并发写。本侧先扩展独占Issue250，新增 `lpft_probe/runtime.py` 与 `tests/test_lpft_probe_runtime.py`，不改现有生产入口。原f46策略及10项CPU通过记录不覆盖本次新增代码。

`LPFTRun` 拥有独立模型副本、成功更新游标与Adam状态；自动在头训练预算结束时一次切换，完成后拒绝额外更新。内存快照包含模型、策略、显式输入SHA声明、参数/缓冲结构、Adam及Torch CPU/单CUDA设备RNG。恢复返回新模型，先核验再设置全局RNG，不覆盖输入模板；失败更新使实例不可继续或导出，须从已保存快照重新恢复。固定BN缓冲不允许改变，必须全部活动参数都有有限梯度。按阶段游标核验Adam每个参数的实际step，拒绝超参/角色/预算/张量异常。

输入SHA是消费者提供的声明，组件不读取实际文件或自动证明其真实性。必须由生产消费者先核真实输入及checkpoint哈希。本阶段不实现磁盘原子发布、Python/NumPy/采样器状态、数据访问守卫、校准选模、完整CLI或跨设备等价；损失回调需使用固定数据或已纳入Torch RNG的随机性。单机单设备，不是多卡训练协调器。

新增12项测试（11CPU及1显式GPU），CPU覆盖0至8每个断点的完整序列化恢复、与原阶段策略逐步精确比较、边界重置一次、完成零更新、13种损坏拒绝且模板/RNG不变、快照独立、失败实例停止、缺失头梯度/错误模式拒绝、输入声明拒绝、运行时Adam合同改变拒绝及现有G4网络阶段内部恢复。GPU项仅在显式分配后启用同设备恢复验证。新增候选源码完整SHA `fa407abd7b3331488af02291d20949d912c10e9e` 已推送；首次两次push TLS失败后成功，未强推。增量bundle为 `F:/E2E_VCA_worktrees/side-lpft-issue250-runtime02.bundle`，不会更新任何生产checkout。当前本侧LXC独立认证仍被拒，rmvision banner超时；已向主对话请求安全移交空闲认证会话，未索取或复制凭据。新增测试未执行，结果待远程验收，不声称通过或提升精度。

待运行命令（授权Linux，独立干净检出上述SHA，CPU不占GPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p 'test_lpft_probe*.py' -v
```

预期发现22项，CPU路径21项执行、1项显式GPU跳过；这是预期数量而非实际结果。远程通过后才补tested SHA、实际耗时与退出码。GPU专项和仓库检查按实际资源与回执继续。

状态：原策略的主规划独立远程CPU小测试10项通过，原始记录归档待收齐；新增运行状态已实现但待远程验收，GPU、全仓回归、完整训练入口和真实效果未验。允许变更仅为 `lpft_probe/`、`tests/test_lpft_probe*.py` 和本报告。没有修改主入口、main、已有模型/配置/CLI、current-status.json或主计划；不合并其他分支。后续搭建由本侧接手，主对话收口中的显存探针另行交接，避免双写。临时侧聊不创建自身定时任务，长训练及值守由主聊天登记。
