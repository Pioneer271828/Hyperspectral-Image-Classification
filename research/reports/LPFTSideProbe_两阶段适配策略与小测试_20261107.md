# Issue250：隔离LP-FT两阶段策略候选与小测试

> 当前进度以文末“独立连接恢复后的真实短测与显存路径”一节为准；前文保留各阶段历史记录，旧的连接阻塞和待测描述不是当前状态。

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

### 后续搭建接手与阶段内恢复扩展（2026-11-07）

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

主对话于交接消息确认实际在授权Linux独立检出运行上述候选：tested SHA `fa407abd7b3331488af02291d20949d912c10e9e`，**21项CPU通过、1项GPU跳过，用时4.101秒**。日志 `/home/lbx/hsi-pioneer271828/runs/issue250-side-lpft-runtime02.log`。这是主对话执行、侧聊核读交接的记录；本侧尚未取得原日志字节/哈希与原始退出码，不冒充本侧复跑。GPU专项和仓库检查待独立连接恢复后执行，不因CPU成功宣称GPU一致。

### G4源模型和目标数据接入（2026-11-07，新增待验）

用户要求将重点转回直接研发后，新增 `lpft_probe/g4_short.py`，源码 `1cdfd12d00f59eb509345e73a7cda7b22d54d260`。`G4ShortCase` 从固定Issue184 launch选取目标/模型种子/架构，复用现有完整源接受检查、fresh目标初始化和已去除query字段的目标上下文，核对初始模型与源/图/标签身份。三数据集使用同一接口，保留原三头损失；初始一次ordered BN校准后进入先头、后骨干小学习率两阶段。每步检查输入未变，支持阶段内状态恢复，输出三头训练/验证端点指标。

该接入上限12步，没有模型选择、候选重复BN校准、query或长训练释放；固定BN是显式配方因素，不能称与184训练配方完全相同。当前是供小规模研发验证的可调用组件，无正式训练CLI/批次协调器。新增两项源/目标合成接入测试覆盖真实网络6步、初始/末端三头指标、阶段内部恢复、参考文件篡改及超预算拒绝；尚未执行。这两项不包含在前述21项通过中，三个真实数据集尚未运行，未证明精度提升。

主对话已明确交接后续搭建；显存探针实测 `7d09fd14821964fb398b8ff1669537b0fcbbdce8`，报告提交 `ae803d2086531df0d223be11f61ae5928e3b8a95` 只读保留。原Issue249为failed、35/54独立接受、vca暂停，恢复训练未启动。本侧没有覆盖其分支或current-status。主对话确认工具不支持可验证的跨聊天SSH会话移交，本侧已向用户请求可用连接配置；研发代码仍仅落在独立Issue250分支，等待连接后执行新增接入验证。

状态：原策略及阶段内运行状态合计21项CPU通过、1GPU跳过；新增G4源/目标小步接入待验，GPU、全仓回归、完整训练入口和真实效果未验。允许变更仅为 `lpft_probe/`、`tests/test_lpft_probe*.py` 和本报告。没有修改主入口、main、已有模型/配置/CLI、current-status.json或主计划；不合并其他分支。后续搭建由本侧接手，保持独立工作树，避免双写。临时侧聊不创建自身定时任务，长训练及值守由主聊天登记。

## 独立连接恢复后的真实短测与显存路径（2026-11-07）

### 实现与环境

本侧已通过独立SSH登录授权LXC，未移交或占用主对话会话。实际环境为用户lbx、hostname lbx、LXC、Python 3.10.20、PyTorch 2.7.1+cu128；Python入口 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。本节全部执行均在Linux完成，所有训练输出位于 `/home/lbx/hsi-pioneer271828/runs/`，没有把权重、数据或认证内容提交到仓库。

首次G4接入版本 `1cdfd12d00f59eb509345e73a7cda7b22d54d260` 在校准前未将完整模型置eval，两项测试被校准入口拒绝，耗时5.159秒；日志 `issue250-g4short01-cpu.log` 保留。修复仅为显式 `.eval()`，源码 `a163c46185df320792b064ca3330fb0cbb7ff2f9`：G4接入CPU 2/2通过9.731秒，GPU同设备恢复1/1通过2.526秒，checker通过417 Python/130 JSON。分别见 `issue250-g4short02-cpu.log`、`issue250-runtime-gpu01.log`、`issue250-g4short02-checker.log`。

新源码 `6a1b31dacefd3f7b29bdd852ccd9339b1f21d67f` 增加显式 `saved_tensors=device|cpu`。默认保留原device路径；CPU路径使用已安装PyTorch的 `torch.autograd.graph.save_on_cpu(pin_memory=False)` 暂存前向中间张量，反向时传回原设备，不改变dtype或加入混合精度。已直接读取该安装版本函数实现，未复制外部训练器。真实数值同一性需要实测，不能仅根据API含义断言GPU逐位一致。

该策略同时写入G4案例身份和runtime快照；runtime格式升为v2，旧v1快照及跨策略恢复会被拒绝，没有迁移或覆盖a163旧产物。模型、三头损失、Adam、4+8阶段预算及学习率保持本候选原合同；并非恢复Issue249或修改Issue184配方。远程独立分支 `codex/side-lpft-offload01`，检出 `/home/lbx/hsi-pioneer271828/checkouts/side-lpft-issue250-offload01`，测试前后核完整SHA及干净状态。

### 定向验收

下列命令均在上述6a1检出，设置 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`：

```bash
CUDA_VISIBLE_DEVICES='' /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p 'test_lpft_probe*.py' -v
CUDA_VISIBLE_DEVICES=4 LPFT_RUNTIME_DEVICE=cuda /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest test_lpft_probe_runtime.RuntimeTests.test_cuda_offload_tiny_model_exact_updates_and_resume test_lpft_probe_runtime.RuntimeTests.test_cuda_same_device_resume -v
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/check_repository.py
```

CPU共27项，25通过、2项显式GPU跳过，用时9.875秒；两项GPU另跑2/2通过2.509秒，退出码均0。CPU和小模型GPU验证了原路径/暂存路径的逐步损失、参数与Adam状态精确一致，以及暂存路径的阶段内部恢复。G4合成源/目标接入测试也使用CPU暂存策略。checker通过417 Python/130 JSON。没有重复全仓回归；本次不声称完整网络GPU逐位等价或全项目验收。

证据日志（均在runs目录）：

| 文件 | SHA256 |
| --- | --- |
| issue250-offload01-cpu.log | 882c0998aea2e144360d5a6a67f8f26456d9103610c3765f8fccff56e1ec77cd6 |
| issue250-offload01-gpu.log | a5802ce545b3da80b9fdd0cc4d5059e2bbe1090ebc0d711f4a4c77d8d25e45b12 |
| issue250-offload01-checker.log | 23ee692550b55b02de4ce748013d6a9141d1c6d23114f2e78e60ad51a22984b3 |

### 第一组真实短测：保留HongHu失败

外部目录 `runs/issue250-lpft-real-short01`，执行a163冻结消费者，seed64、none架构、每场景12步（先4步头、再8步骨干），头学习率0.0005、骨干0.000005，原三头损失。PaviaU在GPU4，Xuzhou在GPU5；Xuzhou结束后HongHu使用GPU5，所有启动检查free≥12288MiB且util≤60%，每卡一队列。仅训练/验证端点，无query、选模或长训练。

| 场景 | 完成更新 | 耗时秒 | GPU峰值分配字节 | 首末训练loss | fused验证OA初→末 |
| --- | ---: | ---: | ---: | --- | --- |
| PaviaU | 12 | 27.896159 | 7633015808 | 6.922524→6.409271 | 8.8889%→31.1111% |
| Xuzhou | 12 | 35.770623 | 16315833344 | 6.630093→6.213160 | 13.3333%→20.0000% |
| HongHu | 4 | 35.717785 | 23042810880 | 仅头阶段完成 | 无训练后端点 |

第一组实际28/36成功更新，8步未执行。HongHu在首次解冻骨干时显存耗尽，请求额外540MiB时剩余仅6.31MiB，PyTorch已分配约21.46GiB，保留未分配仅50.66MiB；因此当前失败更符合容量不足，不能凭此归因于大量碎片。原失败回执、trace和0–4步快照保留，没有原地重训或变更阈值。PaviaU与Xuzhou仍分别有3类、6类fused验证零召回；12步用于工程接入，不能据loss下降宣布精度改进有效。

### 第二组HongHu显存可行性短测

新版本6a1在独立 `runs/issue250-lpft-offload-short01` 登记单HongHu12步，CPU暂存，GPU5；预算先向用户告知，预计60–300秒。保留第一组失败，不将第二组作为原路径成功恢复。

外部计划复制时遗留旧聚合字段 `planned_total_updates=36`，而目标列表仅HongHu、每例上限12以及新字段 `planned_updates=12` 均一致。发现后保留原文件并写独立 `plan-aggregate-erratum.json`，记录原计划哈希和正确合计12；不改运行中输入，不增加预算。该元数据错误属于本次执行限制，后续正式消费者应在首更新前自动核验所有聚合预算。

实际12/12完成，用时339.340586秒，比预计上界多39.34秒，峰值GPU分配2100396544字节（约1.956GiB）。第一组失败时约21.46GiB；两者是失败路径与完成路径的显存记录，不构成同等完成工作的加速比。没有记录本次完整进程的CPU峰值，后续调度不能只根据GPU节省量决定并发。

loss由9.655043降至9.194064。HongHu验证fused OA保持3.6364%，graph由10.0000%至11.8182%，CNN由3.6364%至4.5455%；末端三头分别仍有20/16/18类零召回，最低召回均为0。fused NLL由3.256815降至3.138264。短预算未显示fused精度改善，不作成功收益结论，更不能替代完整训练/独立query判据。

只读快照复盘确认：三场景头阶段骨干变化均0；微调阶段PaviaU/Xuzhou/HongHu实际改变骨干张量元素59652/59812/59670；各自三头在两个阶段均实际变化，固定缓冲全程不变，Adam步数与阶段游标一致。HongHu两版本的初始模型张量精确相同，但首步loss存在约9.54e-7差异，因此保留完整GPU原路径与暂存路径未证明逐位一致的限制。不能将新路径作为旧冻结训练的透明替换。

复盘输出 `/home/lbx/hsi-pioneer271828/runs/issue250-short-review01/summary.json`，SHA256 `1aba228f7bd5163df05d818515e7f76b32677aa2c55873a0d09506f27d9f5c0a`，包含四条运行记录、三头训练/验证逐类召回、快照实际变化和全部原产物哈希。只读复盘脚本 `/home/lbx/hsi-pioneer271828/runs/issue250-short-review01.py`，SHA256 `f4389d4748c240baa82b7e37a635c054dfc19eb0c3b9b563c4536fa18100868cf`；退出0，更新0、query0。本复盘读取已保存指标及快照，不独立重算概率，不称完整实验独立验收。用户本机副本位于 `F:/E2E_VCA_worktrees/side-lpft-issue250-evidence02/summary.json`，checkout外保存。

### 当前结论与下一依赖

两阶段策略、阶段内部恢复、G4完整源/目标接入以及显式CPU暂存已实现；对应定向CPU/GPU测试和三场景真实小步接入完成，HongHu原路径OOM作为负例保留。共实际成功更新40步（原28+新12），不存在“第一组36步全部通过”的结论。原冻结生产文件、主入口、main、Issue249产物和暂停状态没有变更。

Issue250/PR251仍保持open/Draft：还没有完整训练CLI、三方法同预算对照、完整真实训练及精度收益验收，也未完成全仓回归。下一研发项为带首更新前预算一致性核验的正式短测/对照消费者，随后按P4/§19登记头训练、直接微调与LP-FT的共同初始化及同预算比较；内存策略与BN策略作为显式因素，不将复合变化全部归因于LP-FT。任何长训练须先告知总预算、估时和资源安排，再向主聊天移交半小时值守；本轮没有启动长训练或改动定时任务。
