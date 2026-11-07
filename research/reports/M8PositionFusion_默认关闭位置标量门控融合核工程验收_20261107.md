# M8 默认关闭的位置标量门控融合核工程验收

Issue230 / PR231；对应主清单§2.3的门控/置信度/物理可靠度区别及原末端64维、0.8/0.2合同，实施§12/§18.3 F/§18.4 B0、R、F、R+F。实际tested **a6ef1f93adbbf68b62c24e4fd9ae78834f311a95**，分支 `codex/position-gated-fusion`，工作树 `F:/E2E_VCA_worktrees/position-gated-fusion`。本次为独立核及原分类头/损失合成接入工程；目标整网、BN、训练runtime与完整真实四组均未验收。

## 用途、文献及复用边界

已验214/216只覆盖固定系数/辅助权重，不重复。本次新增顶层 `position_gated_fusion.py`、`tests/test_position_gated_fusion.py` 和参考核查JSON。原 `model/`、配置、损失、骨干、CLI和公共checkpoint loader完全不改。实施§12列举branch_fusion.py，但整个model目录已冻结，故用顶层伴随实现，后续显式adapter另项验收；没有放宽冻结门槛。

阅读[Arevalo等原文1702.01992v1](https://arxiv.org/abs/1702.01992v1)引言/相关方法、§3.1公式和图2、训练/结果背景§5、结论§6及脚注5作者代码入口；17页并非全部附录/参考逐项读完。论文用门控制两种表示的组合，在合成与电影多模态分类验证；不能推出HSI收益。作者[源码固定95226b0d265743e7a9330549bc35b95fcb60e1dd](https://github.com/johnarevalo/gmu-mmimdb/blob/95226b0d265743e7a9330549bc35b95fcb60e1dd/model.py)已读GatedBimodal/GatedClassifier、README与LICENSE。根MIT版权2020 John Arevalo；LICENSE Git blob1eda536f416828fe16535351c8e9c2e8130ceb19，model.py blob6f9f68eda5f714a25899196697e28d0e781a2118；跟踪树无嵌套LICENSE/COPYING或submodule。该核查不覆盖外部框架及链接数据/权重许可。

无外部源码复制。独立PyTorch实现仅借鉴sigmoid凸组合思想：作者代码有额外模态投影/tanh及通道门，本项目直接消费既有CNN/Graph最终特征，位置标量门并加偏置。不是完整GMU复现。Theano/Fuel/Blocks未安装/执行，MMIMDb与外部权重未下载；固定源码只读，不执行其README命令。详细核查 `docs/references/m8-position-gate-reference-review.json`。

## 实际实现

默认 `enabled=False`、0训练参数，对合法有限FP32/FP64同形[N,C]精确调用旧fixed_fusion，固定系数0/1仍为原端点。启用时每位置由两路特征拼接产生一个sigmoid标量，再按该权重凸组合；C64共 **129可训练参数**，零权重+logit(0.8)偏置直接分配，不消耗构造RNG。启用初态只承诺数值接近原固定融合，不承诺逐位等价。

无跨位置统计、额外投影、归一化、教师、损失或选模变化。版本buffer8字节，拒绝非法形状、FP16/整数/混合dtype/device、非有限输入/参数/score以及未知版本。空N返回合法空特征与[N,1]门。门是混合系数，不称校准置信度、因果重要性或物理观测可靠度；融合计算与检查同步的性能成本尚未测量。

自有配置/状态配对包含严格policy/version/通道/开关/alpha及必需marker，支持weights_only序列化和显式设备/dtype恢复。全部键/配置/形状/版本/值先核再复制，拒绝strict=False/assign=True，错误状态不部分覆盖已存参数。此为独立核协议；其extra_state包含配置字典，不能直接当作原G4仅张量state_id的整网权重。Issue232须用专属适配、绑定外置配置并保持整网张量状态，不宣称公共checkpoint格式已迁移。

## 实际环境、命令和测试

授权LXC `/home/lbx/hsi-pioneer271828/checkouts/issue230-position-fusion-engineering02`，分支 `codex/issue230-final02`，干净完整tested SHA同上。Python3.10.20、Torch2.7.1+cu128、CUDA12.8、GPU0 RTX4090；Windows仅编辑/读文件/Git/SSH/SCP。无本轮rmvision复跑或CI执行声明。

公共环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`；Python为 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。

```bash
CUDA_VISIBLE_DEVICES= POSITION_FUSION_DEVICE=cpu python -m unittest discover -s tests -p test_position_gated_fusion.py -v
CUDA_VISIBLE_DEVICES=0 POSITION_FUSION_DEVICE=cuda python -m unittest discover -s tests -p test_position_gated_fusion.py -v
CUDA_VISIBLE_DEVICES=0 POSITION_FUSION_DEVICE=cpu MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu EDGE_TARGET_DEVICE=cpu EDGE_RUNTIME_DEVICE=cpu FUSION_TARGET_DEVICE=cpu FUSION_RUNTIME_DEVICE=cpu NORMALIZATION_TARGET_DEVICE=cpu NORMALIZATION_RUNTIME_DEVICE=cpu BRANCH_LOCAL_TARGET_DEVICE=cpu BRANCH_LOCAL_RUNTIME_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit a6ef1f93adbbf68b62c24e4fd9ae78834f311a95
```

CPU **6/6，1.073s**，GPU **6/6，2.335s**；当前分支全量 **533/533，214.298s，跳过0**；checker **322Python113JSON**通过。基于primary68c58的此分支未合其他候选PR，533不冒称所有未合分支联合回归。GPU专项/全量分别核free≥12288MiB/util≤60%，`runs/issue230-{gpu,full}02-gate.txt`保留；全量可见GPU供既有钩子，不称纯CPU。精确命令在checkout外 `issue230-validation02.sh`绑定hash。

独立NumPy FP64公式、所有输入/权重/偏置解析导数以1e-12比较，中央有限差分8位小数通过；不把同一PyTorch实现复制当oracle。凸界、饱和近端点、同输入门梯度0、交换特征与反转score、位置置换/分块、空输入和错误拒绝通过。固定关闭FP32/64输出与输入梯度对旧函数逐位，构造CPU及使用GPU RNG无变化。

原predict_features三头logits/probability、原13项掩码概率损失保持，合成3步SGD/每次专项调用中门与三头实际参数变化；无标签行监督特征梯度0、辅助权重0时两辅助头梯度0。只训练合成特征上的门和分类头，不称原完整训练器或骨干训练通过。GPU验核数学/功能和精确重载，不接受独立原生CUDA优化轨迹等价。

## 证据、失败、限制及下一项

工程原件 `runs/m8-position-fusion-issue230-engineering01/receipt.json`，同字节仓库副本 `docs/architecture/m8-position-fusion-engineering-results.json`，SHA **2f99fc7bc1579121851bd2a9acc3dba3c2ec58197ffc7ddb18d77fd551985e39**。日志 `runs/issue230-{cpu,gpu,full,checker}02.log`，首次CPU日志01及两个validation脚本/receipt driver均保留在checkout外。真实输入/初始化/NN/BN/优化/query **全部0**。

初始 **7cbfe5560a7642bc1db821b895de36e7300dbd68** CPU6/6、1.047s；checker因新增文件位于冻结model目录拒绝，在GPU/full启动前停止。最终仅新文件移动顶层、import和参考路径修正，旧文件/门槛未动；首版clone/log/交互stderr拒绝记录保留，不称数值失败。最终无数值测试失败。

验收结论：**独立核与原三头/损失的合成工程通过**；目标整网/BN/runtime、真实精度、速度、容量、充分收敛、校准可靠度及全四组未测。下一 **Issue232** 完整none源目标组件、strict绑定/BN/专属loader；真实零前向证明须先消费本项完整工程receipt。新真实数值实验仍按184全54接受→独立331审计→186真实曲线/逐类/成本复盘→原331全冻结query→204预登记结论后登记，不接管198、不改冻结生产、不合main。主入口只同步报告/证据/参考/current-status及远程投影，简报后继续搭建。
