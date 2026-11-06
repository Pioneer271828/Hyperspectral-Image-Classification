# G4完整源仅头与适配器目标工程验收

Issue192 / PR193；对应研究主清单P4 §10.3第4–6项，以及实施计划§19“补机制与独立重复”。在188支持/K输入、190有限目标训练入口之后，接入完整源仅训头和仅训轻量模块两个明确对照。用户要求正常回合连续搭建；本工程独立于184，不改变其冻结生产、配方、预算和验收顺序。

## 实现与执行选择

工作树`F:/E2E_VCA_worktrees/issue-192-light-target`，分支`codex/issue-192-light-target`，基于190交付`934e3d24cc4366fd48e6c5539449b89f09bfb458`，唯一写入人为本聊天助手。新增`g4_light_target.py`、`scripts/train_g4_light_target_short.py`、`tests/test_g4_light_target.py`；原331及184全部Python/config文件未改，无多智能体、无main合并。

复用完整源none/活动M4绑定与初始化、188清洗输入、190路径/文件/上下文守卫和原TargetRuntime三头损失、Adam数值、首次最小fused验证NLL及严格重载。阅读两计划及原设计D4 §6.2后，将未指定参数集合的“轻量模块”明确为项目已有VCA适配器＋三个目标头；这是本次执行选择，不是新外部研究算法、LoRA或第三方实现，不引入新依赖或外部训练协议。

| 对照 | 训练参数 | 优化阶段状态 | 选择阶段 |
|---|---|---|---|
| `source_pretrained_head_only` | 仅三个新目标分类头 | 骨干全部参数和BN缓冲区冻结，骨干eval；仅头训练 | 初始ordered无标签目标BN校准，认证缓存选中支持/验证特征；候选重绑同一骨干，不重复算BN矩 |
| `source_pretrained_adapter_heads` | VCA适配器＋三个新目标头 | 骨干所有模块保持eval，允许声明参数反传；CNN/graph/M4/归一化参数及BN缓冲区冻结 | 同样先初始ordered目标BN校准；适配器改变后候选重新前向并ordered校准BN |

这两种方法的训练/校准/缓存行为不同，须作为完整适配方法比较，不能把未来差异归因于单独的参数数量。M4在容量/残差模型中保持原结构但参数冻结，反向仍可穿过这些操作到VCA。不是M4模块单独微调、BN仿射更新、TTA、零样本或严格空间隔离。

固定none/容量/残差三模式×两方法，12步、5次验证事件（1/4/7/10/12）、2+10恢复和完成0恢复。显式support64/512/2048、K1/5/10、独立模型种子与head seed=model+1000；K+5标签成本不变。入口不提供full/1500、批次循环、研究登记或query。188manifest仍execution-disabled；真实对照另行登记。

参数合同记录完整可训练名称/数量和冻结参数值hash；optimizer仅注册允许参数，每步核对冻结参数、梯度、缓冲区身份与允许参数实际变化。恢复绑定方法、输入和源身份，错误方法/预算/缓存/权限/冻结参数/BN变化/metadata均拒绝。损坏head-only缓存在初始化和成本文件写入之前拒绝。

`adaptation-cost.json`分别记录完整源初始化、初始BN校准、每步优化及不变式核验、候选校准/重绑和验证；新运行实例的恢复会额外记录初始化/校准成本。既有`feature-bank-cost.json`保留缓存前向/复制成本。成本记录不进入确定性权重/optimizer/history身份；完成后直接重载及0更新恢复不改文件。记录排除输入准备守卫、trainer序列化和完成只读验证，不作端到端部署提速主张。

## 实际版本、环境与测试

实际tested/工程生产完整SHA：`37e24c56f13465fef4270698e2e4c3cbcb6ba028`。

| 环境 | checkout/分支 | 结果 |
|---|---|---|
| rmvision Linux，Python3.12.3，Torch2.7.0a0+ecf3bae40a.nv25.02，CUDA12.8，GPU0 RTX4090/driver610.88 | `/home/rmvision/issue192-light-engineering01` / `codex/issue-192-light-target` | CPU5/5，37.322s；GPU5/5，61.108s；完整582项581通过+1既有GPU专用跳过，364.507s；checker369Python86JSON、help通过 |
| LXC Linux，Python3.10.20/Torch2.7.1+cu128，CPU | `/home/lbx/hsi-pioneer271828/checkouts/g4-light-target-issue192-engineering01` / 同分支同SHA | 真实1条件×6初始化/参数角色，63文件不变，52.443937秒；更新0/forward0/BN校准0/query0，无GPU上下文；checker369Python86JSON通过 |

Windows仅编辑/读文件/Git/SSH/SCP。GPU专项启动前free14407MiB/util1%，完整回归前free14351MiB/util0%，均满足free≥12288MiB/util≤60%，GPU0单队列。完整回归保留GPU0可见，供既有source hooks/M4显存统计等用例使用；配对目标长测和本模块专项设CPU。LXC GPU1冻结训练未被修改或占用。CI runner未注册，以上为手动远程验收。

```text
# rmvision，/home/rmvision/venvs/hsi-baseline/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -p test_g4_light_target.py -v
CUDA_VISIBLE_DEVICES=0 LIGHT_TARGET_DEVICE=cuda python -m unittest discover -s tests -p test_g4_light_target.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit 37e24c56f13465fef4270698e2e4c3cbcb6ba028
python scripts/train_g4_light_target_short.py --help
# LXC，相同线程限制，CUDA_VISIBLE_DEVICES为空
python /home/lbx/hsi-pioneer271828/transfer-records/issue192-real-preflight01.py
python scripts/check_repository.py --expected-commit 37e24c56f13465fef4270698e2e4c3cbcb6ba028
```

六组合成网络实际完成完整12与2+10两条路径，恢复采用新运行实例而非仅复用内存对象；另有真实CLI2+10+0（合成support2048/K5）。每个成功专项本模块合成更新156次，不是任何真实数据训练预算。CPU比较model/optimizer/scheduler/history/选中概率精确相等；GPU核对状态恢复及各自三次同路径精确概率重载，不扩称跨路径GPU bitwise一致。

每步声明参数有实际变化，适配器组有正梯度和适配器实际更新，仅头组adapter梯度/变化均0；全部冻结参数和BN缓冲区在每次优化前后不变。五项专项还检查参数名称、源/头初始一致、错误权限/冻结值/BN/方法/metadata/预算、损坏缓存提前拒绝且文件不变、成本阶段/恢复重复校准，以及错误SHA/full选项拒绝。原有GPU专用CLI在完整回归仍单独跳过，其历史182/186 GPU验收保留，本轮不冒充重跑；本模块新CLI已CPU/GPU通过。

## 真实数据仅初始化与权限复核

固定Xuzhou/support512/K1/model64，9训练+45验证=54个目标标签；复用188manifest和184冻结launch中的3种完整源规格。六个方法/模式初始化均与原184对应完整源微调初始state精确相同。独立驱动用字面参数前缀重建预期可训练名称，和实际requires_grad及合同逐项相等，不把调用生产权限函数当独立oracle。

仅头均1,755个可训练参数；VCA适配器＋三头均12,220个，其中适配器10,465。完整模型参数分别为none 61,696、容量70,016、残差69,952，不把可训练参数下降直接等同实测推理加速。CNN、graph、M4及归一化参数冻结，所有骨干模块处于eval。该复核禁止Adam和所有Module forward，未执行真实初始BN校准、缓存、12步更新或概率评分；真实BN/训练链路仍未测，只由合成工程验证，不能写成真实六模型训练完成。

真实回执`runs/g4-light-target-issue192-preflight01/receipt.json` SHA256 `4cf1f7c5c35051a525c54386d038b28ebc799a75c7c4e52d5591967daaf41acc`，外部驱动`transfer-records/issue192-real-preflight01.py`及日志保留在checkout外。输入、完整源权重与回执、launch共63文件前后不变，原331生产干净。只生成验收回执，六个计划训练目录未创建。汇总工程回执`/home/rmvision/issue192-light-engineering01-receipt.json` SHA256 `88bbd254208c26dde8b87b96f5eeaf751050e2c3ce9259581d1c341649d6b219`，同字节副本`docs/architecture/g4-light-target-engineering-results.json`；代码身份、准确日志hash、参数角色和未测范围在该JSON中保留，真实数据、标签张量、权重和运行日志不提交。

## 保留失败、验收边界与下一项

首版`12f46d8f38f9cf3d2f143773e81197d9a4f81bd5` CPU5项中4通过、1错误，24.203s。六角色训练/恢复/冻结检查通过，CLI因source-spec测试夹具位于输出目录祖先而被190隔离守卫正确拒绝；37e24只将夹具移到独立输入目录，不放宽守卫，并增加损坏缓存在任何成本/模型写入前拒绝及BN漂移负例。原`issue192-cpu01.log`保留；原算术、预算、阈值及选模未改。

**有限12步完整源仅头/适配器目标工程验收通过；真实效果尚未验收。**生产tag `module-g4-light-target-engineering-v1`指实际37e24，Issue192独立关闭、PR193待评审，main不合并；主入口仅同步报告/摘要/状态/远程投影，不合未评审源码。

下一阶段保持184全部54训练接受→独立331审计→186真实拟合/逐类/曲线/成本复盘→全54冻结同协议query及预登记判据报告；随后才登记支持/K及本项参数对照的真实实验。空间/传感器身份、波长质量、后续M1/E1/M6/M7/M8及其他计划必测范围继续保留。工程通过不等于训练完成、充分收敛、精度收益或部署加速。

本聊天报告位置：2026-11-06 Issue192完整源仅头与适配器目标工程模块完成报告。该报告作为模块交付的一部分，已交付内容后续不重复。
