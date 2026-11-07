# G4 M8位置门控末端融合完整源目标组件工程验收

Issue232 / PR233；研究主清单§2.3与原末端64维固定融合合同，实施§12/§18.3 F/§18.4四组B0、R、F、R+F。实际tested **2bce85d1258c7908086e0f6147158e883a3c7b89**；分支 `codex/g4-position-fusion-target`，工作树 `F:/E2E_VCA_worktrees/g4-position-fusion-target`。本次验收初始化/算子/BN/专属重载组件；“完整源”指复用已接受完整none源，未执行真实目标训练、完整四组或query。

## 用途、实现与复用

回读两主计划相应章节、230核报告和214固定融合组件。复用230实际a6ef1f93adbbf68b62c24e4fd9ae78834f311a95的独立位置标量核，消费其完整passed工程回执，并逐文件核原 **357个Python/config文件** 同hash。214实际045a35ae431b75149367256a476b40b2a3777440提供完整none源目标初始化、原三头概率损失与ordered BN；222专属loader模式最小适配。原论文/固定作者源码/MIT核查见 `docs/references/m8-position-gate-reference-review.json`，本项无新外部源码复制、执行、依赖或许可主张。既有公共model/config/loader/损失/采样/预算文件不改，先前交付文档仅作为当前台账上下文接纳。

新增 `g4_position_fusion_target.py`、`scripts/audit_g4_position_fusion_initialization.py`、七项测试和内部复用清单。fixed/position_gate×scratch/完整none源FT共四角色。fixed不添加算子或状态，保留原骨干类型；启用在本地实例安装骨干子类，只覆盖末端fuse_features，继承原forward_features、三头predict_features/fuse_and_predict和230核心forward。M4/M6/M7/E1关闭，静态全目标无标签图、DomainBN、新头及共享参数权限不变。

位置门直接消费原CNN/Graph各64维特征，每位置一个sigmoid凸组合系数，**129新增训练参数、另8字节版本buffer**。无额外模态投影或新损失。初始权重0/偏置logit(0.8)，构造无CPU/使用GPU RNG变化；全部原共享、新头和目标域张量逐位匹配旧214。启用初态只承诺数值接近固定融合，不承诺逐位相同。门不解释为校准置信度或物理可靠度。

230独立核的字典extra_state不直接进入原G4仅张量state_id：本地Bound类使用nn.Module默认get/set_extra_state，整网状态保持全Tensor；完整核配置强制保存在G4 binding/sidecar。实际类、方法、版本、配置、schema、权限、源及核文件身份均验证，专属full-model协议不开放独立核payload。旧通用loader对启用态因多出的position_fusion权重/marker严格拒绝，避免静默恢复成固定融合。

专属loader核文件对与显式binding，按原架构重建模型并安装本地门，再严格载入有限FP32张量，核原完整嵌套BN回执、观察/源state身份及eval状态；加载后再次核文件对。该组件不开放12/full/campaign/query runtime。

## 实际环境、命令和结果

授权LXC `/home/lbx/hsi-pioneer271828/checkouts/issue232-position-target-engineering01`，实际分支 `codex/issue232-final01`，干净完整tested SHA同上。Python3.10.20、Torch2.7.1+cu128、CUDA12.8、GPU0 RTX4090；Windows仅编辑/读文件/Git/SSH/SCP。未声明rmvision复跑或CI自动执行。

公共环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`；Python为 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。

```bash
CUDA_VISIBLE_DEVICES= POSITION_TARGET_DEVICE=cpu python -m unittest discover -s tests -p test_g4_position_fusion_target.py -v
CUDA_VISIBLE_DEVICES=0 POSITION_TARGET_DEVICE=cuda python -m unittest discover -s tests -p test_g4_position_fusion_target.py -v
CUDA_VISIBLE_DEVICES=0 POSITION_FUSION_DEVICE=cpu POSITION_TARGET_DEVICE=cpu MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu EDGE_TARGET_DEVICE=cpu EDGE_RUNTIME_DEVICE=cpu FUSION_TARGET_DEVICE=cpu FUSION_RUNTIME_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit 2bce85d1258c7908086e0f6147158e883a3c7b89
python scripts/audit_g4_position_fusion_initialization.py --help
```

CPU **7/7，32.991s**，GPU **7/7，34.875s**；分支全量 **658项657通过+1既有GPU专用跳过，1107.966s**。checker **404Python105JSON**、help通过。GPU专项/全量各核GPU0 free≥12288MiB/util≤60%，`runs/issue232-{gpu,full}01-gate.txt`保存；全量GPU可见供既有钩子，不称纯CPU。全量包含既有合成1500步审计用例，未新增真实训练批次。精确执行脚本 `issue232-validation01.sh`在checkout外绑定hash。

四角色原状态/新头/DomainBN/权限/RNG匹配；fixed两角色对214原前向和总体BN逐位相同。启用后CNN/Graph末层表示及两个辅助头对固定模型保持原值，手工门公式对应真正末端融合位置，改变门只改变融合预测；logits→softmax与原13项概率损失对应，拒绝改辅助权重。

四角色各合成3更新，优化器全部参数唯一登记，CNN/Graph/启用门/三个头的有限正组梯度及实际变化通过，不声称全部标量每步改变。仅5层10个总体mean/var缓冲允许校准变化，其余权重、affine、计数器、观察与源state不变。四角色各3次选中态重载共 **12次/专项**精确；单个启用角色另用CPU FP64独立总体矩，再转原FP32路径，rtol1e-5/atol1e-6不变，不扩称全角色独立概率或全网FP64审计。GPU核功能和ordered重载，独立原生CUDA优化轨迹等价未接受。

版本/实际forward/配置/权限/错误源哈希、带标签观察、错误文件对在Tensor解码前、嵌套原BN回执及输出复用/full/query/错误condition/modelseed拒绝通过。实际零前向CLI另用support2048/K5，其他modelseed初始化为合成接口检查，不称真实多支持/种子实验。

## 真实输入初始化与独立状态复核

真实三景×两角色×两条件 **12初始化**，model64/head1064、support512/K1。全部原状态/架构/权限与214已接受的六个default基础身份对应，启用仅129参数与marker；核230完整receipt及357代码身份后执行。**95保护文件不变，19.845399s**。Module调用和Adam构造禁止，真实NN/BN/优化/query0、无GPU上下文；支持/验证释放覆盖只用于身份认证，家族标签/原GT未反序列化。

独立进程仅旧214工厂六次，手工零权重、log(4)偏置与int64版本Tensor，重建 **12个state/config/schema/权限**，不调用232新factory或230核构造。**96文件不变，3.486544s**。同干净checkout的不同进程，不冒称另一个独立checkout、概率/BN/训练审计；未读取支持/验证覆盖层。

工程原件 `runs/g4-position-fusion-target-issue232-engineering01/receipt.json`，同字节仓库副本 `docs/architecture/g4-position-fusion-target-engineering-results.json`，SHA **6d5af4a7ecdd14101467d11e160ae5d0a92b00a91f7fb9ac7c97ad3dc1e37b4e**；
producer `runs/g4-position-fusion-target-issue232-preflight01/receipt.json` SHA **00c55dd8fedf604953a5faf7d9e86bff2039620d3bc813417c8f6c2a47120f7a**；
独立 `runs/g4-position-fusion-target-issue232-independent01/receipt.json` SHA **7994fc9bb801130845fd4cfce68ad0f46313d221826d747d3eb30ea0e77ad247**。
核依赖receipt SHA **2f99fc7bc1579121851bd2a9acc3dba3c2ec58197ffc7ddb18d77fd551985e39**。日志 `runs/issue232-{cpu,gpu,full,checker,help}01.log`，driver/产物均在checkout外。

## 失败记录、限制及下一项

本实现数值测试失败0。Issue首次连接器创建失败，权威GET232返回404后再创建成功，无盲目POST重复；TLS推送失败后相同refs恢复。带入的已交付旧报告三处末尾空行被git diff --check提示，保持既有报告字节，未改旧工程证据；原始提示保留。未放宽数值、标签、选模或冻结守卫。

验收结论：**完整none源目标的初始化/算子/BN/专属重载组件工程通过**。不是M8完整四组或精度/成本/容量/物理可靠度结论。下一 **Issue234** 有界12、新实例2+10/完成0、显式门梯度/变化trace及完整绑定恢复；真实配置证明须消费本项passed工程receipt。

真实B0/R/F/R+F、相近容量/成本、支持/K/空间/传感器/M4组合保留，按184全54接受→独立331全批次审计+198资源终验→186真实曲线/逐类/拟合/成本复盘→原331全冻结ordered query→204预登记结论后登记新数值实验。当前184由新608030在GPU1/3/7续跑，原331代码/预算不动；资源198报告按原235文档精确接纳，不合候选源码或main。主入口只同步报告/证据/参考/current-status与远程投影，简报后继续。
