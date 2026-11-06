# G4 M7仅归一化目标组件工程验收

Issue222 / PR223；对应主清单§7.1/§7.2及模块计划§11、§18/§19的static非归一化图“仅行归一化”对照。实际tested SHA **5f85e2220ce3ce357d99e6faaccca51676ce8502**，分支 `codex/normalized-edge-target`，工作树 `F:/E2E_VCA_worktrees/normalized-edge-target`。组件工程通过；真实训练、精度及成本收益未测。

## 用途和实际实现

新增 `g4_normalized_edge_target.py`、`scripts/audit_g4_normalized_edge_initialization.py`、七项测试及内部复用清单。复用331原 `GraphEdgeUpdater` 的正候选先验、孤立点自环及1e-12分母下限，206 matching none/gated目标初始化和原ordered FP32总体BN。未复制新外部源码/增加依赖，无外部许可授权新主张；原331/206/210/212文件不改。

| 条件 | 匹配M6新增参数 | 仅归一化新增参数 | 必需状态扩展 |
| --- | ---: | ---: | ---: |
| none | 0 | 0 | 0 |
| row_normalized | 0 | 0 | 8字节版本buffer |
| gated | 58498 | 0 | 0 |
| gated_row_normalized | 58498 | 0 | 8字节版本buffer |

四条件×scratch/完整none源FT共八角色。原共享参数、fresh三头/DomainBN、参数权限及RNG对匹配M6保持一致；不称已有M6源预训练。stage1仍用原A，原M6同时更新后、graph stage2前应用先验归一化，不学习评分、不新增候选远程边、不剪枝。保留原极小质量阻尼：3e-15行质量经1e-12下限后行和0.003，不能写所有正行严格和为1。

显式绑定源码/源/配置、算子类/实际forward、channels/mode/floor/version、完整state schema及参数权限。原export kwargs说明learned M7 off，自定义绑定另声明控制；必需版本buffer使旧通用loader拒绝启用检查点，避免静默漏掉归一化。自有成对哈希loader先核文件、严格载入算子状态及原BN回执；组件无有界训练/full/campaign/query入口，后续224另接入。

## 实际环境、测试命令和结果

授权LXC `/home/lbx/hsi-pioneer271828/checkouts/issue222-normalization-engineering01`，干净同tested SHA。Python3.10.20、Torch2.7.1+cu128、CUDA12.8、GPU0 RTX4090；Windows只编辑/读文件/Git/SSH/SCP。未声称rmvision复跑或CI自动执行。

公共环境 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`；Python为 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。

```bash
CUDA_VISIBLE_DEVICES= NORMALIZATION_TARGET_DEVICE=cpu python -m unittest discover -s tests -p test_g4_normalized_edge_target.py -v
CUDA_VISIBLE_DEVICES=0 NORMALIZATION_TARGET_DEVICE=cuda python -m unittest discover -s tests -p test_g4_normalized_edge_target.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu EDGE_TARGET_DEVICE=cpu EDGE_RUNTIME_DEVICE=cpu FUSION_TARGET_DEVICE=cpu FUSION_RUNTIME_DEVICE=cpu NORMALIZATION_TARGET_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit 5f85e2220ce3ce357d99e6faaccca51676ce8502
python scripts/audit_g4_normalized_edge_initialization.py --help
```

CPU **7/7，34.982s**、GPU **7/7，39.268s**、完整 **658项657通过+1既有GPU专用跳过，1108.333s**；checker **405Python106JSON**及help通过。GPU专项与全量各自核GPU0 free≥12288MiB/util≤60%，实际记录 `runs/issue222-{gpu,full}01-gate.txt`；全量保留GPU可见供旧钩子，不称纯CPU。

CPU FP64独立密集先验/梯度手算及原GraphEdgeUpdater mix0精确对应；支持/孤立点/极小行语义、关闭对206与身份图启用等价、实际stage1/raw和stage2/normalized位置通过。八角色各合成3更新，实际三头/图和启用M6正组梯度及参数变化，原ordered BN保护和各3精确重载，共24/专项通过。另一个gated_row_normalized角色用CPU FP64总体矩参考后转原FP32规范化；不称全网FP64或全部角色独立概率复算。

错误算子接口、负/非有限图、dtype、floor/mode/version、源/参数权限/标签、文件对/种子/条件/full/query/输出拒绝通过。旧通用loader对启用四角色严格拒绝；GPU验功能和ordered精确重载，不接受独立原生CUDA优化轨迹等价。

## 真实输入零更新及独立重建

三景×两角色×四条件 **24组初始化**，model64/head1064、support512/K1，匹配206既有none/gated的12个基础身份。剥离版本buffer后全部state、架构、trainable合同与旧接受行一致；78保护文件不变，**22.848739s**。读取已释放支持/验证覆盖用于认证，不反序列化家族标签或原GT。Module调用和Adam构造明确禁止，真实NN/BN/优化/query0，无GPU上下文。

独立进程仅用旧206工厂12次和手工8字节版本buffer，重建24个state/schema/架构/参数合同，未调用222新工厂或新算子前向；79文件不变，**4.046700s**。同一干净tested checkout执行独立重建逻辑，并非另一个独立checkout，也不冒充真实BN/训练/概率审计。

工程原件 `runs/g4-normalized-edge-target-issue222-engineering01/receipt.json`，同字节副本 `docs/architecture/g4-normalized-edge-target-engineering-results.json`，SHA **2930aa218b2dac31aa26fd335e648a1317c54cdc7712906c2189b12d89f185dd**。真实producer `runs/g4-normalized-edge-target-issue222-preflight01/receipt.json` SHA **b4683f0963e7af93ed2f514cca2c902d6aa7649e76385ee23f429473664aba07**；独立 `runs/g4-normalized-edge-target-issue222-independent01/receipt.json` SHA **726d8c8250bb0d3a76fb13b5ba19675b82b01b0b5847bef4f41761ad96e4c6db**。日志 `runs/issue222-{cpu,gpu,full,checker,help}01.log`，真实driver及日志也在checkout外。

## 失败、限制和下一依赖

实现版本数值测试失败0。首次独立入口人工转录producer哈希多一个8，在读取producer/模型前断言拒绝；保留 `runs/issue222-independent01.log`。之后从SCP同字节原回执计算原哈希，再运行独立核验通过，未改冻结源或门槛；更正日志 `runs/issue222-independent02.log`。注册/发布的连接器/TLS失败、权威GET后再创建的记录保留，无盲目POST重试。

工程通过不等于真实效果、训练充分性、精度或提速。仅归一化与学习M7/相近容量的完整对照、支持种子/K、空间/传感器、M4组合仍必测；本次真实初始化只model64、support512/K1，CLI合成另验证K5/2048与其他modelseed，不称完整重复覆盖。

下一 **Issue224 / PR225** 另接入有界12、2+10严格恢复、完成0和M6分方向梯度/实际变化，组件源码继续冻结。新真实实验仍等待184全54接受→独立331审计→186真实复盘→原331全冻结ordered query→204预登记结论；不接管198资源交接，不改184生产。主入口只同步报告、证据、参考清单/current-status及远程投影，main未合并。依用户最新要求简报后连续推进。

