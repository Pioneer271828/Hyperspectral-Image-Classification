# E1目标有界训练与恢复工程验收

Issue #202 / PR #203；开发分支 `codex/g4-coverage-target-runtime`，工作树 `F:/E2E_VCA_worktrees/g4-coverage-target-runtime`。实际测试和生产接口完整SHA **`5fd4c81e6c6fbdb9325fe2c82f89305904191968`**。结论只覆盖工程接口，不等于E1完整真实实验或精度收益。

## 计划、用途与实现

对应研究主清单P3 §9.3–9.5、P4 §10.3，实施计划§14 E1和§19的M1/E1真实迁移接入。前置199已提供完整none源初始化、绑定元数据前向和总体BN组件，196提供冻结真实波段视图，190提供独立支持集/K合同。本项使这些已验收组件在有限12步训练、恢复、选模和重载中保持同一身份，不提前进行正式数值实验。

新增 `g4_coverage_target_runtime.py`、`scripts/train_g4_coverage_target_short.py`、`tests/test_g4_coverage_target_runtime.py`，原331已有Python/config文件没有修改或删除。五条件none/continuous/anchor_only/binary/shuffled共享同一冻结波段记录；none无新增参数，四启用条件各登记194参数。来源仍为完整none源，覆盖参数在目标初始化，不称E1预训练源迁移。

复用原三头选中行GEMM、loss、Adam、调度器和first-min fused全观测验证NLL；没有先算全像素分类头再取监督行。none直接委托196原训练算术。启用条件把视图元数据传入VCA/覆盖调制，记录真实B、记录哈希、覆盖梯度/实际变化、有效anchor和回退原因。194是参数数量，不声称194个标量都变化。无可用物理支撑的index-only夹具实际执行一步时覆盖梯度/变化均为0。

保存绑定完整源/目标初态、支持覆盖层、M1策略、E1条件、元数据、12步预算、原始候选状态及总体BN结果。采用现有动态 `calibration_metadata_key` 接入199独立校准策略，未修改原校准合同。只允许工程12步、2+10恢复和完成后只读0步；拒绝full、query、M4及未声明条件/数据/预算变化。

参考复用沿用199的原文和许可核查 `docs/references/g4-coverage-target-reference-review.json`：FiLM原文v2 §2.1–2.2、作者仓库固定fe43ddf8a22b339dcca2efa07091ce9d498955cf及实际CC-BY-NC-4.0范围。复用项目已有E1算子，没有复制外部源码、训练器、协议、权重或添加依赖，没有运行上游。DOFA仍只是相关文献，不是项目VCA来源证明。

## 实际环境、命令与结果

授权LXC `/home/lbx/hsi-pioneer271828`，Python3.10.20、Torch2.7.1+cu128/CUDA12.8，GPU0 RTX4090；本次没有声称重跑rmvision或自动CI。测试checkout `checkouts/g4-coverage-target-runtime-engineering01`，分支同开发分支，干净完整SHA同上。CPU/GPU/full日志分别为 `runs/coverage-runtime-{cpu,gpu,full}01.log`，checker/help同名前缀。

公共环境为 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`，在干净checkout执行：

```bash
CUDA_VISIBLE_DEVICES= COVERAGE_RUNTIME_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_g4_coverage_target_runtime.py -v
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 COVERAGE_RUNTIME_DEVICE=cuda /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -c 'import torch,unittest;torch.use_deterministic_algorithms(True);unittest.main(module="test_g4_coverage_target_runtime",verbosity=2)'
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -v
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/check_repository.py --expected-commit 5fd4c81e6c6fbdb9325fe2c82f89305904191968
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/train_g4_coverage_target_short.py --help
```

CPU **5/5，104.154s**；GPU **5/5，125.945s**；完整 **604项，603通过+1既有GPU专用CLI跳过，715.690s**。该CLI已由既有专项另验，不把本次skip计通过。checker **381Python/92JSON**、help通过。GPU专项门槛24079MiB/0%，全量门槛23352MiB/40%，满足free≥12288MiB、util≤60%。全量保留GPU0可见以满足旧CUDA钩子，不能称纯CPU回归。

合成数据下五条件×scratch/完整源FT均完成12步，以及新实例2+10、完成0和三次选中态精确重载。CPU模型、优化器、调度器、history、RNG及概率精确恢复。none两种初始化×full/alternate75与196对照保持CPU精确状态和轨迹（仅新增覆盖trace字段）。GPU该关闭对照事前声明使用测试专用FixedGraphReference；其他功能测试走原稀疏路径。不能据此宣称原生CUDA独立优化轨迹精确等价已通过；196原生失败记录仍有效。命令行子进程继承CuBLAS环境，不将父进程Torch确定性开关冒称为所有子进程全局开关。

验证相同视图公平性、参数不重复登记、真实覆盖梯度/变化、index-only显式回退、支持/K/模型种子分离、改条件恢复拒绝、raw source篡改、元数据/输出/早读query/越预算拒绝和真实CLI。拒绝测试属于预期通过条件，本版本没有实现测试失败；先前196/199失败仍保留在各自报告和日志。

## 真实输入零更新核验

producer为三场景×两种初始化×五条件×full/alternate75，共60配置；support512/K1、model64/head1064，固定sampling8128。每组12记录，共720条与196已接受记录一致；其中alternate75角色实际构造30模型，完整初始状态/架构/194或0参数绑定与199已接受30初始化一致。只使用既有真实视图和初始化身份，不重新切360景影像或执行网络。

公开prepare每景认证一次无标签观察和已释放的支持/验证覆盖层，检查原view与optimizer方法仍被直接继承。69文件不变，163.030056s；真实forward0、BN0、优化0、query0，无GPU上下文。会读取已释放support/validation标签，不反序列化家族标签或原GT，不能写成完全不读任何标签。

producer：`runs/g4-coverage-runtime-issue202-preflight01/receipt.json`，SHA `8681d3bb00e3ba425001527460a2c7d49741822184c1921c5d622d8a494362a0`。

独立干净checkout `checkouts/g4-coverage-runtime-issue202-independent01`，同5fd完整SHA。不调用202的settings或runtime，使用原190支持合同和196采样器重建底座，再按登记字段独立组合配置，对60组完整配置哈希、720记录和199的30初态绑定逐项比对，70文件不变。没有重新构造模型或复算训练/概率；独立范围是配置组合身份核验。receipt：`runs/g4-coverage-runtime-issue202-independent01/receipt.json`，SHA `fb64d21acb391566fc065cb6274c9b07017a54226b878d1c5bd10887764ff9b9`。实际独立耗时、完整工程receipt身份见同字节 `docs/architecture/g4-coverage-runtime-engineering-results.json`。

## 限制、验收与下一依赖

五条件真实12步、完整训练、源侧E1预训练和精度/速度结论均未验收；真实工作只完成配置和初始化。三场景波长仍approximate、缺实测FWHM/SRF；整景传导图和归一化不等于空间隔离或未见传感器。更多模型/支持种子、K、连续缺口/新子集真实曲线、M4四组交互及其余计划范围继续保留。

GitHub连接器/TLS和不确定注册失败没有冒称成功；不确定POST后先GET确认NOT_FOUND再恢复登记，未重复建立Issue/PR。资源198续接由用户另一会话负责；本项未停启184控制器、未运行其全批次审计或query，没有改变81000预算或冻结331代码/输入。

本项结论为 **E1目标有限12步训练、恢复、公平控制与身份保护工程通过**。下一真实阶段仍是184全54接受→独立审计→186真实拟合/逐类/成本复盘→全54冻结ordered query及预登记结论；之后再另登记支持/K、参数范围和M1/E1真实同条件对照。用户要求继续搭建时，只推进不依赖未完成真实结论的独立工程，不能用工程通过代替效果。

候选源码不合入primary/main；主入口只同步报告、证据、current-status和远程投影。current-status记录本聊天的模块交付报告，避免重复发送。最终交付SHA、标签和独立交付检查随交付台账登记；实际测试SHA始终为5fd，不以文档提交替代。

工程receipt同字节SHA为329b36e4dad04e674eecdbdc6414348ef5ec60c4a76119de70890dc80aeeaa88；独立配置核验实际23.993816s。184本次只读检查点51/54、76500/81000，另有3例4500步；198合法边界交接的原控制器暂停不当作训练失败。该检查点是归档证据，实时进度以远程材料为准。
