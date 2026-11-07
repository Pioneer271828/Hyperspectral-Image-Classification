# G4目标真实波段子集：有限工程验收

Issue #196 / PR #197；分支 `codex/issue-196-observation-target`，独立工作树 `F:/E2E_VCA_worktrees/issue-196-observation-target`。实际最终测试版本 **`b32ac0af1559f59fab1a39bd62d9e849c48d02df`**。本报告属于工程接入与真实输入零更新核验，不是完整实验或精度收益交付。

## 对应计划、用途和实现

对应研究主清单P4 §10.1–10.3、实施计划M1 §5及§19的M1/E1真实迁移接入。复用已验收188支持/K上下文、190完整none源初始化/有界目标运行时和M1-J1采样/重放；194的波长质量和结论边界保持。历史M1-J2/E1负结果不删除。本项只接M1，下一依赖是独立E1目标覆盖度元数据/调制接入。

新增三个文件：`g4_observation_target.py`、`scripts/train_g4_observation_target_short.py`、`tests/test_g4_observation_target.py`。仅允许none架构下scratch或完整源全量微调，full和alternate75两种工程视图。模型、支持集、波段采样种子分别记录；12条实际波段索引、元数据、上下文、输入、源和预算身份在更新前冻结。局部NumPy RNG采样不读取标签、不推进训练RNG；恢复按全局cursor重放。

子集对整景选择已经归一化的光谱列，实际改变B，同步波长/FWHM/带号，保留两个坐标通道、Q/A及支持/验证标签。复用原三头损失、Adam、调度、12步预算、完整观测ordered目标BN校准和首次最小fused验证NLL选模。新增拒绝“完整元数据单位可判别，但某个实际子集自动单位含糊”的更新前检查。元数据文件、源spec和数据文件均哈希绑定，输出目录与各输入隔离；策略进入identity/训练状态/checkpoint，完成恢复为只读0更新。没有full/campaign/query入口，也没有M4/E1/轻量控制的组合开关。

复用来源为项目已验收的`loadData/observation_sampling.py`、`joint_observation.py`和目标运行时。已回读`docs/guides/joint-observation-m1j1.md`和`docs/references/e1-reference-review.json`中的原文/作者代码/固定SHA/许可核查；FiLM/DOFA是相关背景，不是本采样器的外部实现。不新增外部源码、依赖、训练器、权重或数据协议，不声称运行了上游实验。

## 最终验收环境与命令

最终Linux环境为授权LXC `/home/lbx/hsi-pioneer271828`，Python `envs/hsi/bin/python`。干净checkout `checkouts/g4-observation-target-issue196-engineering03`，完整SHA同上；独立消费者/重算checkout `checkouts/g4-observation-target-issue196-independent01` 同SHA。包版本、GPU型号及日志哈希见同字节证据JSON `docs/architecture/g4-observation-target-engineering-results.json`。

公共环境：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`。

```bash
CUDA_VISIBLE_DEVICES= OBS_TARGET_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_g4_observation_target.py -v
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 OBS_TARGET_DEVICE=cuda /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -c 'import torch,unittest;torch.use_deterministic_algorithms(True);unittest.main(module="test_g4_observation_target",verbosity=2)'
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -v
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/check_repository.py --expected-commit b32ac0af1559f59fab1a39bd62d9e849c48d02df
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/train_g4_observation_target_short.py --help
```

CPU专项5/5，58.376s；GPU专项5/5，67.504s；最终全593项592通过+1既有GPU专用CLI跳过，571.491s。checker375个Python/89个JSON及help通过。Python3.10.20、Torch2.7.1+cu128/CUDA12.8，GPU0 RTX4090/driver595.71.05。全量保留GPU0可见，因为旧source/M4探针会访问CUDA，不能称为纯CPU回归。GPU启动前GPU0空闲19898MiB、利用率17%，符合12288MiB/60%门槛。

CPU完整观测控制与190的model/optimizer/scheduler/history/RNG/概率逐项精确一致。GPU完整控制使用**测试专用小图稠密固定归约参考**，相同精确断言通过；没有将此参考接入生产代码。其余GPU功能测试走原稀疏训练路径；CLI子进程继承CuBLAS环境，但不继承父进程的Torch全局确定性开关。合成实际训练覆盖两种初始化的12步与新实例2+10、三次精确所选模型重载、完成0更新、真实减少B的VCA调用、梯度和图参数更新、记录/单位/输入/输出/预算/query拒绝及CLI。未声称GPU独立优化轨迹或GPU断点轨迹逐参数精确。

## 原生CUDA负结果与修订范围

首版`e74b310528c0e6b6c846bd30777bcc716046ca5b`在rmvision CPU5通过40.585s、全593项592通过+1既有GPU专用跳过447.330s；原生GPU5中4通过、1完整控制参数哈希失败72.888s。外部诊断的原入口自身重复也不同：参数最大差`1.325458288192749e-4`、概率最大差`5.939602851867676e-5`；原入口与新full视图比较出现1个验证CNN分类变化，不能按相同OA或小误差接受为原生等价。两侧均选12不改变此结论。

仅增加CuBLAS/Torch确定性开关的LXC `b5bf3929e795ec8c8cfe0d24a5468a4d8262556a`仍是4通过+1严格完整控制失败63.980s；CPU5通过60.360s。固定图归约参考探针严格通过12.348s后，最终版本将该参考明确限制在完整控制的合成数值测试。原生跨运行等价仍记为失败/已知限制，未放宽原概率、分类或重载门槛，没有修改184或任何生产默认训练算术。未来M1真实实验须在登记中明确其数值执行合同，不能将本参考验收写成默认原生CUDA轨迹等价。

rmvision阶段日志在`/home/rmvision/issue196-{cpu01,gpu01,full01}.log`，诊断在`/home/rmvision/issue196-native-control-diagnostic01/receipt.json`及`issue196-native-diagnostic01.log`。随后rmvision SSH banner超时，未伪称在该主机重跑最终版本；最终验收改用已授权LXC。第一次增量bundle缺d8文档祖先被拒绝，改用实际4f祖先打包；一次SCP连接关闭保留，不使用未成功传输文件。独立复核01脚本手抄producer哈希多一位，在任何输入重算前拒绝，02脚本从原文件核对后修正；原脚本/日志保留，不改原产物。首次证据打包命令缺PYTHONPATH而在导入时失败、未写receipt，补显式环境后成功；第一次提前取不存在的receipt也未被使用。

## 真实输入与独立零更新复核

沿用188的完整27支持/K上下文和184被冻结的none初始化；不读取原GT/query，不做真实网络forward、BN校准、优化或新图构建。CPU访问断言拒绝Adam和所有Module forward，并确认没有GPU上下文。

首轮真实producer在干净`e74b310`执行，外部`runs/g4-observation-target-issue196-preflight01/receipt.json`：27上下文、54策略、648记录、36个实际整景视图、6个scratch/完整源初始态与184一致，65文件不变，368.940980s。实际full/75%波段数PaviaU103/78、Xuzhou436/327、HongHu270/203；三者波长仍approximate、FWHM仍缺失，数组局部带号不升级为仪器原始索引。

最终`b32ac0a`独立消费者重新验证全部54策略（含新增逐子集单位拒绝），另用独立NumPy选样和通道索引重建648记录及36整景视图，不用producer采样/重放/视图函数作为唯一oracle：27预算/上下文、波长/光谱/坐标/图/标签身份和66受保护文件通过，145.149790s。原6初始化依赖函数与首轮一致，本次独立重算不冒称又执行了6个新初始化。原producer回执不回写，修订后消费者和独立结果单独保留在`runs/g4-observation-target-issue196-independent01/receipt.json`。

工程合成训练预算与真实实验预算分开：本模块真实训练0、真实校准0、query0，54份策略不是54个已训练模型。当前输入有完整场景归一化及原完整观测无标签图上下文，所以仅支持受控观测增强接入，不是严格空间隔离、未见传感器或真实物理标定证明；没有精度、收敛或速度收益结论。

## 验收结论与后续

结论：有限M1目标工程接入和真实输入零更新绑定验收；原生CUDA独立轨迹等价失败如上保留，不纳入已通过范围。E1覆盖度目标接入、完整真实M1/E1对照和独立空间/支持集/传感器重复仍未完成。184仍由原实验及资源接管现场推进；本模块未停启其控制器、未运行全批次审计或query。后续先完成E1独立工程依赖；真实数值实验仍遵守184接受→独立审计→真实复盘→全冻结ordered query的既定顺序。

最终同字节工程receipt SHA256为`813424ffb70bd932915b9c5ed2a5d26a7684405c9ff75a922854f6dcab23d089`；真实producer SHA为`76895299ebd914806c4c33f9c5e4f448902edfa32b8e2f50301337ef7c1f6a18`，最终独立receipt SHA为`98e7390ffb3c8b36db9ffd5395d282be749e153cae3b91eee0a53b3540aa799e`。以原文件哈希和提交JSON为准。最终原始日志在LXC `runs/issue196-{cpu,gpu,full,checker,help}03.log`。

主入口只同步本报告、证据JSON、current-status及远程生成投影，不引入候选Python源码；PR供独立评审，main未合并。模块聊天交付位置记录在current-status，不重复交付已完成模块。
