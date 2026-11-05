# G4完整54训练复盘导出工程验收

本项对应研究主清单P4 §10.3、实施计划§19和`docs/guides/post-training-review.md`的曲线、逐类、模块实际参与和成本复盘要求。Issue186 / PR187是正在运行的Issue184实验的复盘支持工程，不是另一项数值实验。用户2026-11-06明确要求在正常工作回合继续模块搭建；定时任务仍只值守原长训练，每30分钟检查一次。

## 用途、实现与冻结边界

独立工作树`F:/E2E_VCA_worktrees/issue-186-matched-target-review`、分支`codex/issue-186-matched-target-review`，唯一写入负责人为本聊天助手，无多智能体。基于已验收生产331及当前实验文档分支7b6c401f894750bae97d99b9eb1418f54d3a0ada新增三个文件：

- `g4_matched_target_review.py`：全54接受/独立审计绑定、曲线推导、三头逐类与成本汇总、PNG/CSV/summary导出。
- `research/reproduction/m4g4b/matched_target_report.py`：干净Linux/完整expected SHA的显式入口。
- `tests/test_g4_matched_target_review.py`：7项专项，覆盖负例、实际合成网络材料和导出。

复用项目已验收的`none_source_report.head_record`约定及源复盘绘图方式，独立适配目标首次最小验证fused NLL选模；没有引入新的外部研究算法或下载执行上游代码。原331的全部Python/config文件保持逐文件不变，模型、损失、优化器、采样、标签权限、初始化、预算、数值门槛、旧审计及query接口均未修改。生产checkout与训练输入未写入。

导出器必须先通过原`full_acceptance`完整54/81000接受门槛，再核独立审计policy、producer/consumer源码、launch/acceptance/protected-manifest哈希及每例完整证据相等；不接受缺例、重复例、不同选步/召回/重载、query访问或非零更新。额外文件清单不得覆盖接受材料已认证的哈希。输出要求全新同级目录，避开训练、源、标签、观察及审计目录；开始和结束检查全部输入，未完成时不发布passed summary。

每例导出总损失和30步滑动均值、训练/验证融合OA、三头验证OA、fused NLL、实际LR、首次最小NLL选步、三头逐类support/recall、零召回类与无支持类（NA）、末段实际步差、模块正梯度步数与末态参数变化。18个场景×方法×模式组分别保留三个模型种子的均值与样本SD。模块参与与参数变化复用事前独立审计证据；非选中事件只读已认证记录，不声称逐事件独立概率复算。

## 实际版本、环境与验收

实际最终源码/测试SHA：`ad60ce0a863aad1991ee6c84d547c3efd995290f`。授权Linux容器rmvision，独立`/home/rmvision/issue186-review-engineering03`、分支`codex/issue186-review-engineering03`，干净SHA核验。Python `/home/rmvision/venvs/hsi-baseline/bin/python`，3.12.3；Torch2.7.0a0+ecf3bae40a.nv25.02/CUDA12.8，GPU0 RTX4090。不在Windows运行Python/测试，不改全局环境。CI runner仍未注册，以下均为实际手工远程验收。

```text
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0
python scripts/check_repository.py
python research/reproduction/m4g4b/matched_target_report.py --help
MATCHED_TARGET_DEVICE=cpu python -m unittest test_g4_matched_target_review -v
MATCHED_TARGET_DEVICE=cpu python -m unittest discover -s tests -v
MATCHED_TARGET_DEVICE=cuda python -m unittest test_g4_matched_target_study test_g4_matched_target_execution test_g4_matched_target_audit test_g4_matched_target_worker_cli -v
```

最终结果：CPU专项7/7通过50.309s；GPU回归16/16通过231.098s；完整565项为564通过+1个GPU专用CLI跳过，276.393s，该CLI已在上述GPU16中运行通过。checker通过360Python/83JSON、冻结runtime不变；CLI帮助通过。

专项包含一例合成9类目标真实网络1500更新、原独立选中态核验与导出推导，导出前后原始合成产物不变；这不是54例真实研究训练。另以54组合成JSON材料检查真实51行CSV×54、18组汇总、全部54次绘图调用和输入哈希保护；全54导出测试隔离替换已单独验收的完整接受模型oracle及绘图函数，不能写成54模型真实推理或54张完整PNG验证。真实PNG由单例专项实际生成并视觉QA，六图布局、选步线、三头召回和LR清晰可读。

首版`7a6f38970ad1752a3cf2e3698da9c85359ef0c7e`专项6/6通过47.225s，但测试CSV流出现ResourceWarning；db5eea3d28f590fbe1e5185dae1ae02ba960076b关闭文件并补CLI错误SHA拒绝，专项7/7通过49.524s、全565项564通过+1个GPU专用跳过283.197s。末段窗口也明确记录实际149更新差，不误称150。之后自查发现额外输入收集可能覆盖已认证哈希，最终ad60ce0用认证哈希优先并补负例，重新完成最终SHA验收；首轮日志均保留。GitHub传输TLS失败保留为同步故障，正常重试推送，没有强推或平行提交。

## 证据、限制与下一阶段

外部工程回执：`/home/rmvision/issue186-review-engineering03-receipt.json`，SHA256 `b323baeefc7c98907d255785380423632707daae97eeef3f5b0336078f59380d`；仓库静态摘要`docs/architecture/g4-matched-target-review-engineering-results.json`为原回执同字节副本。最终日志`/home/rmvision/issue186-review-{cpu,full,gpu,help}03.log`和PNG均在checkout外。测试命令、耗时、哈希、完整tested SHA及原源码不变检查写入回执；当前状态只更新`current-status.json`后远程投影。

同步前再核生产331工作树干净，Issue184已25/54控制器接受，第26例继续，仍无完整接受文件。原launch、registration哈希不变；运行现场protected-files原文件哈希为`024c947ebd2c4accf45c0d45672b96f09a3e87dd7a1fdf46bc51eac38196798f`。此前聊天交接文本误抄为65位，今后以原文件64位值为准；仓库静态登记未写入错误值，生产文件/接受门槛没有修改。

**Issue186工程验收通过**，生产tag `module-g4-matched-target-review-v1`指实际tested SHA ad60ce0；PR187供独立评审，不合并main。本项只交付复盘导出工程。Issue184的真实54训练、全批次独立审计、真实54曲线/逐类/成本复盘和同协议query尚未完成；不读取query，不声称充分收敛、精度收益或受控提速。逐例墙钟包括资源等待、2+1498恢复及接受成本；controller elapsed不包含首轮freeze，父日志启动到退出总墙钟另报；源27054和短目标216更新单列。

下一依赖保持：当前54全部接受后停止该批次定时值守，在正常工作回合用干净331独立审计；passed后使用本模块的新版本另目录复盘，再建立全54冻结query边界并按预登记判据评价。工程完成不关闭Issue184/PR185、不移动生产tag；支持种子/K/空间/传感器、head-only/light-module及其余计划必测范围保留。main不合并。

本聊天交付位置：2026-11-06 Issue186复盘导出工程模块完成报告；真实54实验完成报告另行交付，后续心跳不重复本工程报告。
