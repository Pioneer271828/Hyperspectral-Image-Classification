# G4既有54批次三卡续跑工程与现场交接

Issue198 / PR200。对应研究主清单P4 §10.3、§17.2固定预算与成本、§19运行身份，以及模块计划§19完整M4目标适配和资源排程记录。用户在侧聊明确要求完成当前Issue184剩余任务的2—3卡续跑并通知主任务。本项仅交付资源调度工程和真实现场交接，不交付184全批次实验结论。

## 实现、版本与边界

唯一写入工作树 `F:/E2E_VCA_worktrees/issue-198-resource-handoff`，分支 `codex/issue-198-resource-handoff`，基于已接受190交付 `934e3d24cc4366fd48e6c5539449b89f09bfb458`。实际tested/资源生产源码为 **`5db71e82ad3fc36f3cef7046a22c1162be756097`**。只新增 `g4_resource_handoff.py` 和专项测试，原331全部Python/config字节不改；无外部机制、上游源码或新依赖。未修改primary/current-status、父任务工作树或main。

资源控制器从独立冻结生产checkout导入原331模型、数据、训练、ordered验证与严格逐例审计函数。原launch、registration、每模型provenance及数值合同原样保留，资源附录另外固定控制器版本、原进程身份、接受前缀、剩余case/阶段、卡池、预算和文件哈希。附录明确仅覆盖原registration的物理GPU排程字段；每个模型仍单卡训练，并行的是独立case。

控制器统一写全局进度；每卡一个任务、每个case阶段唯一归属，按first2→continue1498→readonly0→原逐例audit推进。每次认领及数据准备后的实际更新前均检查free≥12288MiB/util≤60%。资源不足只允许文件不变的0更新等待并重新排程；真正执行失败保留产物、停止后续认领，不按分数或失败重训。支持未开始、精确2步断点、完成1500但待验收三个交接状态；活动训练锁或训练子进程存在时禁止交接。

## 实际工程验收

Linux LXC，Python3.10.20、Torch2.7.1+cu128、CUDA12.8。工程checkout `/home/lbx/hsi-pioneer271828/checkouts/g4-resource-issue198-engineering01`、资源生产checkout `checkouts/g4-resource-issue198-production01`，均为干净5db、同分支。所有执行在Linux；Windows仅编辑、读文件、Git和传输。rmvision SSH/SCP连接被远端关闭，未计为验证成功；改在已有授权LXC验收。

| 检查 | 实际结果 |
|---|---|
| CPU专项 | 8项：7通过、1 GPU专用跳过，4.801s |
| GPU3专项 | 8/8通过，102.296s |
| 完整回归（GPU7可见，配对目标显式CPU） | 585项：583通过、2 GPU专用跳过，457.483s；新资源GPU专项另已通过 |
| 额外跨物理卡合成实际训练 | 原CLI GPU3先2步，资源worker GPU7继续1498及完成0更新，GPU3原严格审计：1/1通过，100.260s |
| 资源生产checkout checker/help | 368 Python、86 JSON通过，完整SHA和工作树干净，help通过 |

```text
# /home/lbx/hsi-pioneer271828/envs/hsi/bin/python；工程checkout，5db完整SHA
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -p test_g4_resource_handoff.py -v
CUDA_VISIBLE_DEVICES=3 RESOURCE_TEST_GPU=3 python -m unittest discover -s tests -p test_g4_resource_handoff.py -v
CUDA_VISIBLE_DEVICES=7 MATCHED_TARGET_DEVICE=cpu python -m unittest discover -s tests -v
python /home/lbx/hsi-pioneer271828/transfer-records/issue198-crossgpu01.py
python scripts/check_repository.py
python g4_resource_handoff.py --help
```

专项实际运行原冻结CLI2步、新worker1498步、完成0恢复和原1500/6指标/54逐类召回/3精确重载审计。跨卡附加驱动只调整合成fixture的CLI卡号/可见卡，不修改生产数值函数。三卡队列并发、重复/缺失身份、预算和失败停队列主要为调度fixture检查，不冒充54真实模型已验收。CI未注册，以上均为显式远程执行。

工程回执 `runs/g4-resource-issue198-engineering01/receipt.json` SHA256 `583d117835664b8973af0a6d86c5128958a38a152f37c3fc30a6883bdcce1c4d`。本分支 `docs/architecture/g4-resource-handoff-engineering-results.json` 是精简摘要，保留原回执路径/哈希及测试日志哈希，不声称与大回执同字节。

## 真实交接和独立只读见证

原批次在第51例完成接受后捕获安全边界，接受分布PaviaU18、Xuzhou18、HongHu15，共76500更新。剩余仅HongHu/model2048的none、capacity_control、x_residual三例全微调，均未开始；**剩余4500、总预算81000**，没有丢弃或重复前51例。原51例每例1500/51事件/三精确重载的证据继续由原进度行保存，逐类弱项不因换卡删除。

原协调器273458已核实际身份，在无训练子进程、无任何case锁时暂停；写好资源附录并完成工程/跨卡验收后，先建立等待接管的新PID，再仅结束旧协调器。旧wrapper正常留下 `EXIT_CODE=137 TOTAL_WALL_SECONDS=77555`，结束UTC `2026-11-07T00:47:44Z`（北京时间08:47:44）；这是已登记协调器交接，不是数值失败、没有终止训练子进程。77555s为原wrapper段（21h32m35s），应与新资源段分别报告，不将新控制器elapsed当全批次成本。

新控制器 **608030**，卡池 **GPU1/3/7**，资源目录 `/home/lbx/hsi-pioneer271828/runs/g4-matched-target-issue184-resources01`。`amendment.json` SHA256 **`8e8004aa6b56da7d4b59b0526cbe5248cc588de76138e624313e0c1e5f9be414`**；目录内保留 `execution-handoff.json`、controller日志、每阶段日志、events和卡资源判定。三例first2均实际完成；后续continue允许在登记卡池内换卡，实际归属按阶段events记录。

独立外部见证驱动不导入资源生产模块，不执行优化或query。2026-11-07T00:50:16Z回执通过，29.373429s：重新计算原7930输入与前51例5763文件的哈希、核前51进度行完全相同、旧PID不存在、原331与新5db均干净；核3个first2事件共6更新、唯一阶段、日志哈希以及真实worker GPU可见性。见证时none/容量/残差游标分别211/211/121，三个continue worker分别在GPU3/7/1；全部仍在训练，不是54接受结论。

见证回执 `runs/g4-resource-issue198-independent01/receipt.json` SHA256 **`c444385c6caa03ec977342af749815ec085b772b795709158ee302b878f01e3d`**，本分支 `docs/architecture/g4-resource-handoff-live-witness-results.json` 为核哈希一致的同字节副本。该见证仅覆盖交接当时的保存、身份、排程和真实更新；没有在训练中运行全54数值独立审计。

## 保留失败、限制和下一依赖

初版52c的CPU专项7项出现5通过/1错误/1跳过（5.062s）：全局/proc扫描遇到受限proc1/cwd，尚未操作真实协调器。后续d811改为核身份协调器的直接子进程检查，3fec补原CLI→新worker真实续训，5db补0更新资源等待转卡和精确剩余预算验证；原日志保留。rmvision两次连接失败、未成功SCP均保留，不当作测试运行。首次取回大工程回执因终端输出截断被拒收，后从原回执导出小摘要并逐字节核哈希，原大回执未改。

**资源调度工程和51例安全边界后的三卡真实续跑交接通过。**这是Issue198范围，不等同Issue184完成、充分收敛或精度收益。支持断点换卡并不声称不同物理卡训练轨迹bitwise一致；已有ordered数值/分类阈值及同路径三重载精确要求未放宽。

主任务已收到新PID、资源附录/哈希、旧137计划原因、预算及监控字段；普通30分钟值守由主任务维护。不要只盯旧PID或将并行progress新增行按旧长度切片，需按case身份判断新增接受。原184/185保持open/Draft。全54训练接受后仍必须独立331全批次审计、补充资源执行终验、186真实曲线/拟合/逐类/选步/参数更新/成本复盘，再原331全54冻结同协议query和预登记判据报告。原producer最终验收仍须324指标/4320召回/162重载/81000及全部输入不变；非选中事件的概率不扩称独立复算。本分支不提前关闭184、不运行query、不启动下一真实实验。
