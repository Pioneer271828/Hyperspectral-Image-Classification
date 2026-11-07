# 完整54冻结评价预登记分析工程验收

Issue204 / PR205；分支 `codex/g4-matched-target-analysis`，工作树 `F:/E2E_VCA_worktrees/g4-matched-target-analysis`。实际测试/分析器完整SHA **`73234283581c536f6e42c49bf0b3326ce3067d60`**。本项是预登记判据与结果身份接口工程，不是184真实结果分析完成，也没有读取真实query。

## 对应计划与实现

对应研究主清单P1 §7.1配对交互、P4 §10.3固定预算迁移，实施计划§13 E0、§18.5、§19。直接实现184在训练前冻结的效果判据；原分析计划SHA `0bd71779546fd5ee0cbaab57b7b3639a1c378bcf9d1379f8afb1a80231a92dee`。此前331 query入口只做描述性配对汇总，本模块提供独立且事前固定的完整收益判定。

新增 `g4_matched_target_analysis.py`、`scripts/analyze_g4_matched_target_full.py` 和 `tests/test_g4_matched_target_analysis.py`。所有331已有Python/config文件保持不变，不修改训练、冻结、query释放入口或数值门槛。不引入论文算法、外部代码、依赖或新的实验配方；复用项目已验收混淆指标函数，独立实现已登记判据。

主要比较为残差FT−容量FT。对每个模型种子先求三场景等权fused OA差，再报告三个种子的均值、样本SD及逐种子差。均值必须严格大于0且3/3种子严格为正；每景三种子平均AA下降≤1pp、HongHu平均OA下降≤1pp、每景每类平均召回下降≤2pp。所有门槛使用混淆整数计数的精确有理数运算；展示小数和四舍五入不参与判定。

残差FT−noneFT及三模式各自FT−scratch为四个独立次要主张，逐项使用同一保护条件，不能替换主要失败。scratch三架构差、容量FT−noneFT为四个描述比较，另保存残差−容量在FT与scratch间的差中差；不将描述性正差改为新成功主张。三头OA/AA/kappa/NLL、54例原始类别计数/召回、18组汇总、每景/种子配对及逐类平均差均保存在分析JSON中，分支诊断原样保留。三模型种子不当作支持集重复或显著性检验。

OA/AA/kappa/召回从混淆计数核对并计算，NLL只能认证为已固定query输出，不能从混淆矩阵独立重算。输入必须覆盖54独立角色、每例三头和所有类别；相同目标的各方法/种子/头必须有相同逐类支持数。额外保留各阶段已记录成本，区分并行case耗时之和与墙钟，不据资源等待时间宣称速度收益。

## 结果访问与文件保护

文件入口要求：原331 producer/源码指纹、原331 all54冻结和ordered query、已完成81000/324/4320/162的独立接受边界、ad60真实复盘、原184注册及分析计划完整匹配。核对每个query JSON与整批receipt一致，绑定对应冻结权重哈希，核原描述性paired_summary，要求54PNG/54CSV复盘导出全部保留。按原冻结边界在处理前后核文件；输入身份不能被新哈希覆盖，输出必须是新的同级目录且与源、输入、复盘、冻结、query目录隔离，不覆盖旧结果。

组件不调用模型、不反序列化Tensor或原GT、不释放query标签、不构造优化器。它消费冻结query的聚合指标，不能把该阶段说成从未读取query结果。真正的数据访问时机仍是全54接受→独立审计→真实拟合/逐类/成本复盘→全54冻结query之后。

## 实际环境、命令与验收

授权LXC `/home/lbx/hsi-pioneer271828`，Python3.10.20、Torch2.7.1+cu128/CUDA12.8；checkout `checkouts/issue204-analysis-engineering01`，干净完整SHA同上，分支 `codex/g4-matched-target-analysis`。所有执行在Linux，本机仅编辑/Git/SSH/SCP；没有本轮rmvision复跑或自动CI声明。

公共环境为 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`。实际执行：

```bash
CUDA_VISIBLE_DEVICES= /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -p test_g4_matched_target_analysis.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu /home/lbx/hsi-pioneer271828/envs/hsi/bin/python -m unittest discover -s tests -v
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/check_repository.py --expected-commit 73234283581c536f6e42c49bf0b3326ce3067d60
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python scripts/analyze_g4_matched_target_full.py --help
```

专项 **9/9，13.732s**；完整 **613项，612通过+1既有GPU专用CLI跳过，731.784s**；checker **384Python/93JSON**、help通过。新代码只有计数/JSON分析，新增GPU专项0。全量为满足旧CUDA钩子保留GPU0可见；启动前free24079MiB、util0%，符合12288MiB/60%门槛，不能称纯CPU全回归。

手算测试验证场景等权与像素加权产生相反结论、首次严格正差/零差、配对种子和ddof1样本SD、精确1pp/2pp边界及一计数越界、OA为正但场景AA保护失败、主要失败而四次要均成功、三头差异和描述性差中差。与原331 paired_summary在共同描述范围一致。缺例/重例/坏角色/缺头/未支持类别/支持数变化/非整数或布尔计数/指标篡改/NaN均拒绝。

完整合成54角色×三头/18组场景覆盖真实文件入口及CLI，设置torch.load与模型调用拒绝断言；旧native27/未完成query/错哈希/未完成复盘/越权更新/路径重叠/重复输出均拒绝，所有原输入字节不变。fixture的.pt与.png仅为明确标记的文本schema占位物，没有训练81000步、实际checkpoint审计或54PNG视觉QA，不冒充真实完整验收。

## 证据、失败与下一项

原始日志 `runs/issue204-{cpu,full}01.log`，checker为 `runs/issue204-checker01.json`、help同前缀。工程receipt `runs/g4-matched-analysis-issue204-engineering01/receipt.json`，SHA **`e5871f1907810419194a7cab9b122a5b39220eaa0fbff62563e63b9a1cea93f4`**；同字节仓库副本 `docs/architecture/g4-matched-target-analysis-engineering-results.json`。

本版本实现测试失败0；连接器/TLS故障保留，不把传输错误写成成功。196/202原生CUDA独立优化轨迹限制不因本模块通过而解除。实际真实query receipt读取0、模型推理0、新优化0、原GT反序列化0，没有真实效果结论。原184/198控制器和资源交接未由本模块改动。

验收结论为 **完整54冻结评价的预登记判据/身份/导出工程通过**。用户要求完成模块后持续推进，本聊天汇报进度后已开始独立Issue206既有M6完整源目标初始化/BN组件工程；真实训练的主依赖顺序仍保持184完整接受、独立审计、186复盘、原331冻结query、本分析器判定。其他支持/K、参数范围、空间/传感器、M1/E1/M6/M7/M8效果和两计划剩余机制均保留。

主入口只同步报告、证据、current-status及远程投影，不合候选源码/main；用户untracked保留。测试SHA始终为732，最终文档提交不替代该身份。
