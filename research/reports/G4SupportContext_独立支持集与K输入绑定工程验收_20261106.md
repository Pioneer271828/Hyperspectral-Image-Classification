# G4独立支持集与K输入绑定工程验收

Issue188 / PR189；对应研究主清单P4 §10.3第3、5、6项和模块计划§19“补机制与独立重复”。用途是为支持种子512/2048、K1/5等后续真实重复提供经过绑定与隔离的输入，不将当前Issue184固定support64/K10批次误当支持集重复。用户2026-11-06要求在正常工作回合继续模块搭建，本项与原冻结长训练独立；没有启动下一数值实验。

## 实现与复用

工作树`F:/E2E_VCA_worktrees/issue-188-support-context`，开发分支`codex/issue-188-support-context`，基于已交付186的`eb803121fda5f85f60365366be4a00d59f331e85`。唯一写入人为本聊天助手，无多智能体。

新增`g4_support_context.py`、`scripts/prepare_g4_support_contexts.py`、`tests/test_g4_support_context.py`。复用已验收149的`target_label_protocol`（生产`2f9e661bfcb777ed6b8dd2e8b6211a998eac4fdc`）、151的无标签观察/图和`target_data_context`语义，没有重新生成支持集、划分或图，也未引入新的外部算法、源码或依赖。原331及当前184的全部Python/config文件不变，模型、损失、标签访问、训练预算和选模规则未改。

- 冻结标签构造/独立核验两回执、9个目标/支持种子family、3组观察/record及独立输入规格，共18个输入文件。
- 产生3数据集×支持种子64/512/2048×K1/5/10的27个小型覆盖层，每个仅含`train_indices/train_labels/val_indices/val_labels`四个CPU张量；原无标签观察及Q/A只读复用，不重复复制影像。
- 强制显式支持种子、K和模型种子；模型种子独立记录，head seed=model seed+1000。各K保持嵌套支持、同一V5与同一family query身份；跨支持种子的query集合并非自动相同。
- 每个条件单列K+5标签成本：PaviaU/Xuzhou为54/90/135，HongHu为132/220/330。复用标签不重复计为新增标注；不同支持种子允许交叠，不冒充互斥空间划分。
- 读取方不反序列化curator family，只获得选中K和V5；拒绝额外query/test张量、非平衡/交叠标签、错种子/K、缺例/重复27角色、改变图或观察、输入哈希变化、路径越界及执行开关。manifest始终`engineering_only/execution_enabled=false/optimizer_updates=0`。

这是逻辑输入接口约束，不是操作系统安全沙箱；既有149构造者读取完整标注进行oracle分层的历史语义保留。本项只读已经验收的family和无标签观察，不重新读取原始GT或query标签。

## 实际版本、环境和测试

实际源码/测试完整SHA：`937861fe87ce1fe619268063dfed1193d416853e`。

| 环境 | 独立checkout/分支 | 实际结果 |
|---|---|---|
| rmvision Linux，Python3.12.3/Torch2.7.0a0+ecf3bae40a.nv25.02/CUDA12.8 | `/home/rmvision/issue188-support-engineering01` / `codex/issue188-support-engineering01` | CPU7/7，6.562s；全572项571通过+1既有GPU专用CLI跳过，285.960s；checker363Python84JSON/help通过 |
| 授权LXC，Python3.10.20/Torch2.7.1+cu128 | `checkouts/g4-support-context-issue188-engineering01` / 同名codex分支 | 真实27输入准备13.406379s，输入18文件不变，零更新/零query |
| LXC独立消费者，同完整SHA | `checkouts/g4-support-context-issue188-independent01` / 同名codex分支 | CPU7/7，9.649s；真实27独立复核160.581760s，54阶段/108标签张量/27观察图、47文件不变 |

Windows仅编辑/读文件/Git/SSH/SCP，所有Python和检查在上述Linux容器执行。新模块是CPU数据协议，无新增GPU算法，本轮不虚列GPU专项通过；完整回归中的既有GPU专用CLI在本版本未执行，原冻结实现已有182/186的GPU验收记录，未冒充本次重跑。CI runner仍未注册。

```text
# rmvision: /home/rmvision/venvs/hsi-baseline/bin/python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest test_g4_support_context -v
MATCHED_TARGET_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit 937861fe87ce1fe619268063dfed1193d416853e
python scripts/prepare_g4_support_contexts.py --help
# LXC: /home/lbx/hsi-pioneer271828/envs/hsi/bin/python; CUDA_VISIBLE_DEVICES为空
python scripts/prepare_g4_support_contexts.py \
 --expected-commit 937861fe87ce1fe619268063dfed1193d416853e \
 --inputs /home/lbx/hsi-pioneer271828/runs/g4-support-context-issue188-inputs01/inputs.json \
 --inputs-sha256 e24757547958725b2c47ff9297c08e044db716013e3a239226b79cf2097f1a35 \
 --out /home/lbx/hsi-pioneer271828/runs/g4-support-context-issue188-preflight01
python /home/lbx/hsi-pioneer271828/transfer-records/issue188-context-independent01.py
PYTHONPATH=.:tests python -m unittest test_g4_support_context -v
```

7项专项覆盖全部27组合与原消费者逐字段精确一致、模型/支持种子分离、嵌套K/固定V5、读取方不解析family、有限输出、未接受/已变输入、额外标签张量、输出隔离/端到端保护及错误SHA提前拒绝。真实独立核验直接从原support_order重建预期标签，核原验证标签、观察和Q/A；不是重新调用生产导出路径冒充独立oracle。没有模型前向、优化或概率计分，未初始化GPU上下文。

## 证据与保留记录

真实27个覆盖层总计126599字节（约123.6KiB），与manifest、producer receipt保存在`runs/g4-support-context-issue188-preflight01`。manifest SHA256 `57d13699414b1c041c5332d834a392e7d3af499684a56886d75e76c3c87d66c6`；独立回执`runs/g4-support-context-issue188-independent01/receipt.json` SHA256 `c3fe49f855a2a6ca059b2e7cc2b6fd310308b7c495f9f83c6646bb18e217b611`。

汇总工程回执`/home/rmvision/issue188-support-engineering01-receipt.json` SHA256 `566280b7f110a522ea950b04f79d6e89d9914dbeac0ca40fc2f4b24e50b0eaf7`，同字节副本为`docs/architecture/g4-support-context-engineering-results.json`。LXC元数据包SHA256 `0e40d952661c95931bb4b41648d4d4b54181ab3e05efbbd977340d5aae84083a`；全部原始产物、日志和独立驱动在checkout外，未提交标签张量、数据或运行日志。

本版本未发生模块代码/数值测试失败。一次远程元数据提取包装因heredoc尾标记引起NameError，发生在已成功读取CPU/full结果之后；已改用外部文件驱动归档，原测试日志未改，不重跑择优。GitHub首次create transport失败后先GET核无Issue，再成功创建188，未产生重复Issue或平行提交。当前状态只改`current-status.json`并远程投影。

最终文档交付同步时，共享clone的alternates嵌套达到Git上限，新`issue188-support-delivery-source01`检出失败；保留该失败目录，不reset/clean。随后`git clone --no-local`建立独立对象库，source02投影通过，后续最终source/primary验证均使用无alternates的独立clone。该问题属于交付对象库隔离，不修改冻结训练、实际tested SHA或已有验收结果。

## 结论和下一依赖

**本输入绑定工程验收通过。**生产tag `module-g4-support-context-v1`指实际937861f；PR189独立评审、main不合并。支持/K真实效果、空间隔离、传感器家族泛化、head-only/light-module和其他计划必测项仍未完成，不能由27输入绑定替代。

当前Issue184真实54批次继续，最近检查41/54、61500/81000接受，第42例运行，完整接受/独立审计/复盘/query尚无；新增输入模块没有改动它。HongHu残差scratch的验证CNN第11类零召回（5支持）已保留，不调整配方或预算。后续顺序保持全54接受→独立331审计→已交付186导出真实复盘→统一冻结query/预登记结论；然后才登记后续支持/K真实重复及对应显式训练接入，不能将本engineering-only manifest交给旧64/K10训练器执行。

本聊天交付位置：2026-11-06 Issue188独立支持集与K输入绑定工程模块完成报告。该报告是工程交付的一部分，后续心跳不重复；真实支持/K实验另行完整报告。
