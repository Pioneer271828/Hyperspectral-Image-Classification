# WP06：度数与距离匹配删边控制

Issue347 / PR348；对应主§16/§22与实施§19.7。分支`codex/wp06-matched-edge-controls`，tested SHA `5ba61748cbf6d499346aaca0077de5b3452e215b`。基准main `0a6ec94d1773ade858fe7deee25f9f810f271830`。

## 用途与实现

原E3-C5只匹配逐行删除数，本模块新增独立冻结控制规划：以现有重要边规则为基准，低权重与随机目标控制同时匹配删除总数、每节点行/列度数和预先声明的全图距离分箱数量。自环保留，只用现有非自环候选，不增加远程边。目标掩码必须不同于基准。采用现有SciPy1.15.3的MILP，最多2048节点/65536存储项，每个控制求解最多60秒；时间上限是逐控制，不是整个调用。未安装依赖、未复制外部实现。

求解必须返回最优状态，随后独立整数复核所有约束。无可行不同掩码/空或全删退化返回no_distinct_control；超时等状态返回unresolved，不提供可执行掩码。冻结记录绑定图、像素坐标、预算、分箱、种子和结果哈希，读取后不重新求解；verify只证明离散约束，不能独立证明最优或不可行。最优性与不可行性在本次小图穷举中另行验证。

图遵循A@X有向依赖语义，行/列度数都控制；**分箱相同不等于精确距离、边权质量、连通性或物理尺度相同**。随机线性目标不是可行掩码的均匀随机抽样，低权重是约束下的最低总权重，不是逐行独立最低边。没有改变原模型、训练/选模默认值，也未把大权重称为因果重要性。

## 实际验收

授权LXC `lbx@10.69.37.243:1222`，`/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，干净`checkouts/wp06-matched-edge01`，CPU单线程、CUDA_VISIBLE_DEVICES为空。

```sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests
python -m unittest test_matched_edge_controls test_graph_edge_intervention test_graph_support_diagnostics -v
python scripts/check_repository.py --expected-commit 5ba61748cbf6d499346aaca0077de5b3452e215b
```

19/19测试通过、无跳过，11.339秒；专项4项包括穷举最优值、度数/距离/实际COO删除、输入/RNG、保存恢复、无替代/全删/零删、超时返回可行中间解也拒绝、身份与几何负例；另15项原网络干预和图支持回归通过。checker559 Python/185 JSON通过。没有全仓或GPU回归。

## 独立留存及限制

`runs/wp06-matched-edge-evidence01`包含7个3/4/5节点手工图计划与输入、三随机种子64/512/2048。独立穷举32058个候选，核实24个最优控制、4个确无不同替代控制；输入前后哈希一致，14文件登记，3.158860秒。训练/query更新0；求解器整数优化不计作神经网络训练更新。

receipt SHA256 `83e9a860dd28d45af7e37e81f31df48ccbea8cd55ec200cb4d0b9498812935b7`，同字节副本`docs/architecture/wp06-matched-edge-engineering-results.json`。原测试日志`runs/wp06-matched-edge-tests01/combined01.log`保留。独立穷举脚本通过本聊天SSH输入执行，未另存脚本，不虚构脚本文件或哈希。

本轮无实际失败，故障注入为预期拒绝；旧E3-C5退化与低召回结论保留。本模块只交付局部图掩码规划工程，没有真实训练模型/目标标签的身份绑定、真实逐类指标、GPU成本或删边后重训结论。下一项把新冻结计划接入已接受GraphRouteRun的选中态只读复盘，补三头逐类及低权重对照，再单独登记等预算重训。WP06完整实验仍未完成；不因工程通过宣布关系有效。当前无新长训练或定时值守。
