# WP06：匹配删边图的等预算重训与恢复

Issue351 / PR352；主§16/§22、实施§19.7。分支`codex/wp06-matched-edge-retraining`，tested SHA `c9ddaa68db6ecafedf96a6e90310e375c13f9959`；基准main `98cd441bd7fc6b6d7cdeedaad52df197e1d13ed5`。

## 实现和合同

已完成动态路由模型只提供固定图拓扑，覆盖全部开发训练/验证块。none、important、low_weight及random64/512/2048各臂从同一**原始源权重**重启，使用相同曝光计划、Adam、学习率、随机种子、损失和first-min完整验证NLL选模规则。选择拓扑的前序训练成本单独计入，不能隐藏。拓扑固定不等于Q和边权固定：沿用已验RouteContextBank，Q/边权仍随模型训练。

每块控制必须满足原边数、逐行/逐列度数和距离分箱合同；任何块没有合格替代，整臂拒绝。保存的银行绑定选中源状态、上下文/原始输入、候选和选中边；重训合同另绑定原始初始化与预算。恢复拒绝错臂、错输入、错合同。新增独立接口，未改原训练器、backbone或默认配置。当前只接收完成的动态donor，固定donor扩展未实现。

## 实际验收

LXC `lbx@10.69.37.243:1222`，`/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`；干净`checkouts/wp06-matched-retraining01`，CPU单线程、CUDA禁用。

```sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests
python -m unittest test_matched_edge_retraining test_matched_edge_review test_graph_route_training -v
python scripts/check_repository.py --expected-commit c9ddaa68db6ecafedf96a6e90310e375c13f9959
python /home/lbx/hsi-pioneer271828/runs/wp06-matched-retraining-audit01.py
```

CPU13/13通过、无跳过，73.264秒；checker563 Python/189 JSON通过。六臂各24步连续与7+17恢复逐事件、参数、优化器、选中态精确一致；完成态恢复0步；原始初始化与除银行字段外全部训练合同相同；原donor/源不变；真实梯度和图参数更新可见。零删除精确、错误身份/缺块/超时全臂拒绝和跨臂恢复拒绝通过。未跑GPU或全仓回归。

独立留存一个合成九类donor24步与六个重训模型6×24=144步，完成恢复0。核查相同曝光台账和固定逐类验证5例、1458混淆矩阵格、162召回，19.975431秒。六臂都选20步；首末loss约6.678→6.541。三头最低召回均0，零召回类数分别7/8/7；短测不支持充分拟合或效果收益。random64与2048在本小图得到相同掩码/权重文件，不能当作独立模型种子重复。

证据`runs/wp06-matched-retraining-evidence01`的7个模型、plan.json、summary.json及receipt.json；receipt SHA256 `14c26a5784938a4a9741b66675d31380109bb53d72d358969595aee9a09fa5fe`，同字节副本`docs/architecture/wp06-matched-edge-retraining-engineering-results.json`。独立audit脚本保留在上一目录，SHA256 `6f0f408908b51c57d059b07d352ba9c60f4b5148e7b94c221b6d35fa0ce46f08`，不在receipt输出文件列表中。训练/测试产物全部checkout外。

## 保留的失败与限制

首轮tested3506827314c83edfaab05604685f88e9fa6aebe2：13项中12通过、1错误，46.624秒；测试试图覆盖已存在中断检查点，被原不可覆盖写入正确拒绝。修正测试为逐臂独立中断/完成路径，未放宽生产合同；原`runs/wp06-matched-retraining01-tests.log`保留，最终日志`runs/wp06-matched-retraining02-tests.log`。更早40ea测试草稿把advance当增量的错误已在执行前修正，未产生错误训练预算。Git传输TLS失败经重试成功；SSH闲置断开后重新核容器及SHA。

结论：等预算重训/恢复CPU工程通过，真实多场景/多模型种子、GPU成本、逐类保护与关系选择效果未完成。复用选模验证只作开发诊断，不是独立确认。下一项补已训练静态模型的同容量探针输入身份和训练/验证标签权限绑定，复用原采集/探针实现，再推进其余机制证据；不重复已完成实验，不以本工程通过标记WP06全部完成。未启动真实长训练。
