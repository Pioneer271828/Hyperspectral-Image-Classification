# WP06：选中模型全网络信号诊断工程验收

Issue357 / PR358，对应主计划§16/§22、实施计划§19.7。分支 `codex/wp06-selected-signals`，实际tested SHA `2cba7cfc198efd231d676c12d24ba1bd7882b9b5`。这是完整网络诊断接入的CPU工程验收，不是整个WP06或真实效果验收。

## 用途与实现

将已完成GraphRouteRun的选中态、源/数据/开发块身份、目标像素、类别和输入源像素/物理波段绑定为固定计划。在私有模型副本上计算三头局部输入梯度及正负微扰响应，保留原网络各阶段Q、图权重和路由，不用同一邻接矩阵替换各阶段图。记录每次Q、图值与连接是否改变；输入为训练统计归一化后的光谱，微扰不裁剪，坐标不变。未完成训练、query块、错误模型/输入/计划身份均拒绝；不读取目标标签。

复用已验收block_input_diagnostics接口；原模型、优化器、已有梯度、训练历史、随机状态和输入不变。参考原论文与作者固定源码的核查继承Issue355，登记 `docs/references/wp06-selected-signal-reuse.json`；独立实现，未复制或执行上游。

## 实际测试和独立核对

授权LXC lbx@10.69.37.243:1222，解释器 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，干净checkout `checkouts/wp06-selected-signals01`。CPU单线程、禁用CUDA。

```sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests
python scripts/check_repository.py --expected-commit 2cba7cfc198efd231d676c12d24ba1bd7882b9b5
python -m unittest test_selected_signal_diagnostics test_block_input_diagnostics test_controlled_signal -v
python /home/lbx/hsi-pioneer271828/runs/wp06-selected-signals-audit01.py
```

CPU14/14通过，无跳过，8.512秒；checker570 Python/195 JSON通过。覆盖保存恢复、错误源/当前模型身份、未完成训练/query拒绝、三头原生微扰输出精确一致、图变化标记、异常后的拦截恢复及原状态不变。之前生产实现c0dfc3bc48d8ef7c82f4b01746464c5e274229fd同14项8.467秒通过；最终提交只增强未完成训练拒绝测试。未跑GPU或全仓回归。

独立核对复用Issue351的7个合成训练模型（1个24步donor、6个各24步重训对照），不重新优化。每模型1个devtrain块及1个devval块，共14块，84个雅可比条目、168项独立概率分数精确核对；生产完整模型前向84次、独立原模型参考前向56次，7.873714秒。10个历史输入文件哈希不变，新增优化0、query0。构造旧fixture时仅抑制其重复训练，实际checkpoint装载、完成状态和来源认证正常执行，没有替换接受判据。

56个方向图值变化，连接拓扑变化0；有限差分与局部梯度最大差 `5.014570311332565e-05` 原样保留。这不是独立梯度oracle，也未设置或宣称通过严格差值门槛；非光滑激活/路由及FP32均是方法限制，不能未经诊断指定本次差值的成因。

证据目录 `/home/lbx/hsi-pioneer271828/runs/wp06-selected-signals-evidence01`。receipt SHA256 `5d51cd4d253d5cb6ec59c09095a6aeb9b70eb8a7fab2662bcbd756cb7863cd71`，同字节摘要 `docs/architecture/wp06-selected-signals-engineering-results.json`。审计脚本SHA256 `964051cd56613bfdc6fc28d2774107eac56d397e0036f772a184b185cfc41ee5`；测试日志 `runs/wp06-selected-signals02-tests.log`，审计日志 `runs/wp06-selected-signals-audit01.log`。产物均在checkout外。

## 结论与限制

本项CPU工程验收通过，无数值测试失败。传输连接中断保留并重传成功；未改训练代码、预算、损失或选模。复用选模验证块只作诊断，不称独立确认。归一化微扰不等同物理缺失波段；输出局部敏感度不证明远程学习恢复、过挤压缓解或分类收益。没有新增真实OA/AA/Kappa或小类收益，GPU/真实HSI与空间重复保持未完成。

下一项回到实施计划其余缺口：WP07/08仍受真实尺度证据、候选内效果前提限制；优先核查WP09的训练/预处理未知类别排除身份，补现有开放集流程的可验证边界，避免继续堆叠同类诊断。
