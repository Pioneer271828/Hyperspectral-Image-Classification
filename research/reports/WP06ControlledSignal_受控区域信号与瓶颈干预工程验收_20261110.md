# WP06：受控区域图信号与中间瓶颈干预

Issue355 / PR356；主计划§16/§22与实施§19.7。分支 `codex/wp06-controlled-signal`，tested SHA `c6db02760ab2150c2bc6ab1826d6717c57052e2a`，基准main `2ecbcee640f95f21c6b58ef21cc74f2357ee939f`。

## 用途与实现

新增有界CPU诊断，复用现有SharedHGCN两阶段原函数：冻结区域输入、原有向图、模型状态与请求；私有eval副本计算指定源节点/通道到目标节点/通道的雅可比、正负微扰响应，并对照中间区域置零。跳数按A[接收,发送]方向计算；不把传播层数不足、激活抑制和真实过挤压混为一谈。原始参数、已有梯度、训练模式、输入、图和全局随机状态不变，优化0/query0。

依照Topping等[原论文v4 §2.1—2.2](https://arxiv.org/pdf/2111.14522v4)的节点敏感度定义独立实现；作者库固定c14e91d31ec5e9ff2dc13d90ef61b9dc2bdce829，根MIT许可和实际numba曲率/SDRF源码已读。未复制/执行上游，未实现曲率重连，也不套用曲率定理。记录 `docs/references/wp06-controlled-signal-review.json`。

## 实际验收

授权LXC lbx@10.69.37.243:1222，`/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，干净 `checkouts/wp06-controlled-signal01`，CPU单线程/CUDA禁用。

```sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests
python -m unittest test_controlled_signal test_block_input_diagnostics test_graph_route_training -v
python scripts/check_repository.py --expected-commit c6db02760ab2150c2bc6ab1826d6717c57052e2a
python /home/lbx/hsi-pioneer271828/runs/wp06-controlled-signal-audit01.py
```

CPU17/17通过，无跳过，41.838秒；checker568 Python/193 JSON通过。包括解析矩阵幂、方向反转/断连、两阶段可达性、屏蔽必经区域、空干预精确、零权重与负激活、非平凡权重有限差分、稠密/稀疏一致、计划保存恢复和错误身份/范围/预算拒绝。原有局部输入诊断与图训练恢复回归通过；未跑GPU和全仓回归。

独立审计9个初始化图支路：种子64/512/2048 × 身份/有向链/桥图；不是9个训练分类器。核对432个雅可比条目，独立NumPy矩阵微分最大绝对误差8.474014975834443e-09；有限差分最大差3.285706043243408e-06，882次图路径调用，0.725637秒。312个超出可达范围条目严格零；44个拓扑可达条目也为零，说明拓扑可达并不保证传递。该数值来自受控模型，不能解释为真实高光谱瓶颈结论。

证据 `runs/wp06-controlled-signal-evidence01/receipt.json` SHA256 `187cd55afdb623096f7392f52ea18f11ae25b3f70a4821ea1dfcb776696cafec`，同字节副本 `docs/architecture/wp06-controlled-signal-engineering-results.json`。外部审计脚本 `runs/wp06-controlled-signal-audit01.py` SHA256 `134ad979006c62de7d1a32a1abea4236252294b1241f4752c154e64a60916b3e`；完整日志同名前缀。全部运行产物checkout外。

## 结论、限制和下一项

本项CPU工程通过，无测试失败；GitHub TLS握手失败与参考网页读取失败保留，固定源码通过connector读取，Git以显式TLS1.2推送成功。没有新增真实分类精度、训练或长训练。

诊断只覆盖固定区域输入后的图支路；CNN、Q导数、跨支路交互、完整分类器与真正学习远程任务均未包括。中间置零有分布偏移；有限差分跨ReLU分段可能偏离局部导数，本接口记录差异，不放宽原分类协议门槛。输入由调用者绑定，本项尚未验证它来自哪个已接受训练case。下一模块绑定已完成GraphRouteRun的选中态、开发块和实际区域输入，并严格区分图支路局部诊断与全网络结论；GPU/真实空间重复保持未完成。
