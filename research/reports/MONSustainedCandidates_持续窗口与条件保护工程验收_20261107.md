# MON 持续窗口通道候选与条件保护工程验收

对应全量必测清单 MON/MASK、实施计划 §19 和统一训练监测计划的联合判定。新增 `monitoring/windows.py`，把既有 MON-1 激活记录、MON-2 单次实际梯度及更新记录按同一训练样本和更新步配对，跨固定窗口汇总。响应率按有效元素计数加权，权重 RMS、梯度 RMS 和相对更新取每窗最大值。只有每窗通过预声明阈值、并通过最终固定探针的全部条件保护，才输出层内候选索引。冻结参数、缺失梯度、测试分区、缺失保护分桶及空保护支持均拒绝。输出供后续可逆消融，既有模型和训练入口保持原实现。

实测分支 `codex/mon-sustained-candidates`，SHA `42a8a8c7d8becb798b5929a044387b998c566b00`。授权 LXC 检出 `/home/lbx/hsi-pioneer271828/checkouts/mon-sustained-candidates01`，Python 3.10.20、PyTorch 2.7.1+cu128。CPU 监测测试37/37（新增4项）共8.349秒，CLI 帮助和仓库检查322 Python、128 JSON通过。

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_repository.py --expected-commit 42a8a8c7d8becb798b5929a044387b998c566b00
python -m unittest discover -s tests -p 'test_monitoring*.py' -v
python -m monitoring.windows --input <冻结证据JSON> --sha256 <SHA256> --out <新输出目录>
```

示例用小型合成网络的4次实际 SGD 更新和固定探针，采集开关两条轨迹的最终模型哈希一致。通道2训练中低响应，但稀有分桶响应率为1，被保留。通道1进入候选。另验证了不同有效元素数的加权、后期更新排除、错误证据与输出覆盖拒绝。

证据根为 `/home/lbx/hsi-pioneer271828/runs`。日志 `mon-sustained-candidates01.log` SHA256 `e5dd43b20352f99d1b26cc59bdfb2625a1a0199313469f758f20ebb895cd0fbd`。示例 `mon-sustained-example01/result/candidates.json` SHA256 `61c4bebdcef0591785971a35f9e385e50c37f18a111bd251274019919830947e`。

工程接入通过，实验证据仍待真实模型的持续采集、监测开销及逐例可逆消融。GPU 专项和全仓回归本轮未跑。下一步用固定依赖组和单通道/组合干预复核候选，再接 MON-4 映射及受保护的结构压缩对照。本聊天2026-11-07交付本模块工程进度。
