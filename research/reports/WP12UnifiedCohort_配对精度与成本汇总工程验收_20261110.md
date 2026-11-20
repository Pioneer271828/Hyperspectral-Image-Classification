# WP12 配对精度与成本汇总工程验收

对应 Issue362、主计划 §20 和实施计划 §19.13。实测分支 `codex/wp12-unified-report`，提交 `28b0b68efc8094b4499d5d3cdf46b53e02002f3f`。使用授权 LXC 的干净目录 `checkouts/wp12-unified-report01` 和 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，CPU 单线程。

新增 `evaluation/unified_cohort.py` 与 `scripts/summarize_unified_cohort.py`，按单一注册协议接收至少三个模型种子、一个基线和一个或多个候选的完整场景网格。相同场景和种子的各臂须使用相同评估身份和逐类支持，各臂的精度与成本须绑定同一登记检查点。成本按一致的硬件身份、计时范围、重复次数与场景工作负载核对。复用已有混淆矩阵与计时统计实现，输出三头 OA、AA、Kappa、macro-F1、逐类召回/F1，以及延迟 p50/p95、吞吐、峰值字节、参数和包大小。每场景保留各臂原始读数、配对差及样本标准差。

## 输入与使用

- 登记 JSON 包含 `schema=wp12-matched-cohort-v1`、协议 SHA-256、场景字典、模型种子列表、臂列表、基线、计时协议和所有场景/种子/臂的身份。每场景声明类序、工作量及其 SHA-256，每个案例声明评估与检查点 SHA-256。
- 证据 JSON 带同一计时协议。每个案例提供三个头的整数混淆矩阵，按登记顺序以真值作行、预测作列；成本提供同一检查点、场景工作量身份、逐次毫秒读数、峰值字节、参数和包大小。输入样例见 `tests/test_unified_cohort.py`。
- 每种协议生成独立报告。端到端与常驻前向属于不同计时范围。计时协议必须事先定义环境、范围、预热和重复预算。报告依据上游已验收的输入身份证据；消费者验证输入字节与跨记录一致性，上游仍负责真实训练、冻结评分、计时执行和校准指标。

```sh
python scripts/summarize_unified_cohort.py --expected-commit <完整SHA> \
  --plan <登记JSON> --plan-sha256 <登记SHA256> \
  --evidence <证据JSON> --evidence-sha256 <证据SHA256> --out <checkout外新文件>
```

## 验收

```sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python scripts/check_repository.py --expected-commit 28b0b68efc8094b4499d5d3cdf46b53e02002f3f
python -m unittest tests.test_unified_cohort tests.test_dynamic_curve tests.test_evaluation_e0a tests.test_repository_delivery -v
```

仓库检查通过。CPU 36/36 通过，0 跳过，17.192 秒。新增 6 项覆盖精度和成本算术、乱序恢复、配对不齐、评估或检查点错配、工作负载和计时范围错配、类支持及计数、CLI 输入防篡改与输出保护。首轮回归命令误写不存在的 `tests.test_spatial_metrics`，产生 1 个导入错误；使用仓库实际的 `tests.test_evaluation_e0a` 后通过，模块代码保持同一提交。

| LXC `runs/wp12-unified-report01/` | SHA-256 |
| --- | --- |
| `checker.log` | `eff41b2a509a1bcb840ee9b9b1c588fac638b4d896459d66e54217216a042a0e` |
| `cpu.log` | `8fc1f7069b7593c28fa5020d1abdcc1a0a830c5cf92c3f9e5dc780b911fbcda4` |
| `cpu02.log` | `a6adfd85e19054d447c0340130fbc87fd1d26b463a0b42736cbd4d34aeaa508a` |

本次是 CPU 消费者验收，GPU 运行 0，真实训练更新 0，真实评分 0。下一步接入经过审核的真实精度和硬件测量产物。WP04 四组真实短测按用户要求留待后补。
