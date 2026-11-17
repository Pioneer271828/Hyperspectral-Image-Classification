# WP12 真实精度与硬件成本证据接入

对应 Issue363、主计划 §20 和实施计划 §19.13。实测分支 `codex/wp12-evidence-ingest`，完整提交 `38d735e89a2e4f36e4467289a59a9c7275da3e1b`。在授权 LXC 的干净目录 `checkouts/wp12-evidence-ingest01` 使用解释器 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python` 验收。

## 接口与证据边界

`scripts/ingest_wp12_evidence.py` 读取带哈希的登记 JSON，将 Issue184 分析结果转换为 WP12 汇总所需的精度记录。复用原冻结边界检查及全54精度验证器，核对检查点、三头计数、逐类支持和原冻结家族。每场景、种子内用冻结家族身份核对同一评估样本。原 query release 哈希含检查点身份，另行保留。精度侧同时保留三个头的原 NLL 和预登记结论。

登记有 `accuracy`（`analysis`、`receipt`、`freeze`）、`legacy_cpu_cost`（`plan`、`receipt`、`execution` 或 null）、`hardware`（`plan`、`receipt`、`baseline` 或 null）三组。文件均通过绝对 `path` 和 `sha256` 固定。命令如下。

```sh
python scripts/ingest_wp12_evidence.py --expected-commit <完整SHA> \
  --registration <登记JSON> --registration-sha256 <登记SHA256> \
  --out <checkout外的新JSON文件>
```

正式硬件计划与回执使用 `wp12-frozen-hardware-cost-v1`。计划固定精度输出的 `protocol_sha256`，逐场景登记工作量及其 SHA-256，逐例登记场景、种子、臂、评估身份和检查点身份。回执绑定计划字节哈希，逐例交付同一检查点的毫秒样本、峰值字节、参数及包大小。计划和回执共享硬件身份、计时范围和重复预算。对齐后由原统一汇总器重算精度与成本，按场景和种子给出配对结果。实际采集端仍负责确保计时定义、环境、工作量与峰值内存的测量证据。

旧 CPU 计时接入诊断区，保留真实原始样本、范围、参数和状态张量字节。该路径明确区分状态张量字节、包大小和峰值内存。若只有精度和诊断计时，报告状态为 `accuracy_ready_hardware_pending`。

## 验收结果

- 真产物只读接入 54 模型、162 分类头和 PaviaU CPU 稠密/紧凑各 12 条延迟。稠密 p50/p95 为 5145.682/5166.496 毫秒，紧凑为 5100.678/5157.721 毫秒，均为旧常驻输入前向。与全54检查点重合数为 0，旧计时缺峰值字节与包大小。输入的 6 份 JSON 在运行后再次核对 SHA-256，保持一致。合并字段为 null。
- 合成完整输入走同一 CLI，输出 54 行、18 组场景/臂摘要，输入哈希保持一致。负例覆盖精度产物错配、检查点或家族错配、硬件计划错配、缺行、计时范围混用、峰值缺失及旧检查点篡改。
- 仓库检查通过，579 Python、198 JSON。CPU 发现 30 项，28 项通过、2 项历史固定版本用例按原规则跳过，34.126 秒。首轮遗漏 `PYTHONPATH` 中的 `tests` 导致既有成本用例导入错误；补齐后在同一提交通过。首次合成 CLI 驱动的登记文件名与夹具冲突，改用独立登记文件成功，保留失败日志。

```sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests PYTHONDONTWRITEBYTECODE=1
python scripts/check_repository.py --expected-commit 38d735e89a2e4f36e4467289a59a9c7275da3e1b
python -m unittest tests.test_wp12_evidence tests.test_unified_cohort tests.test_g4_matched_target_analysis tests.test_g4_cpu_cost tests.test_repository_delivery -v
```

| 证据，LXC `runs/wp12-evidence-ingest01/` | SHA-256 |
| --- | --- |
| `real-registration.json` | `9626b86e4f5706468c778008d3a988df44b2c28c1a055e19da38a99fe98c0475` |
| `real-evidence.json` | `e3e908a2b4fb6c7c3a158844b8ca1f704f01beb7321938dcbf63d3822c470d2c` |
| `cli-fixture/report.json` | `88a863da720493ee2d29aac34496570bc8fff46180036a66e3144376dd6c6c4c` |
| `cpu02.log` | `547c7b60675f446d70a055a46907ee04ab8b0901914afff1c0d57d3e8def804a` |
| `cli-fixture02.log` | `84d48d66047d018a5ba93b5c8e4e8c6c990daaeef1a024c90787d870bfb7b696` |

接入验收通过。本轮 GPU 执行、真实训练更新和新 query 评分均为 0。下一依赖是同一组冻结检查点的同硬件成本采集。WP04 真实短测仍按用户安排后补。
