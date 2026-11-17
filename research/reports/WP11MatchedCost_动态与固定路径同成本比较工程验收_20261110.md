# WP11 动态与固定路径同成本比较

Issue #371，实施计划 §19.12，承接 DY-2B 的冻结工作点与逐类质量报告。复用原成本曲线与门控协议，不引入新的研究机制。

配对登记固定曲线协议哈希、每个自适应工作点对应的固定路径 ID、p50 和 p95 各自的相对容差。比较使用同一份确认曲线报告的原始计时样本，复算 p50 和 p95，以固定路径的对应值作分母。两个容差须同时满足。每个自适应工作点恰有一个登记配对，所有配对均输出。输出包含整体与各场景、条件的 OA、AA、逐类召回、NLL、ECE 差值及支持数，并保留两个工作点各自的经验保护判定。未定义差值保留为 null。

CLI 接收带 SHA256 的登记与曲线文件，在 checkout 外生成新报告。文件哈希核对登记内容与来源；真实登记时间、测量来源、空间独立性仍由实验准入证据确认。

```sh
python scripts/report_wp11_matched_cost.py --expected-commit <clean-commit> \
  --curve <curve.json> --curve-sha256 <sha256> \
  --registration <registration.json> --registration-sha256 <sha256> --out <new-report.json>
```

分支 `codex/wp11-matched-cost`，实际测试 `28d1272294994b03aa9b36220c2dc3923471365f`。授权 LXC 使用 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`、单线程、CUDA 不可见。回归命令 `python -m unittest tests.test_dynamic_cost_pairs tests.test_dynamic_curve tests.test_gate_calibration tests.test_route_controls tests.test_repository_delivery -v` 通过 38/38，2.861 秒。仓库检查通过 593 Python / 198 JSON。

专项核对配对完整性、原始计时复算、固定路径身份和输入替换拒绝。合成验证中 p50 相同但 p95 超限，正确标为成本未匹配，仍保留差值。实际 CLI 工程报告在 LXC `runs/wp11-matched-cost-validation01/synthetic-report.json`，SHA256 `14813cd999923cddfce52560ebf5ee809de94dad75095a13ff1a42adf9835225`。真实独立确认、真实同成本固定路径比较和部署成本仍待验。
