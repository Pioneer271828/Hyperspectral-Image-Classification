# WP12 冻结54成本批次登记与汇总

Issue #370，实施计划 §19.13。使用 Issue365 的三个无标签观测、Issue368 保留原始文件的推理入口和 Issue363 冻结精度来源。

完整登记覆盖 3 场景 × 3 模型种子 × 6 实验臂。以从头训练且未加残差的 `target_scratch_full/none` 为汇总基线。每例在独立进程中执行 3 次预热、12 次计时，单例超时 900 秒。各场景和种子组成一个比较区块，六臂顺序随区块轮转。预算合计 162 次预热、648 次计时前向，另有 108 次校验前向。计时范围为常驻输入的整景三头前向，保留原始样本、p50、p95、吞吐、CUDA 峰值分配、参数数和权重加配置文件字节数。模型加载、传输和前向前后校验处于计时范围之外。

登记核对全部检查点、配置侧车、观测和精度证据。运行阶段将批次绑定到同一物理 GPU UUID、计时环境和历史 worker `ad107501595f9b9cb625086e8457a852d7a60789`。每次启动前复核至少 12288 MiB 空闲显存且利用率不超过 60%。接受记录以作业与结果哈希绑定。失败尝试单独保留，续跑复验并跳过已接受案例。全部通过后生成 `cost-plan.json`、`measurements.json`、`hardware-receipt.json` 和 `combined-report.json`，保留原精度结论和三头配对结果。

真实登记在 LXC `runs/wp12-cost-frozen54-campaign01/registration.json`，SHA256 `e373b6648314e3e0514506ecf78fa8dcb582791a24f98fa23d2060f89a84c978`。输入描述与验证日志在 `runs/wp12-cost-campaign-validation01/`。

```sh
python scripts/run_wp12_cost_campaign.py --expected-commit <clean-coordinator-commit> \
  --out <external-campaign-directory> run --registration-sha256 <registration-sha256> \
  --worker-root <preserved-historical-checkout> --gpu-uuid <physical-gpu-uuid>
```

分支 `codex/wp12-cost-campaign`，LXC 源码测试提交 `8327b0f94a7e1fa86f9c42c459e28e578f6d202a`。仓库检查通过 590 Python / 198 JSON。CPU 回归 28 通过、1 项 CUDA 跳过，28.686 秒。专项用 54 个合成回执覆盖分段续跑、错误作业拒绝、接受记录篡改拒绝、完整汇总和完成后零重跑。真实 54 例已登记，GPU 空闲显存核查最高 4236 MiB，实际计时与真实联合报告待执行。WP04 真实短测仍待补。
