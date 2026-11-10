# WP11 独立确认准入与固定路径执行

Issue #373，实施计划 §19.12。复用开发场景、轻模型发布、冻结门控、配对计时与成本曲线。校准和确认只接受原空间计划中未使用的验证块，每块只能承担一个角色，保留正空间缓冲。训练块、选模块和保留测试块均拒绝。校准和确认阶段使用独立冻结输入。

完整模型要求带固定证据哈希的上游历史审计，列出同场景全部祖先中归一化拟合、训练和选模所用块。拟合及训练须在 devtrain，选模须在 devval。轻模型重核原训练合同和选中工件，两个模型须绑定同一开发几何。入口核对上游审计的文件身份与声明范围。实际历史完整性由上游审计负责，单凭模型文件与几何相同无法补齐。

入口先核验冻结候选、模型、块、固定路线、单核和重复预算，再读取独立授权的稀疏确认标签。其哈希、行数、授权位置与逐类支持须符合登记。固定路径每块只调用登记的轻或完整路径，覆盖不足时强制完整路径。冻结自适应路径保留门控开销。五臂测量按轮转顺序运行，报告要求 20 的倍数以同时满足既有四轮重复合同和五臂平衡。比较连接到冻结曲线、逐类保护和 p50/p95 匹配接口。

```sh
taskset -c <registered-core> python scripts/run_wp11_confirmation.py \
  --expected-commit <commit> --registration <registration.json> \
  --registration-sha256 <sha256> --out <new-external-directory>
```

分支 `codex/wp11-confirmation`。授权 LXC 实测源码 `9817916b94624c8fce889760ccb757252e571c48`。单线程 CPU 相关回归 40/40，5.000 秒，仓库检查 596 Python / 198 JSON。命令为 `python -m unittest tests.test_wp11_confirmation tests.test_route_controls tests.test_gate_calibration tests.test_dynamic_curve tests.test_dynamic_cost_pairs tests.test_repository_delivery -v`。通过真实调用次数检查、合成轻模型冻结、独立标签包、五臂实际 CPU 执行、配对报告，以及训练/选模/测试块、祖先足迹越界、整景历史和观测更换拒绝。合成轻模型有一步优化，确认运行更新为 0。

合成 CLI 回执位于 LXC `runs/wp11-confirmation-validation01/synthetic-cli/receipt.json`，SHA256 `793d076cba8818b68f74f7f041c16f9fa11e890f4bce5d3dd846f34f38525cad`。同目录配对报告 SHA256 `47d6df54a0c0b91efef796213c9ad12429899d29419ecc1f11e5cba466f1b0e1`。

真实元数据审查确认徐州原开发协议有 25 个未使用验证块，四项输入核对不变。受查轻模型发布仅有合成验收，真实阶段仍缺匹配的轻模型、完整模型同场景祖先审计与独立授权的校准和确认标签包。真实预检 `runs/wp11-confirmation-validation01/real-geometry-preflight.json`，SHA256 `5a089a5b72e9e6da5ca1accfa0f43eaa8e8fc62baed0235df872d5a72a7a3d75f`。真实确认、统计非劣和部署成本仍待验。
