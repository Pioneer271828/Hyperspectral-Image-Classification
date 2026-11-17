# WP12 原始冻结推理运行时隔离采集验收

对应 Issue368，主计划 §20/§22、实施计划 §19.13。承接 Issue365 发现的8个原有文件差异，采用独立历史运行副本，未修改原生产checkout，也未将当前骨干直接作为旧精度的成本运行时。

## 版本与实现

- 现代适配/汇总分支 `codex/wp12-historical-cost-adapter`，实际tested SHA `a4b4408d993bfd9a93eb7ead4ce933974c58f543`。
- 历史隔离worker分支 `codex/wp12-historical-cost-worker`，实际tested/worker SHA `ad107501595f9b9cb625086e8457a852d7a60789`。基于原生产 `331d718567b1a9726934f445665999df972392b1`，只新增三个文件，原392个Python/config文件逐个一致。
- 两边的 `evaluation/wp12_cost_primitives.py`、`evaluation/wp12_resident_cost.py`、`scripts/measure_wp12_historical_cost.py` 字节相同。现代完整消费者不拷进旧运行时；worker只用原始加载、原始前向和原始ordered算术。

新的纯标准库计时/文件校验辅助避免导入后来添加的推理机制；其统计结果与已有 `evaluation.dynamic_curve` 的中位数、inclusive p95、均值、样本SD和吞吐完全相同，已测试正常样本和无效样本。现代 `assemble` 增加显式 `--worker-commit`，允许接收独立且事先固定版本的历史worker，不再要求worker与汇总器是同一提交。

历史入口必须接收带SHA的原全54 freeze，并验证原producer为331、原文件清单完整、原文件逐项未改和当前工作树干净。`environment` 导出实测环境；`measure` 再核固定job和环境，仅接受ordered算术。该source准入不替代全54案例/观测/精度绑定，后者由Issue363/365及完整汇总门槛保证。

## 内存口径修订

复核Issue364发现CPU高水位读取点在收尾张量校验前，不能充分支撑原报告“包含全部校验”的表述。本版将CPU读取点移至收尾张量/文件校验后，口径明确为 `fresh_worker_peak_rss_through_validation_bytes`，仍包括导入、加载和校验，不包括之后CLI检查/结果序列化，也不是增量内存。GPU仍为计时前向期间的峰值allocated bytes。旧工程夹具及旧报告保留；旧环境哈希与新环境不同，不混用其计时/内存记录。

## 实际验收

授权LXC、Python3.10.20、PyTorch2.7.1+cu128，解释器 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。现代目录 `checkouts/wp12-historical-cost-adapter02`，历史目录 `checkouts/wp12-historical-cost-worker02`。

```sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests PYTHONDONTWRITEBYTECODE=1
python -m unittest tests.test_wp12_historical_cost tests.test_wp12_resident_cost tests.test_wp12_cost_preparation tests.test_wp12_evidence tests.test_unified_cohort tests.test_repository_delivery -v
python scripts/check_repository.py --expected-commit a4b4408d993bfd9a93eb7ead4ce933974c58f543
# 在独立历史worker目录执行：
python scripts/check_repository.py --expected-commit ad107501595f9b9cb625086e8457a852d7a60789
python scripts/measure_wp12_historical_cost.py --expected-commit ad107501595f9b9cb625086e8457a852d7a60789 \
  --freeze <原freeze绝对路径> --freeze-sha256 7dd7e87334766d74005d14345e994fc2e700e109f6d8141b462004831496b48c \
  --out <新外部结果JSON> measure --job <固定job> --job-sha256 <实际jobSHA>
```

现代回归33项中32通过、1GPU实测跳过，43.334秒。新增CPU峰值边界测试保证记录取收尾校验后的读数；源码变化/缺文件/脏状态/错生产版本/错摘要拒绝。先前版本31通过/1GPU跳过、42.833秒日志保留，随后按修订版本重新测试。

两份干净源码仓库检查均通过：现代588 Python/198 JSON、历史360 Python/81 JSON。历史checker的旧main状态只属于该冻结快照，不用于判断当前main；本次没有合并main。

另在干净现代checkout通过实际 `assemble` CLI接受历史worker的单例工程记录；显式传错worker SHA时拒绝且没有成功输出。`single-fixture-receipt.json` 与预期拒绝日志 `wrong-worker.log` 保留。它是接口验收，不是54例正式成本回执。历史worker属于独立实验运行分支，不作为回退main的合并候选。

独立CLI在两个干净运行时上读取**同一原生合成模型及36像素观测**，完整三头输出和全部模型状态指纹精确相等，环境相等；没有读取query标签、更新模型或对真实场景重新评分。此单例等价性属于工程冒烟，不冒充54真实模型数值复算。历史准入实际核392原始文件一致，状态 `original_files_preserved`；现代分支8文件不同的原诊断仍保留。

证据根 `/home/lbx/hsi-pioneer271828/runs/wp12-historical-cost-engineering01/`：`cpu02.log`、两份checker日志、`historical-admission.json`、两份独立环境/job/measurement JSON。`comparison.json` SHA256：`5f1623797d2fa14faad60b1748fd7437b8dfa1116f4c46150aa18f8e1244d319`。

## 结论与剩余

CPU隔离采集工程通过，历史代码保持原样，原运行时准入问题已有隔离执行方案。当前没有真实GPU或全54硬件结果，没有新增分类精度/收益结论。最新资源检查8张卡空闲均低于12288MiB门槛，最大4156MiB，未启动GPU测试。

下一项为在此固定历史worker上登记并执行完整硬件测量，使用Issue365导出的同场景观测和原54权重，再接入Issue363精度汇总；常驻前向仍不能替代端到端成本。WP04真实短测继续后补，WP12整体真实精度与硬件成本验收仍未完成。
