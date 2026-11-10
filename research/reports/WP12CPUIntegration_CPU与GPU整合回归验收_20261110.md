# WP12 CPU 与 GPU 整合回归验收

对应 Issue361、主计划 §20 与实施计划 §19.13 的工程回归。实测提交 `5d12d81ed77ab6a85d9bbabfdcbfdf0db57c6477`，授权 LXC 的冻结目录 `checkouts/wp12-cpu-integration01`。解释器 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，Python 3.10.20。测试前后工作树干净。

| 验收 | 结果 |
| --- | --- |
| 仓库检查 | 573 个 Python 文件、197 个 JSON 文件，冻结运行时及已接入模块检查通过 |
| CPU 全量发现 | 1125 项，1118 通过、7 跳过、0 失败、0 错误，2929.738 秒 |
| GPU 指定回归 | 物理 GPU 4。MON2 更新 10/10，MON3B 干预 10/10，VCA 发布兼容 3/3 |
| GPU 专用合成回归 | 同一提交、同一物理 GPU。匹配目标 worker 1/1（94.406 秒），资源切换 worker 1/1（99.698 秒） |

CPU 使用单线程和 `CUDA_VISIBLE_DEVICES=''`，运行 `python /home/lbx/hsi-pioneer271828/runs/wp12-cpu-integration02.py`。该脚本执行全部 `unittest` 发现并输出逐项回执。原运行 01 在生成回执之前结束，恢复连接后确认原进程不存在。运行 02 使用新输出目录，发现清单与 01 逐字节一致，回执中的 ID 顺序与清单完全一致。INT8 性能分析器在 CPU 环境打印设备查询告警，该测试返回 `ok`。

GPU 通过 `CUDA_VISIBLE_DEVICES=4` 串行执行。三组指定回归分别设置 `MON2_DEVICE=cuda`、`MON3B_DEVICE=cuda`、`REPAIR_DEVICE=cuda`，并运行对应测试文件的 `unittest discover`。两项专用用例设置 `MATCHED_TARGET_DEVICE=cuda` 和 `RESOURCE_TEST_GPU=4`，分别运行匹配目标 worker CLI 与资源切换 worker 的完整合成用例。两项均验证 2 步中断、1498 步续跑、完成后 0 步只读复核，以及 6 项三头指标、54 项逐类召回和 3 次精确重载。五次 GPU 启动检查均记录空闲 24082 MiB、利用率 0–10%。

CPU 的 7 项跳过包括上述 2 项 GPU 专用用例，以及 5 项要求使用历史固定提交 `f090ca302192a065de852c43e07e4f3beaeea6b4` 的历史兼容用例。后五项沿用既有验证记录。

| LXC `runs/` 下的证据 | SHA-256 |
| --- | --- |
| `wp12-cpu-integration01.log`（原中断记录） | `d9450d8521e3b9a7ac7978497a1b2bfb5874e7f4ace13ad097942e6ffb0bcb2d` |
| `wp12-cpu-integration02/inventory.json` | `a84a7fc04a93b7d23414c5ea505c0aa0bd33e795d19e165f603e75c1bb31ab5f` |
| `wp12-cpu-integration02/receipt.json` | `902bbde33163fced38b57fe5a3719a5d990f635afcb21b7f8ae735ea9a210d8e` |
| `wp12-cpu-integration02.log` | `3fcfede305767d57bcdb22f8b8e6a13626fd189ff7e6ef8dbeafbdde8728aba0` |
| `wp12-gpu-focused01/test_monitoring_updates.py.log` | `8e3872733eaddac5fe80445aaba72b0b7bd44bd5d4e863e5e83e1ace51800850` |
| `wp12-gpu-focused01/test_monitoring_interventions.py.log` | `7bc78f67fc062e1cfb4f2320cc7fe9d8ecd188561028e552fcc737e860232ded` |
| `wp12-gpu-focused01/test_vca_release_compatibility.py.log` | `87a60a0b9dbb31737fd6914d2e60be74089cc0dd8ddaabb6c8357802fc2b9a25` |
| `wp12-gpu-fixtures01/test_g4_matched_target_worker_cli.log` | `9e0114da60c87126060ded7c9752d8ebed1617b7990b891056358f664aedeace` |
| `wp12-gpu-fixtures01/test_g4_resource_handoff.log` | `e482bbdad3d6cb5947d8d0dace69bd16aea0fd811938df422e91361a5875ad0d` |

WP12 工程回归通过。真实训练、三种子统一评价、精度与成本结论仍需相应实验。此次验收为授权容器内手工执行。
