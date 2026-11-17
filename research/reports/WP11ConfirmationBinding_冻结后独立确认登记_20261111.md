# WP11 冻结后独立确认登记

对应研究主计划 §22 WP11 和实施计划 §19.12，承接 Issue #383 的独立标签托管。校准前的准入文件绑定固定路径、计时预算、对照参数及确认标签文件哈希。校准回执绑定同一准入和设计。新入口 `scripts/prepare_wp11_confirmation.py` 用校准冻结回执生成确认登记，允许确认标签文件继续缺席。校准保留完整路径时只输出 `keep_full` 回执。

确认入口在读取开发包或确认标签前重新从冻结回执生成对照规则，核对实际固定路径、曲线参数和比较判据。此路径沿用原模型前向、覆盖回退、保护和同成本判据。已有未预承诺设计的合成协议仍可按原入口重放。新设计的登记不能省略冻结绑定或替换参数。

授权 LXC 使用 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，分支 `codex/wp11-confirmation-registration`，完整 tested SHA `662075e15532087102e1dd171f4b0395dae7b118`。CPU 单线程，执行 `python -m unittest tests.test_wp11_confirmation_registration tests.test_wp11_gate_registration tests.test_wp11_calibration tests.test_wp11_registered_light tests.test_light_head_training tests.test_trained_light tests.test_wp11_confirmation tests.test_wp11_label_custody tests.test_wp11_held_aside tests.test_dynamic_cost_pairs tests.test_dynamic_curve tests.test_route_controls tests.test_gate_calibration tests.test_repository_delivery`，通过 69/69，19.473 秒。`python scripts/check_repository.py --expected-commit 662075e15532087102e1dd171f4b0395dae7b118` 通过，608 Python、198 JSON。测试覆盖改设计、改确认标签哈希、修改曲线参数、删除冻结绑定、完整路径回退及合法延迟放行。

独立运行四个 CLI：`prepare_wp11_gate.py`、`run_wp11_calibration.py`、`prepare_wp11_confirmation.py`、`run_wp11_confirmation.py`，每阶段都使用文件路径、SHA256 和新输出目录。4/4 成功，合计 21.827 秒。前置登记时两套合成标签均缺席，校准只恢复校准包，生成最终登记后才恢复确认标签。最终合成确认 9 类各 1 个样本，包含实际路径执行和计时；夹具生成后优化更新 0、query 读取 0。此验收范围为接线、权限边界和可重放性。真实轻模型、独立托管标签、真实逐类保护与部署速度仍待。

证据 `/home/lbx/hsi-pioneer271828/runs/wp11-confirmation-binding-cli01/receipt.json` SHA256 `b2ce29c874641f68370a63eb205ea2d07e099775c45c6100c4db5e0ddb9d9ad5`。驱动 `runs/wp11-confirmation-binding-cli01.py` SHA256 `3949ed37c20eeb1ba76619a12c4ca068d80301c7e128377d372d05db09b1118f`。54 例成本批次登记保留，最新最高空闲显存 6992 MiB，等待 12288 MiB 门槛。
