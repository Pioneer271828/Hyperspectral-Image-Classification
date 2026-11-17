# WP11 真实模型复用与独立标签托管

## 范围和结论

Issue #377 对应研究主计划 §22 WP11、实施计划 §19.12。完整模型历史、同划分开发包和无标签角色计划可用于后续登记；两份真实稀疏标签包等待各自的外部托管授权，同划分训练轻模型仍缺，真实门控校准和独立确认待验。

## 完整模型和轻模型

- 使用徐州 G3 的 `seed-2056/class_first/target_scratch_full-5000`。候选依据划分一致、从头训练及未耦合图架构确定。历史生产提交 `a9e7dcc4011c9925df0bc29d8eb755a403737828`。原训练 5000 步，532.499 秒，每 100 步开发验证，按融合验证 NLL 最早最小值选中第 4900 步。无源权重祖先。
- 比对 6 个历史源码文件、模型与侧车哈希，复算 devtrain 归一化，重放 5000 个采样块及锚定类，核对 50 次验证记录与历史最优项，用当前模型严格加载 CPU 权重。归一化、训练触达和保守选模范围分别为 73、46、5 块，与 25 个留出候选块无交集。原开发支持为 90 个训练标签和 45 个选模标签。
- 历史开发验证 OA 为 75.56%，逐类召回依类别顺序为 0%、60%、100%、20%、100%、100%、100%、100%、100%。本轮核对存档及选模规则，未重算历史预测。第 1 类历史零召回和第 4 类低召回保留，独立确认仍需遵循原逐类保护门槛。
- PaviaU `dy-cascade-pavia01/light.pt` 经 CPU 回读确认为 `vca-light-cnn-cpu-v1`，源自完整模型的 CNN 路径权重复制，缺少独立训练回执，场景也不匹配。已查阅的独立轻模型测试包属于合成工程验证；本轮未找到与本徐州模型、开发划分和训练回执匹配的真实 `vca-trained-light-cpu-v1`。全场传导式模型及空间短测均保留原有适用范围。
- 复用隔离导出器准备训练/选模开发包。78 块、26,076,288 个光谱值逐值回读一致。包内保留已授权的 90/45 标签；重建后的其余观察位置为零，不序列化留出或测试影像。

## 标签托管与验收

- `scripts/prepare_wp11_label_custody.py plan` 核对原几何、开发清单和 25 块无标签影像，再以空间行列顺序固定前 12 块为校准、其余 13 块为确认。角色选择发生在接触新标签前。每组真实类别覆盖待托管方提供，空间互斥与缓冲不证明统计独立。
- `package` 要求单角色托管授权，并在读取稀疏数据之前核对角色、用途、固定角色计划及来源哈希。仅接受该角色登记矩形内排序去重的局部索引和有效类别。若缺类，只写覆盖回执；两份授权申请已准备，等待实际签发。程序的授权可信来源由数据托管方与运行登记负责，文件格式校验本身不能代替授权。
- 授权 LXC，Python `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，分支 `codex/wp11-label-custody`，CPU 工程和真实材料核查提交 `6d17d001d7c396de5f20305f1893fbe379bd59fe`。`python -m unittest tests.test_wp11_label_custody tests.test_wp11_confirmation tests.test_wp11_held_aside tests.test_dynamic_cost_pairs tests.test_route_controls tests.test_gate_calibration tests.test_repository_delivery` 通过 32/32，5.241 秒。`python scripts/check_repository.py` 通过，600 Python、198 JSON。真实 `plan` 5.923 秒。真实原始 GT 读取、query 和更新均为 0；标签包释放路径用合成测试验收。
- WP12 复查最高空闲显存 6000 MiB，低于已登记的 12288 MiB 门槛。本次 54 例成本批次未启动。WP04 真实短测按用户安排后补。

## 证据与后续

外部证据位于 `/home/lbx/hsi-pioneer271828/runs/wp11-label-custody-validation01`，包含审计脚本和日志、`real-model-audit.json`、`full-history.json`、`development-roundtrip.json`、`role-registration.json`、`roles/role-plan.json` 以及各角色的 `*-custodian-request.json`。

| 文件 | SHA256 |
|---|---|
| `real-model-audit.json` | `057f86a14a521a98bd8133d81d15e1fb9b186223fbbb003d91f0bd17d2d50ae1` |
| `full-history.json` | `c7c2866d8e869b6f96b01e72eb751f7c6f809d269d8c2aadd510bf3d9d176dd1` |
| `development-roundtrip.json` | `04f8e9c14b38a5b726c92b2b33f1584b2550f54fdb564e99e36972d4c1fa2731` |
| `roles/role-plan.json` | `59db3398d7bcc8f548031de3dd838491dbe0a7d9026f29bfbf92876f78c0c9cb` |
| `calibration-custodian-request.json` | `4cb2caeb943ceffce7d5999ff715c302a6ae1e8d78b34c46fd18d7a430096018` |
| `confirmation-custodian-request.json` | `f257d1314cca22ea3063154cfcfa9c679b0cbb3c8e5a618f69a9e74837e12919` |

后续用同划分的开发包和完整模型历史准备可恢复的轻模型登记入口。真实校准与确认需两份单独授权，以及完成独立训练、严格发布回读的轻模型；保护门槛按预先登记执行。
