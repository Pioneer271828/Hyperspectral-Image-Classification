# WP11 校准冻结与独立确认隔离

Issue #381 对应研究主计划 §22 WP11、实施计划 §19.12，接续 #377 和 #379。旧协议以完整预测和标签共同固定各阶段。新增稀疏标签协议，以确认标签的固定哈希代替冻结前的确认预测要求；确认阶段把实际完整概率和计时证据重新绑定。旧协议的工件和哈希合同继续通过原测试。

## 接入与工程验收

- `scripts/run_wp11_calibration.py` 只接收校准观测与校准标签包。确认标签与影像路径均不在输入中。先验证模型、历史、角色和光谱输入，再核验校准托管回执与块文件，最后读取校准标签。确认文件由后续阶段读取；确认入口继续核验固定门控、原始准入清单、校准请求身份与独立标签哈希。
- 沿用预登记阈值、逐类保护和最早并列顺序。支持不足时保留完整路径。冻结输出记录两个端点的实际预测身份；确认曲线拒绝同标签下替换端点预测的结果。校准和确认的工程接口均不打开 query。
- 实际测试为授权 LXC，Python `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，分支 `codex/wp11-calibration-freeze`，干净提交 `93926301cf3d94dafc4d6fae31bff8362591fb56`。`python -m unittest tests.test_wp11_calibration tests.test_wp11_registered_light tests.test_light_head_training tests.test_trained_light tests.test_wp11_confirmation tests.test_wp11_label_custody tests.test_wp11_held_aside tests.test_dynamic_cost_pairs tests.test_dynamic_curve tests.test_route_controls tests.test_gate_calibration tests.test_repository_delivery` 通过 64/64，13.765 秒。`python scripts/check_repository.py` 通过，604 Python / 198 JSON。
- 合成实际模型测试覆盖确认影像和标签文件均缺席时校准成功，冻结后恢复这两类工件再走确认和同成本报告。错误托管角色、错误输入哈希在读标签前拒绝，支持不足保留完整路径。独立合成 CLI `python scripts/run_wp11_calibration.py --expected-commit <tested-sha> --registration <calibration-spec> --registration-sha256 <sha256> --out <new-dir>` 复现冻结。回执 `/home/lbx/hsi-pioneer271828/runs/wp11-calibration-freeze-validation01/cli/receipt.json`，SHA256 `b2aca6e622c00efaaed8709a1be2dc4c41d2d19365de0edb245d2422baadb62f`。

## 徐州真实候选请求

按 #377 的实际角色计划准备 12 个校准块和 13 个确认块的候选请求身份。复用历史完整模型的图尺度 100 与固定 devtrain 归一化，回读 8,399,104 个原始光谱值并复核 107 个输入哈希。保留每块和每组候选请求哈希；获得授权后如有空标签块，按实际合法子集重算组哈希。模型前向、新标签、原始 GT、query、优化更新均为 0。

首次两次预检分别因历史文件哈希和开发包哈希转录多一字符而在输出前拒绝。后续从原始 `sha256sum` 和已核开发包回执取得正确值，并修正上一份报告三条转录。保留前两版外部脚本与原输入。成功回执 `/home/lbx/hsi-pioneer271828/runs/wp11-calibration-observation-readiness03/receipt.json`，SHA256 `ff50316e06175abf0aa27339ab0495d5e6580b72691c92d588827e9074075c56`。

实际门控结果需同划分训练轻模型与两份独立授权标签，工程验收和候选准备没有形成真实精度或提速结论。WP12 GPU 最高空闲显存 6000 MiB，低于 12288 MiB 门槛，54 例成本批次保持待资源。WP04 真实短测后补。
