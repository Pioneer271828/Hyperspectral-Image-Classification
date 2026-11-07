# G4 轻量适配全54冻结门禁工程验收

对应研究主清单 P4 §10.3、实施计划 §19 和 Issue249 预登记中的冻结前置条件。新增 `g4_light_frozen_boundary.py`，从固定登记、完成状态、完整接受回执和54份独立审计收据建立冻结清单。逐角色核对卡号、1500步/51次验证、三个精确重载、六项三头 train/val 指标、逐类计数、选中权重及侧车、原184配对角色/初态/支持与验证标签身份和保护文件。冻结产物在新同级目录，清单记录零新更新与零 query 访问。

实测分支 `codex/g4-light-frozen-boundary`，完整 SHA `480b8a96e6db3660c2a40a63aaa6980ded01fa3e`。授权 LXC 独立检出 `/home/lbx/hsi-pioneer271828/checkouts/issue249-freeze-boundary01`，Python 3.10.20，CPU 专项 6/6、0.718秒，CLI 帮助和仓库检查（415 Python、131 JSON）通过。

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_repository.py --expected-commit 480b8a96e6db3660c2a40a63aaa6980ded01fa3e
python -m g4_light_frozen_boundary --help
python -m unittest discover -s tests -p test_g4_light_frozen_boundary.py -v
```

专项使用合成的全54角色元数据和替代权重字节，覆盖成功冻结以及未完成状态、缺收据、审计角色失配、权重篡改和路径重叠。用 Issue249 最早已接受的三例验证实际审计/结果字段与原184参考身份接入。实批次在运行时执行入口，确认按完成状态拒绝，未写冻结目录。单例接口核验使用收据内的哈希表，真实全量文件重读待训练完成后的冻结运行。

证据位于 `/home/lbx/hsi-pioneer271828/runs`。专项日志 `issue249-freeze-boundary01.log` SHA256 `76ce22444aa20616dff5d86ccaa03f811dac5ecce3c4968f72b803ec22fc8351`。三例接入和提前调用拒绝日志 `issue249-freeze-live-boundary01.log` SHA256 `6346c2c5180957f20d211b84e5df13973f673db12b6b5e75746c3efedb66d0ff`。真实提前调用的拒绝栈 `issue249-freeze-running-cli01.log` SHA256 `9167e1c560ecc4fb8efffa30fd108054fc4ae3b8934579f8e0bab2e00ebcdd8f`。

门禁工程验收通过。GPU 和全仓回归未运行，本模块没有前向、权重反序列化、优化或 query 读取。真实全54训练接受、训练复盘、冻结正向、同协议 query、预登记比较和成本结论仍按顺序推进。本聊天2026-11-07交付门禁工程进度。
