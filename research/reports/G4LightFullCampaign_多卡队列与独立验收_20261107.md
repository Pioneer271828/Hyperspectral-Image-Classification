# G4 完整预算轻量适配多卡队列与独立验收

Issue247 / PR248 对应研究主清单 P4 §10.3 和实施计划 §19。接续 Issue245 的完整预算单例入口，为仅训练三头、VCA 加三头提供 2～3 卡登记队列和逐例独立验收。Issue249 使用原 Issue184 的三景、三结构、三个模型种子、K10/V5 和支持种子64，与对应全量微调配对。

## 实现与验收

- `g4_light_full_campaign.py` 按固定角色顺序分配独立队列。分派锁随子进程继承，避免协调器退出时重复占卡。每次起步、续跑、只读重入和验收各自过资源门槛。非资源失败保留日志和断点，停止新工作并让在途子进程退出。续跑只信持久化验证游标，异常中断丢失的未持久化物理步可能重做，逻辑更新预算仍按固定游标计数。
- `g4_light_full_audit.py` 核验 1500 步、51 次验证、首次最小融合 NLL、学习率重放、选中态和权威优化器/调度状态、冻结参数与缓冲区，以及实际可训参数更新。逐例独立复算三头 train/val 指标和逐类召回，做三次精确选中态重载并核对文件哈希。未选中事件核对证据哈希与历史。验收允许在内存中重建无标签 BN 初始校准，不保存新训练产物，也不读取 query。
- 实测环境为授权 LXC，Python 3.10.20、PyTorch 2.7.1+cu128、RTX4090。分支 `codex/g4-light-multigpu-campaign`，完整 tested SHA `2cfca12ae52308189c81c6d32dcea10f0ac1b50c`。CPU 4/4 用时 83.384 秒，物理 GPU1 4/4 用时 77.643 秒，仓库检查通过（413 Python、130 JSON），CLI 帮助检查通过。

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -p test_g4_light_full_campaign.py -v
CUDA_VISIBLE_DEVICES=1 LIGHT_CAMPAIGN_TEST_DEVICE=cuda:0 python -m unittest discover -s tests -p test_g4_light_full_campaign.py -v
python scripts/check_repository.py --expected-commit 2cfca12ae52308189c81c6d32dcea10f0ac1b50c
python -m g4_light_full_campaign run --help
```

双卡端到端用物理 GPU0、GPU1 和合成数据，运行原 worker 与独立验收子进程。两种方法各 1500 步，完成 2+1498+0 和验收后只读重入，总计 3000 步、146.557 秒。专项也覆盖资源等待、失败保留、部分持久化游标、进程继承锁及三卡角色划分。真实 CUDA 子进程强制中断、三卡端到端和全仓回归未执行。

## 证据与真实批次

证据根为 LXC `/home/lbx/hsi-pioneer271828/runs`。工程验收 `g4-light-full-issue247-engineering-acceptance01.json` 的 SHA256 为 `7c95ae22e456152cae4fc9a0f0bef0bab3989f140ec17e73b7ebd83f25906a79`。双卡合成验收 `issue247-synthetic-multigpu01/light/training-acceptance.json` 的 SHA256 为 `ef355c5d3420362be7ccc980934b5f423e0a5887771ed04bfeab73a4e169e088`。专项和端到端日志、合成审计收据均在 `issue247-export01.tar.gz`，SHA256 `dce2bf7119e0601cf1cbf0861de1fa176590b0b7742217fa7a5dd4f1b6c01359`。

Issue249 已用同一 tested SHA 启动物理 GPU1、GPU5 的真实队列。2026-11-07 15:46（北京时间）已确认运行。固定 54×1500＝81,000 步，双卡估计 6～12 小时，另计外部资源等待。登记 `g4-light-full-issue249-registration01.json` 的 SHA256 为 `b3cca8f1028004da30177704b107e959cd2fa47aa7892390522e4589fe39e735`，分析计划 SHA256 为 `c666ca8b0c1d78471b662ef900e0c6302fb8d7de3cbd2761eed42bf6da7c1d00`。运行根为 `g4-light-full-issue249-registered01`，协调器日志 `issue249-coordinator01.log`，值守每30分钟。

Issue247 工程验收通过，本聊天交付后继续搭建 Issue249 的全批次冻结门禁及同协议评价。真实适配收益要等完整训练接受、复盘、冻结 query 及预登记判定。
