# MON HongHu 失效诊断与逐轨迹小类保护

对应两份 2026-10-28 计划的监测与结构剪枝、实施计划 8b 和 §19。接续固定编号的三场景三种子复验。诊断沿用已有概率，修正后的候选筛选只使用已完成 M4-J2 / Issue #115 的历史检查点和训练标签。

## 为什么 HongHu 失败

- 原规则从 PaviaU seed64 选出低幅值通道，再复用编号。seed64 通道 44 在 PaviaU 五个快照的 RMS 低幅排名为 5/2/3/3/3，在 HongHu 为 35/124/122/92/121。其他两种子没有通道对齐。PaviaU 上的全局平均幅值不能代表 HongHu 或另一模型的依赖。
- HongHu 有 22 类，固定验证集 1947 点，其中第 4 类 817 点。三个基线的融合头前两类概率差中位数为 0.814–0.866，PaviaU 为 0.979–0.984，Xuzhou 为 0.988–0.993。在 HongHu 的 9 个单关条件中，原本正确变错误共 36 次、原本错误变正确共 31 次，按条件计可重复出现。翻错点在关闭前真类对最终竞争类的概率差中位数仅 0.0100。
- HongHu seed64 关闭 116 时，第 18 类有一个点从正确变为预测第 10 类，真类对该竞争类的概率差从 +0.07219 变为 −0.03622。第 18 类只有 17 个验证点，召回 16/17→15/17，下降 5.88 个百分点。HongHu seed512 关闭 44 时第 14 类 34/37→33/37，尽管整体 OA 略升，逐类仍失败。各条件图头概率变化最大约 5.07e-7。证据定位到关通道后融合决策翻转和类内样本数对百分比的放大，未分离物理波段、空间纹理或传感器因素的因果贡献。

## 筛选修正及验收

新增 `monitoring/historical_screen.py`。每个种子单独筛选，以相同 303/603/903/1203/1503 步冻结快照，在三个场景原训练位置复测第二段输入 BN 输出。每个场景的整体训练像素以及训练支持≤25 的各小类，都要在各自条件下将该通道排入最低 16/128。取全部条件的交集，最多 3 个。空交集保留所有通道。最终验证仍沿用 0.5 个百分点和小类零新增错误规则，筛选阶段不读取验证标签。

- 干净分支 `codex/mon-trajectory-screen`，tested SHA `4695896d38eab523c1b46d92b664102541e0af69`，授权 LXC `/home/lbx/hsi-pioneer271828/checkouts/mon-trajectory-screen01`，解释器 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。
- `python scripts/check_repository.py --expected-commit 4695896d38eab523c1b46d92b664102541e0af69` 通过。CPU 单线程 `PYTHONPATH=.:tests python -m unittest discover -s tests -p 'test_monitoring*.py' -v` 42/42，其中新增两项。GPU 专项和全库回归本轮未执行。
- 用同一解释器从干净 checkout 执行 `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 python -u /home/lbx/hsi-pioneer271828/runs/mon-trajectory-screen-real01.py`。三种子 45 个场景与步数组合、105 个小类分桶，CPU 墙钟 624.052 秒。独立程序从保存的 RMS 复算全部交集，重验 29 个原输入文件。更新 0，验证和 query 标签读取 0。验证入口 `mon-trajectory-validation01.py` 收到空名单，按计划跳过候选干预，留存回执。

| 种子 | 仅跨场景整体低幅值 | 加小类保护后 | 一项否决证据 |
| --- | --- | --- | --- |
| 64 | 124 | 空 | PaviaU 第 3 类，124 在末步排名 99/128 |
| 512 | 109 | 空 | PaviaU 第 9 类，109 在末步排名 35/128 |
| 2048 | 27、75 | 空 | HongHu 第 15 类，27 在末步排名 19/128；PaviaU 第 7 类，75 排名 69/128 |

修复的是候选筛选和空名单处理。当前所有通道保留，原保护门槛与验证失败记录保留。历史快照证据仅覆盖相对激活幅值；后续需补实际训练时配对的梯度和更新，再按固定验证保护判断是否有候选可裁剪。

## 证据

以下位置相对于 `/home/lbx/hsi-pioneer271828/runs/`。

| 文件 | SHA256 |
| --- | --- |
| `mon-historical-margin-diagnosis01.json` | `f750819e937306a583037cf20d664a4900ab02a11a8bd909dbc7e1e448f7496f` |
| `mon-trajectory-screen-real01/plan.json` | `00b0a27da4970145a64d6e25efc03ed5db27ca62d55ac9b7b2aafd3407d65afc` |
| `mon-trajectory-screen-real01/receipt.json` | `2336df72cfc1264aeadb8c7e6cfcfc4e5defddc55a54000d7717f2141b483403` |
| `mon-trajectory-screen-independent01.json` | `bf75728baaf1fc6392df0c4a413ff5ca77a2844d4f59e7b0816dde0eb47e9dd9` |
| `mon-trajectory-validation01/receipt.json` | `e2a88fad022cb3a7589e025f0ff0e394671ce4426547ecbab88959d0304ec48f` |
| `mon-trajectory-screen-tests01.log` | `90a356118da707209026e5d6044c5b0c516f557e5ce355abfa142ac4036eedf2` |

本聊天 2026-11-08 HongHu 失效诊断与筛选修正为交付记录。
