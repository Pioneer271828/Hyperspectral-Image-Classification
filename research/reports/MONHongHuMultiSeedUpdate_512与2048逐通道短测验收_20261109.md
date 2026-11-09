# MON HongHu 512 与 2048 逐通道短测验收

对应两份 2026-10-28 主计划的 MON-2/3、实施计划 8b 和 §19。接续 Issue115 的历史真实训练状态与 PR262 的逐轨迹小类筛选，补齐模型种子 512、2048 的梯度和实际更新证据。每种子两个历史状态各独立执行一次原 Adam 更新，共新增 4 步。两个种子均无合格候选，保留全部 128 通道。本项短测验收完成，MON 长窗口及正式压缩效果仍未完成。

## 环境、身份与预算

- 授权 LXC `lbx@10.69.37.243:1222`，`systemd-detect-virt` 返回 `lxc`，Python 3.10.20，解释器 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。
- 干净 checkout `/home/lbx/hsi-pioneer271828/checkouts/mon-trajectory-screen01`，分支 `codex/mon-trajectory-screen`，完整 tested SHA `4695896d38eab523c1b46d92b664102541e0af69`。复用既有训练与监测实现，源码没有修改。外部驱动仅替换种子、冻结输入哈希、历史游标和输出身份，原驱动及逐行 diff 均保留。
- 单线程 CPU，与 seed64 短测的运行路径一致。GPU 训练及 GPU 专项 0。本轮未新增或重跑单元/全库回归；此前 CPU 监测 42/42 属于历史证据，不列为本轮结果。
- 运行前冻结两个种子共 4 次更新。以原 seed64 游标 393、1203 为参照，仅按原调度选择最近的、下一任务确为 HongHu 的已保存状态，等距取较早状态。512 为 363、1173，2048 为 393、1203。选择不使用损失、验证分数或 query。
- 各自恢复模型、Adam、CPU/Python/NumPy RNG，核对历史 trace 和模型/优化器指纹；保留原三头损失、学习率及数据。CPU 独立续一步不声称精确重放历史 CUDA 轨迹，也不写回原训练状态。

从上述 checkout 依次执行下列命令，产物均在 checkout 外。

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python -u /home/lbx/hsi-pioneer271828/runs/mon-honghu-update-seed512-short01.py
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python -u /home/lbx/hsi-pioneer271828/runs/mon-honghu-update-seed2048-short01.py
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python -u /home/lbx/hsi-pioneer271828/runs/mon-honghu-update-multiseed-audit01.py
```

## 实际结果与独立验收

| 模型种子 | 独立实际更新步 | 两次三头总损失 | 短测墙钟 | 最终候选 |
| --- | --- | --- | --- | --- |
| 512 | 364、1174 | 5.757894、1.101908 | 171.394 秒 | 空 |
| 2048 | 394、1204 | 5.589934、1.167650 | 168.610 秒 | 空 |

两次损失来自各自独立恢复的历史状态，不能作为本轮连续训练曲线或充分收敛证据。两个短测合计 340.005 秒。六个关联参数张量覆盖第一段点卷积、深度卷积权重/偏置、第二段 HongHu 域 BN 仿射权重/偏置、第二段点卷积输入轴。

4 个步骤×6 个张量×128 通道，共 **3,072 组参数通道记录**。所有张量的全部通道均有非零梯度；512 第 1174 步的 HongHu BN 权重有 127 个通道实际更新，其余张量/步骤均为 128 个。单个未变化元素不直接判为可剪通道。

独立审计从保存的更新前/后权重与原始梯度，用 NumPy 复算 18,432 个标量读数，含梯度 RMS、更新 L2/RMS、更新前权重 L2/RMS、相对更新。核对原始权重等于历史状态、全部输入哈希、更新计数和候选交集，4/4 步通过，耗时 5.308 秒。每种子保护 9 个文件，合并去重 15 个均不变。审计新增优化步 0，训练和审计未出现数值失败。

共享参数计数分别对应全局游标。HongHu 域 BN 的独立计数为 seed512 的 132→133、433→434，seed2048 的 135→136、406→407，均由原任务 trace 复算。激活 RMS 和小类桶进行了 schema/有限值检查，未从完整原始激活独立复算，不扩大审计范围。

## 为什么继续保留全部通道

- seed512 通道 109 在第一段点卷积的梯度 RMS 排名为 6、3，相对实际更新排名为 50、97。
- seed2048 第 394 步通道 27 的同一张量梯度排名为 1，相对更新排名为 62；通道 75 分别为 2 和 47。通道 27 在第二段点卷积输入轴上的相对更新排名为 114。
- 排名从低到高，最低 16/128 为原门槛。每个种子要求同一通道在两步、六张量上的梯度 RMS 与相对更新都处于最低 16，再与此前逐轨迹的三场景/小类活跃度名单取交集。两个种子仅梯度与更新交集已为空，最终名单均空。

HongHu 原固定验证门槛保持为融合 OA、AA、逐类召回下降不超过 0.5 个百分点，小类不新增错误。小类仍按原训练支持≤25 定义，HongHu 为类 15、21，训练支持分别 11、14，既有验证支持分别 6、7。没有合格候选，按原协议跳过验证干预，因此本轮没有新的逐类召回读数，也不声称验证精度提高。验证标签和 query 标签使用均为 0，既有 HongHu 失败与小类证据保留。

加上此前 seed64 的 2 步/1,536 组记录，三个模型种子的短测证据共 6 次独立更新、4,608 组记录；seed64 未重跑。两个采样步不代表长窗口持续低活跃，也不证明所有通道绝对不可压缩。本轮结论是原门槛下没有可晋级候选。

## 证据与下一依赖

远程证据根 `/home/lbx/hsi-pioneer271828/runs/`。

| 文件 | SHA256 |
| --- | --- |
| `mon-honghu-update-multiseed-registration01.json` | `029ab7b86d47823115ea0b55af18c2809ed19745ca55a654828b61e446d140e5` |
| `mon-honghu-update-seed512-short01/receipt.json` | `a09590d57dd5722fdb88cc748384ef822752c5d50bd6ac579b476f6c0aa0e9d3` |
| `mon-honghu-update-seed2048-short01/receipt.json` | `8b199b124c4fc599034b7eef4dfa33976b456a2e83c57168cfed5bd17a5e8b577` |
| `mon-honghu-update-multiseed-independent01.json` | `ed26feaef95044541449181c46c5ee50080e9425a63e65e2fb4c8f8920a3e76a` |
| `mon-honghu-update-multiseed-audit01.py` | `6aefec368dea9ef26b31583d8356b209574d0de1c8d2396db43e232d5588ce79` |
| `mon-honghu-update-seed512-short01.py` | `265af49b968669c06a2c46166c2db69f88d5126aca843a972d0cd29cb798fd7e` |
| `mon-honghu-update-seed2048-short01.py` | `e78d18a39e1c646f22b044df45b2328a0de7eee79916f1cf87e83a49113776660` |
| `mon-honghu-update-multiseed-evidence01.tar.gz` | `314aa653a4e81460d9bf84b721c9fc29786f3ef82fc3a35900400c827f203a97` |

外部本地副本 `F:/E2E_VCA_worktrees/mon-honghu-update-multiseed-evidence01.tar.gz`，含原始张量、逐通道 telemetry、驱动差异、登记与独立审计，不提交 Git。报告、状态及连接偏好进入主入口；没有合并未评审训练源码或 main。

按用户 2026-11-09 最新要求，重心转回 P4 §10.1–10.3 与实施计划 §19。先复用独立支持集/K 的既有工程，核实真实短测缺口，再推进空间隔离及可成立的传感器家族留出；元信息/语义映射不足不能标为严格传感器泛化。MON 长窗口、其他场景配对更新、正式裁剪/完整恢复、GPU/部署成本与量化剩余项继续登记，不为寻找剪枝候选新增长训练。

本聊天 2026-11-09 seed512/2048 HongHu 逐通道短测为本阶段交付记录。
