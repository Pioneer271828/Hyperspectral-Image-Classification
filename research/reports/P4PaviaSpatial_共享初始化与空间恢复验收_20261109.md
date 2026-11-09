# P4 PaviaU 四组空间短测与恢复验收

对应研究主清单 P4 §10.3 第 2、3、5、6 项，实施计划 C1 §4、§19。使用固定空间块和已通过独立审计的四个角色发布，覆盖支持种子 512/2048 × K1/K5。每组从头与共享迁移各一例。训练程序只接收原影像和已发布的支持/V5标签。验证按最早最低融合 NLL 选步。

## 初始化合同

认证源 checkpoint 仅登记 Xuzhou 498 次和 HongHu 504 次更新，侧车的两个任务头也与此一致。源采用静态图。目标采用逐块学习图、PaviaU DomainBN 和三个新头。迁移从四个明确命名的共享子树复制 29 个同名、同形、同 dtype 张量，其中包括 VCA 的 3 个缓冲张量。源域归一化和源头不进入目标。初始化后对目标完整状态执行严格加载，并逐项核对共享值与源一致。

目标图由原模型构造器在模型种子 64 下初始化，6 个学习图参数的初始哈希在所有从头/迁移对照间相同。目标域归一化的 25 个状态张量及三个头的初始哈希也相同。两方法共享训练块抽样序列。训练继续沿用原三头监督损失、Adam、学习率和有界 12 步预算。

## 实际环境和验收

实现分支 `codex/p4-pavia-spatial-resume`，实际 tested SHA `89253cbb5fab42762b0129e8522b13b9ff3dbebf`。LXC checkout `/home/lbx/hsi-pioneer271828/checkouts/p4-pavia-spatial-resume01`，Python `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。两条单线程 CPU 队列，每例比较原图连续 12 步与测试块影像置零后的 2+10 步，另调用一次完成态恢复。总计 192 次实际更新。登记到独立回执约 198.330 秒。

| 核验项 | 结果 |
| --- | --- |
| 连续与恢复 | 8/8 对模型、Adam、调度器、RNG、训练轨迹、验证历史、选中状态逐位一致 |
| 空间边界 | 两份输入哈希不同，测试块置零后训练统计与上述状态逐位一致。重新计算发布哈希后把支持点换到测试块，仍被拒绝 |
| 恢复保护 | 各拆分运行 2+10/完成恢复 0。改用另一份登记恢复被拒绝，原状态文件哈希不变 |
| 新图 | 全部 192 步均通过学习图梯度为正及参数实际变化检查 |
| 选中态与逐类 | 各路径严格重载选中态后概率逐位一致。独立核对 24 个验证头与 216 项逐类召回 |
| 数据权限与回归 | 14 个冻结文件不变，训练与审计读取原始 GT/query 均 0，CPU 回归 29+2 全过，checker 321 Python/128 JSON 通过 |

| 支持种子与 K | 从头选步 | 共享迁移选步 |
| --- | ---: | ---: |
| 512/K1 | 7 | 4 |
| 512/K5 | 12 | 4 |
| 2048/K1 | 12 | 4 |
| 2048/K5 | 1 | 4 |

八例选中态融合验证 OA 均为 11.11%，各有 8 个类别零召回。该结果记录真实工程短测的拟合限制，完整预算效果与最终空间测试仍待验收。

## 命令与证据

在实际 checkout 使用上列解释器，设置 `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。

```bash
python /home/lbx/hsi-pioneer271828/runs/p4-pavia-spatial-register01.py
python -u /home/lbx/hsi-pioneer271828/runs/p4-pavia-spatial-worker01.py 0
python -u /home/lbx/hsi-pioneer271828/runs/p4-pavia-spatial-worker01.py 1
python /home/lbx/hsi-pioneer271828/runs/p4-pavia-spatial-audit01.py
python -m unittest discover -s tests -p 'test_spatial*.py' -v
python -m unittest discover -s tests -p test_development_training.py -v
python scripts/check_repository.py --expected-commit 89253cbb5fab42762b0129e8522b13b9ff3dbebf
```

两条 worker 调用 `python -m transfer.p4_spatial --registration <路径> --sha256 <登记哈希> --case <编号> --out <目录> --until <2或12>`，每次恢复启动独立进程。证据根 `/home/lbx/hsi-pioneer271828/runs/p4-pavia-spatial-real01/`。

| 文件 | SHA256 |
| --- | --- |
| `registration.json` | `fc61d64f204a78595856cd84394809911cce89eb83e19a097ecb3fb1980d4ad2` |
| `registration-split.json` | `22680bb3534618ec0db3a0aa9d91e2f7ce1030c27eb14665f8b217ec4f40064e` |
| `initialization.json` | `096d1e9155a8406919505212dd6209ff3d9047e831a0b5bd49fde8193c32243e` |
| `acceptance.json` | `413b23ab9350201bbd2cf10de675ff355a06a2f7d2207774ec487da709d6509c` |
| `regression.log` | `dc12727b83588ba376c9e5b371c42c72143aaf9c87086f77b4d56c39b8aa27fb` |

历史静态→学习图严格整网加载拒绝、错角色发布和错登记恢复拒绝均保留。正式短测无训练或审计异常。固定几何下 Xuzhou 类 3、HongHu 类 15/18 的验证缺口仍保留。

严格传感器留出预审已启动。`runs/p4-g2-source-family-preflight01/receipt.json`（SHA256 `ce6ebf38d1e99fbb2f4d13477f90d961aac82f47298217b484eeb9f12c6b1e25`）核对三个登记源结果和头侧车。现有 checkpoint 绑定各域独立头，尚缺认证的共享语义映射和祖先/选模访问记录。三条候选均涉及家族身份待核的 Xuzhou。WHU-Hi 目标的排除清单应覆盖 LongKou、HanChuan 与 HongHu。该回执只记录预审阻塞，下一阶段继续准备证据和拒绝测试。
