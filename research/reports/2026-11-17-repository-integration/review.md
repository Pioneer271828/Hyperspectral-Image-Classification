# 仓库整合最终审查（2026-11-17）

这次审查将已验收的工程代码和历史记录纳入 GitHub `main`。仓库状态由 [current-status.json](../../../docs/architecture/current-status.json) 维护。训练仍暂停。原 Windows 主工作树 `F:/E2E_VCA` 保持在 `codex/f-drive-primary` 的 `a339ce8d7e5f0a600363b1a8826b96d8a16ae794`。主审已只读检查原有 92 个跟踪文件的未提交差异，主要是注释和文档字串。这些修改没有纳入整合提交，用户工作现场原样保留。历史工作树、分支、标签和外部数据、权重及日志各自保留。

## 合并身份

| PR | 审查分支和头提交 | 实际远程测试提交 | `main` 合并提交 | 完整树 |
|---|---|---|---|---|
| [#494 低秩基础](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/494) | `3df9c70aad1842da2e3f0f8718355c5b7378d08c` | `3df9c70aad1842da2e3f0f8718355c5b7378d08c` | `9bc7fbfb51fd17b0df7f214fc2d407e2d8b901ee` | 合并树与受测树一致 |
| [#495 仓库整合](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/495) | `codex/repository-integration-20261117` · `36eb43eb530378fbe9af19f4ecf59614530175db` | `f95c2a120948b3a085f3c7b05adaf5c32e9ebf7b` | `eb7e5f20882551b6c879b1ba1a5931dc78ec2838` | `b0b894fca1c985fa934ac0b054aeedb191c8bd9d` |
| [#496 可选光谱元数据](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/496) | `codex/repository-integration-meta-20261117` · `60804913568165c50e54d80b70070e851117be52` | `eeb5b3f1807bb1ed95d01ee41ad5bb6723984d7b` | `d00aeea1744a0e627ef3a1c24108ee02666ab100` | `ddf774893cb9fe37f8f960115d11455442f2e2ec` |

#494 是 #495 的基线。#495 合并后，主审将 #496 的基线改为 `main`，再合并其独立增量。#495 和 #496 各自的实际测试提交、PR 头和合并提交的完整 Git 树一致，测试提交和 PR 头都保留在对应合并提交的祖先中。对合并树的核对不计为重新执行回归。

核心模型、数据加载、原训练入口和默认实验配置通过冻结检查。#494 独立验收记录为 7 个专项通过、1,125 个 CPU 全仓用例通过、7 个跳过。独立审查逐项复核了归档的 129 条清单记录、CPU 全集的 1,538 个唯一 ID 和六个分片的零退出状态，以及 CUDA、META 和课程历史回放日志。主审确认补交驱动的哈希、测试发现、唯一 ID 校验和失败退出。GitHub Actions 专用运行器仍待注册。#496 头提交的两个 validation 检查当时均在排队，结论为空。验收证据来自授权 LXC 的手动运行。

## #495 工程验收

- 受测分支使用完整 Git 历史。LXC 的最终候选检查运行 `python scripts/check_repository.py --expected-commit 36eb43eb530378fbe9af19f4ecf59614530175db`，解析 979 个 Python、208 个 JSON，冻结入口检查通过。工作流 YAML 可解析，其中四段脚本通过 `bash -n`。
- 最终全仓 CPU 用 Python 3.10.20 执行了发现的 1,538 项，六分片各 ID 恰好出现一次。1,499 通过、39 按条件跳过、0 失败，历时 1,693.4 秒，检出前后干净。跳过项及原因见外部归档的 `cpu-skips.tsv`。首轮 `e456645a33ad540deabe22559503e25a6c87c9ba` 的 1,497 通过、39 跳过、1 失败和 1 出错也保留。两项异常来自同一个测试子进程无法导入 `research_runtime`。修复测试入口路径后，未设置 `PYTHONPATH` 的对应模块 2/2 通过，随后完整回归通过。
- 冻结 WP10 类别夹具的原生产器为 `01b99515156694b09c250723b89277f475700897`。课程夹具的原始来源为 `9e27e88e0d3631ea6d8f557b6d3127353d768f32`、`4e1187a7d76575273f0bfd3d451e35abe3674c47`。比对运行文件或冻结路径后，原来源通过 Git 祖先关系保留，最终树与实际全仓测试的树一致。最终课程回放 3/3，双因素原回放 2/2。
- LP-FT 同设备及 CPU 卸载恢复 2/2，WP12 常驻 CUDA 前向与显存测量 1/1。短测按当时资源和物理 GPU UUID 选卡并持有共享租约。P4 双卡与 WP07 严格生产入口的合成 CUDA 检查在内层复核时触发 60% 利用率门槛。P4 重试记录为 15,082 MiB 空闲、69% 利用率，WP07 记录为 15,082 MiB、90%。这两项保留为资源准入拒绝。旧 G4 的两个长 GPU 夹具也没有在本轮启动。
- 使用候选工具对既有 WP10 的 18 个发布包、54 次独立进程首结果做只读复核，覆盖 810 个三头块哈希。WP12 迁移读入已封存的两次合成执行，检查 20 例、2,400 步、60 个 case/head，0 新前向、0 新优化，并验证合成证据不能登记成真实迁移。既有 GPU7 的 WP12 54 例冻结成本记录继续作为独立历史证据。
- #397 的历史初态桥在 `77242952ded4fdbaf0a566a59fa5cd52a9ab6c51` 验收 4/4。候选只将该桥的一个旧网络成功用例交给固定源码检出，其余三项仍在候选执行。#331 的五个固定源码成功用例在 `f090ca302192a065de852c43e07e4f3beaeea6b4` 通过，两个相关模块合计 16/16。另有 WP10/WP11/WP12 的 10 项归档回放、1 项受控图归档回放，以及 Torch-Pruning、PyG、TorchMetrics、TensorLy、TorchAO、Tent 和 OpenOOD 的定向对照。

全仓 CPU 的实际命令（在 `repository-integration-regression-nov17` 的干净检出中执行，目录与驱动均在 checkout 外）：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/lbx/hsi-pioneer271828/envs/hsi/bin/python \
  /home/lbx/hsi-pioneer271828/runs/nov17-cpu-suite-shards.py suite \
  f95c2a120948b3a085f3c7b05adaf5c32e9ebf7b \
  /home/lbx/hsi-pioneer271828/runs/repository-integration-regression-nov17/full-cpu 6
```

## #496 光谱元数据审查

原 #468 的 `92586890cc3d9efd2e187cb8aab959a3fcd1f4db` 可复现三种绕过。保持影像和坐标张量字节相同，只改像素 ID、数据分区或空间计划，原公开空间入口仍给出相同的三头输出。原 `restore` 会拒绝这些输入，但该入口未要求调用它。

#496 的专用登记入口经 `transfer/spectral_checkpoint.py` 核对包摘要、影像、像素 ID、数据分区、空间计划、选带和元数据，并将图尺度与 coverage 参考范围绑定到同一个受哈希保护的包。保存前拒绝不兼容模型和 FP64、FP16 或混合浮点类型，读取时再次核对包类型、参数及缓冲区。原数据加载、模型、两参数 attention hook、旧 checkpoint API 与默认调用保持原样。

CPU 专项 32/32、CUDA 合成专项 4/4。最终候选再次通过 checker（984 个 Python、208 个 JSON）、CLI 帮助和四项绑定及精度检查。三个既有冻结场景的 train/val 只读审查核对输入摘要和旧 VCA 输出，0 更新、0 held-out 访问、0 raw GT 访问。参考登记仍缺可获准使用的真实逐带校准表，这项审查未验证真实标定或精度收益。

## 历史真实实验的边界

- P4 保留既有 10/20 例、1,200/2,400 步，剩余 10 例待验。WP07 的 2,910 步与 11,058 步是两份独立的真实计划预算，均未放行。历史合成资格首次出现 cuBLAS 启动错误、0 更新，后续独立计划保留 24/48 步配对及完成态重入 0 步的结果。
- WP10 的 CPU 首结果、WP12 既有 GPU 成本和冻结课程回放各自沿用原协议。WP06 的相容真实端点、独立确认和逐类保护等仍按各实验报告验收。本轮未启动真实训练、长训练、query 计分或接管未完成队列。

## 旧 PR 与分支头处置

主审逐条确认旧 PR 的原 head 未变，记录替代范围和实验限制后关闭 77 项。75 项由 #495 替代，#467 与 #468 由 #496 替代。加上此前由 #494 替代的 [#270](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/270)，共 78 个旧 PR 以替代方式关闭。实际合并的是 #494、#495、#496 三个交付 PR。处置完成后，主审核对当时的 GitHub 开放 PR 列表为空。原分支、标签和工作树仍可追溯，科学待办 Issue 保留开放。

[77 个所替代旧 PR](pr-dispositions.tsv) 的 `candidate_sha` 为早期逐 blob 比较基线 `e456645a33ad540deabe22559503e25a6c87c9ba`，`exact_blobs`、`different_blobs`、`absent_paths` 对应该基线。实际合入的 PR 头见 `review_candidate_sha`，主分支合并点见 `merge_sha`。#416/#474 的已知源旧报告已通过 `da485f7047391067c78259cfc7a0b13684dc1e1f` 恢复到整合候选，保留原 Git ref 以便追溯。

[68 个无 PR 分支头](no-pr-dispositions.tsv) 以冻结盘点为准，按唯一头提交计数，可能对应多个引用。表中 `audited_against_code_sha` 是逐文件审查基线，`doc_only_successor` 是补齐历史文档的后继。41 个在候选保留，3 个原样保留，2 个分别以兼容适配和无调度入口保留。15 个属于主线补丁等价历史，2 个由有边界的整合替代，另有 2 个仅部分保留至状态后继、2 个 META 分开核验、1 个无关档案保留。无共同祖先的归档仍可从原引用获取。

## 复核材料

原始日志、测试清单和运行产物在 checkout 外。LXC 归档为 `/home/lbx/hsi-pioneer271828/runs/repository-integration-delivery-nov17/repository-integration-20261117-evidence-final.tar.gz`，SHA-256 `075c9faa3a9d63455e6595f6c702321aafc8b5cea64d6ba488121c14ffb04d4f`。归档含 `SHA256SUMS`，可用 `sha256sum -c SHA256SUMS` 复核 129 条证据。正式与首轮 CPU 分别在 `main-final/full-cpu/` 与 `main-previous/full-cpu/`，GPU 拒绝、META 三种绑定绕过和完整树比对见相应的 `main-final/`、`meta-previous/`、`lineage-final/`。

CPU 驱动原文件是 `/home/lbx/hsi-pioneer271828/runs/nov17-cpu-suite-shards.py`，已原样补交到 LXC 外部证据目录 `/home/lbx/hsi-pioneer271828/runs/repository-integration-delivery-nov17/nov17-cpu-suite-shards.py`。文件大小 5,279 字节，SHA-256 `52b869b2395010ebcb76fbb3023977871a39b475363cb1a3e307538a4f857585`。它是独立于原归档的补充证据，补交时未重跑测试，也未改写原归档。

外部盘点保留 `nov17-all-pull-requests.tsv`（263 组原始头，SHA-256 `b13f172a062efd2830ab7979d12493927b50bf080f1b1e0c2235ffb6e983c354`）和 `nov17-candidate-file-provenance.tsv`（4,302 条来源与候选 blob 记录，SHA-256 `0a60d4b8a14eb38f633ed155a732d31533fd538340386428d1e17d340eacb173`）。两个仓库内的处置表用于审阅每条决策，逐文件来源仍以该冻结清单为准。

仓库内的状态清单由 LXC 上的 `python scripts/check_repository.py --write-status` 从 JSON 投影。文档提交在独立完整历史的检出中使用 `python scripts/check_repository.py --expected-commit <文档提交完整 SHA>` 检查。
