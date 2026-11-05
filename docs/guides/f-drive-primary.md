# F盘主目录迁移与保留记录

2026-11-05，按用户要求，日常主目录统一为 **`F:/E2E_VCA`**，分支 **`codex/f-drive-primary`**。本次只整理目录、整合已有提交和更新交接，不恢复研究训练，不合并GitHub main。

## 版本范围

主分支基于M4交接 `0de46ee00946166a84c9ad50fd027bba2c56d679`，合入工具整合 `1c40f249e2277a4356d63c8e8db35d6aa9282791`。原模型、训练器、数据逻辑、实验配置与M4交接相同；新增合流包含MON-2/3、快照修复、数值审计及交付工具。源代码无语义冲突；六处文档/索引冲突按最新M4实验状态和单一JSON状态源合并。冻结运行合同明确改以M4交接作比较基准，未删除保护路径。

当前状态源 `docs/architecture/current-status.json`，人类可读清单由远程 `scripts/check_repository.py --write-status` 投影。最新交接为 `research/项目开发交接_20261105.md`。C盘各工作树保留，不自动同步，今后不要沿用旧聊天cwd作为编辑入口。

## 原有文件保留

原F盘分支 `master`，HEAD `dda3a3eba15949d58481357c28965d95e3bfd053`，保留未改。原有工作按以下两层保留：

1. 外部备份 **`F:/E2E_VCA_preserved/before-primary-20261105`**：848个原跟踪文件及新版同名冲突文件逐一复制并核对SHA256。`preserved-files.json`保存原路径/大小/哈希；`files/`保存内容，`displaced-untracked/`保存被新版替换的原未跟踪同名文件。`original-status.txt`保存完整旧状态，`original.index`保存原暂存区，`staged.patch`/`unstaged.patch`为Git二进制差异，`data-inventory.xml`为原数据/权重/日志/结果文件清单。
2. 本地专用stash **`f62205df00bd2cef47d00c4dddc8d45b38e5318c`**，保留原跟踪文件的暂存及未暂存状态；使用普通tracked-only stash，没有把数据目录或所有未跟踪文件打进stash。本地stash与备份均未推送GitHub。

原未提交修改没有自动叠加到新模型，避免覆盖已验收实现；内容和暂存状态均可恢复。`dataset/`、`weights/`、`logs/`、`results/`留在原位置；旧分支跟踪但新分支不跟踪的历史结果/数组等从核验备份恢复原位。其他不与新版冲突的未跟踪材料保留原位。保留的旧材料可能仍出现在`git status`，不是新分支待发布代码；不要对整个目录执行`git add .`。

没有移动F盘共享`.git`，没有删除/移动C盘工作树，没有执行reset/clean或全目录覆盖。不会修改远程生产checkout、原数据集、旧实验权重或历史标签。不要用`git push --mirror`或`git bundle --all`发送含本地stash的全部Git引用。

## 需要恢复旧工作时

不要在新主目录直接应用旧补丁。先检查`git worktree list`，在新的恢复目录创建旧分支副本，再应用本地stash的暂存区：

```powershell
Set-Location F:/E2E_VCA
git worktree add -b codex/recover-original-f F:/E2E_VCA_preserved/original-recovery dda3a3eba15949d58481357c28965d95e3bfd053
Set-Location F:/E2E_VCA_preserved/original-recovery
git stash apply --index f62205df00bd2cef47d00c4dddc8d45b38e5318c
```

恢复目录必须尚不存在；如出现冲突先检查，不强制覆盖。原未跟踪同名文件按`preserved-files.json`从外部备份逐项恢复到恢复目录；其他个人材料仍在F盘原位置。不要覆盖新主目录的同名模型或交接文件。

## 验证边界

本地仅文件读写、备份核对和Git；Python/JSON/静态检查、整合单测与GPU回归均在`ssh rmvision`训练容器运行。实际源码验收SHA为 `30165d628fef2bc52a495cd3dbbe89b78a2e1fdf`，分支 `codex/f-drive-primary`，Docker容器、Python3.12.3，解释器 `/home/rmvision/venvs/hsi-baseline/bin/python`，独立checkout `/home/rmvision/f-drive-primary-20261105`。研究仍暂停：M4-G4B源批次只有1完成、1严格重载失败、16未启动；目录统一和整合通过也不意味着该科学实验完成。

### 本轮结果

- 交付检查通过：300个Python文件语法、71个JSON、8个必需祖先，模型/训练/实验配置与0de46ee保持一致。
- 全量回归 **509/509** 通过（153.272秒）。
- GPU条件专项 **33/33**：M4迁移5（21.843秒）、完整源4（13.752秒）、监测干预10（1.747秒）、VCA兼容3（1.873秒）、冻结评价11（80.998秒）全部通过。CPU精确等价夹具仍按其原范围保留，不扩大为独立CUDA逐比特保证。
- 本机保留核验：848个备份哈希一致；881个原数据/权重/日志/结果路径、大小、修改时间不变；662个恢复原位的旧跟踪文件内容哈希一致。数据目录新增的唯一项目文件是dataset/README.md，不属于原始881文件。
- 日志在checkout外 `/home/rmvision/f-drive-{check,full}01.log`、`f-drive-gpu-{m4a,m4b,mon,cache,query}01.log`。
- 未读取真实query、未运行真实训练、未修改既有生产checkout。本轮GPU训练循环均为测试中的合成夹具。

实际命令（均在远程、干净完整SHA核对后执行；python替换为上述绝对解释器）：

```bash
python scripts/check_repository.py --expected-commit 30165d628fef2bc52a495cd3dbbe89b78a2e1fdf
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
M4G4_DEVICE=cuda OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_m4_transfer.py -v
M4G4B_DEVICE=cuda OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_m4_source_study.py -v
MON3B_DEVICE=cuda OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_monitoring_interventions.py -v
REPAIR_DEVICE=cuda OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_vca_release_compatibility.py -v
G4I_DEVICE=cuda OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_g4_query*.py' -v
```

后续仅Markdown交接/验收记录提交，不代替实际源码tested SHA；最终文档版本另跑交付检查，不重复宣称新增509+33测试。GitHub已推送新分支并设置跟踪`github/codex/f-drive-primary`；本机仍保留旧Gitee origin远程及master，不使用其作为当前开发入口。GitHub Issue/PR连接器本轮不可用，未新建迁移PR，也未合并main。若需评审，可从已推送分支建立PR。