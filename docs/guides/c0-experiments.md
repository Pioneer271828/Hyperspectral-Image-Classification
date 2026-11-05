# C0：配置与可追溯实验

所有 Python 命令均在 `ssh rmvision` 的训练容器执行。使用已同步、已提交且干净的代码，产物必须位于源码目录外。当前解释器可用 `/home/rmvision/venvs/hsi-baseline/bin/python`。

## 配置与已有行为

[示例配置](../../configs/experiments/c0_example.json) 使用版本 1 的 JSON；只声明当前已实现的能力。配置分别管理 `model`、`interaction`、`loss`、`training`、`data` 与 `augmentation`，以及名称/设备。省略字段采用现有多数据集训练入口的默认值，未知字段和未实现协议直接报错。`ResearchConfig` 和既有 `InteractionConfig` 共用交互语义，模型工厂仍创建原 `MultiDatasetVCASAHGCN`。

旧 `train_vca_multidataset.py` CLI 继续可用，默认模型、固定融合、三头损失、数据采样和选模规则不变；新增配置校验与 checkpoint 内的 `research_config` 元数据。旧入口自身保留历史输出路径行为。需要独占目录、失败记录和安全跳过时，使用本指南的新调度入口；原 `run_paper_ablation.py` 保留作历史兼容，不把它的旧输出当作 C0 已登记运行。

`epochs` 保留历史含义：循环是 `0..epochs`，因此 `epochs=0` 仍有一轮训练。默认 `steps_per_epoch=0` 时按实际准备成功的数据集数乘 3；实际优化步数写入 manifest。配置的初始计划按请求的数据集数估计，`skip_missing` 导致实际数量变化时记录实际预算。

`data.protocol` 当前只支持 `legacy_random_pixel`，仍是历史随机像素/整景流程。空间隔离、类别映射、原始文件登记与元数据质量在 C1 中建设；配置文件不能提前宣称这些能力。当前 band-drop 仍是历史置零增强，不是 M1 的真实波段子集。

## 先生成清单

```bash
python run_research_ablation.py \
  --config configs/experiments/c0_example.json \
  --experiments all --seeds 0,2026 \
  --out-root /workspace/hsi-runs/c0-plan --dry-run
```

生成完整解析配置、预算、源码指纹及 20 项任务的 `plan.json`，不加载数据、不构造模型、不训练。`configured` 只运行配置本身；指定 `interaction_baseline,interaction_full` 选择两组。命名预设复用已有 10 组定义；每组重置模式开关后独立应用，避免从模板继承已开启的门控。结构宽度、训练预算等仍以输入配置为准；同容量比较需要选择匹配组。

首次真实运行去掉 `--dry-run`；复用已生成清单时加 `--resume`。只有清单完全相同才允许复用输出根目录；修改配置或选择实验后使用新的根目录。

## 运行身份与目录

运行 ID 来自完整规范配置哈希与源码指纹，包含种子和设备；变更配置字段或受跟踪源码会产生新 ID。源码指纹覆盖受跟踪/未忽略的 Python 文件及 `configs/` 下 JSON，按 LF 归一化，另记录实际完整 Git SHA、分支和 dirty 状态。只改报告文档不会改变源码指纹；每次 attempt 仍保留实际 SHA。

```text
<out-root>/
  plan.json, planning_code.json
  <name>-s<seed>-<identity>/
    identity.json
    attempts/
      0001/
        resolved_config.json
        manifest.json
        training.log
        metrics.json
        model*.pt, model*.pt.json
      0002/ ...
```

每个运行只能有一个写入者。`.lock` 已存在时拒绝写入；进程被强制终止而遗留锁时，需要先确认进程和产物，再由负责人处理，不自动抢占别人的运行。新的失败重试采用新 attempt，不能覆盖旧权重/日志。

manifest 保存规范配置引用、代码和源码指纹、实际预算、框架、分组参数量，以及实际输入数据和监督划分指纹。数据指纹是**准备后**的 `net_input`、波段元数据、Q/A 与形状/区域尺度；监督指纹分别覆盖各集合标签、one-hot 和 mask。它能发现真实输入或划分变化，不等同于原始 MAT 文件版本或严格空间访问协议。C1 会补齐原始数据身份与协议边界。

## 恢复与完整性

`--resume` 的含义是验证跳过或失败重试，不恢复优化器/学习率调度器的中间轨迹。

- 已有成功运行：重新准备数据与划分，核对指纹，再逐一核对成功 attempt 的文件哈希；一致才登记 `skipped`，不会构造优化器或训练。
- 数据/划分变化，或成功产物缺失/损坏：明确报错，不静默跳过，不覆盖原运行。使用新输出根目录组织新的数据版本实验。
- 仅有失败记录：重建模型并从头开始新的 attempt，保留原失败状态、异常和日志。
- 批量运行默认首个失败即停止；`--continue-on-error` 允许继续其他组，存在失败时最终退出码仍为非零。

状态 `completed` 要求指标、主检查点及 sidecar 和数据/参数元数据完整；`skipped` 不表示执行了新训练。失败或中断均不计为成功。运行期间代码改变会使验收失败。

## 本模块验收

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python smoke_test.py --device cpu
```

C0 测试使用合成数据检查完整训练/记录/严格恢复/跳过/失败重试；现有回归继续检查交互和导出。原 `scripts/validate_baseline.py` 只用于冻结基线，不能为修改后的 C0 更新原始指纹来强行通过。当前模块实际测试提交与结论见交付报告，不能从合成检查推导精度提升。
