# 远程存储与项目边界

本项目与容器中的其他训练项目共用资源，所有新数据与后续运行产物进入专用根目录：

```text
/home/train/hsi-pioneer271828/
  PROJECT_BOUNDARY.txt
  datasets/
    incoming/<snapshot_id>/   # 上传暂存，未校验不能用于正式实验
    snapshots/<snapshot_id>/  # 校验后只读发布，保留原始目录和文件
      dataset/
      manifest.json
      verification.json
    current -> snapshots/<snapshot_id>/dataset
  runs/                      # 后续训练、检查点、预测和诊断产物
  transfer-records/           # 传输脚本、日志与操作记录
```

`/home/train` 当前是独立 ext4 持久卷挂载；记录时空闲约 927 GiB。专用根目录以外的目录、项目环境、进程、数据及产物不属于本任务的写入范围。不停止其他训练，不修改其他项目依赖，不做共享目录的批量清理。

## 本次数据快照

- 本机来源：`F:/E2E_VCA/dataset`。
- 预期文件：71 个，共 1426421671 字节，约 1.33 GiB；包括六个主要数据集及原有训练/测试/划分参考文件。
- snapshot_id：`local-dataset-20261029-850ce34e74ea`。
- 原始清单 SHA-256：`850ce34e74ea116211a0d4c95178f33d213cf44c487f58f543dc021cfa671132`。
- 发布状态：71/71 文件 SHA-256 一致，六组影像/标签检查通过，已只读发布。证据见 `research/reports/data_transfer_verified_20261029.json`。

以清单的相对路径、字节数和逐文件 SHA-256 核对完整性，成功后才从 `incoming` 发布到 `snapshots`，并创建 `current`。只读发布防止普通流程原地修改原始数据；需要新数据版本时创建新快照，不覆盖旧快照。快照完整性不代表已有划分适用于新的研究协议。

## 代码与环境

现有冻结基线 checkout 在 `/workspace/Hyperspectral-Image-Classification`，C0 checkout 在 `/workspace/hsi-c0`，两者继续保留。当前本项目解释器为 `/home/rmvision/venvs/hsi-baseline/bin/python`；不修改其他项目环境。已有旧合成验证产物保留在 `/workspace/hsi-artifacts`，后续正式运行优先使用专用根目录的 `runs/`，不要混入源码目录。

当前 C0/C1a checkout 的 dataset/ 保留 README，并添加四个指向固定快照的目录链接；PaviaU 已通过旧读取器验证。新登记入口显式使用 --data-root。历史缺数据负向命令不再适用于已就绪的数据入口，不能把它当作快速测试而启动完整训练。数据不进入 Git；仓库只记录来源摘要、快照引用和校验结论。所有读取验证、依赖安装及代码执行仍在远程进行。
