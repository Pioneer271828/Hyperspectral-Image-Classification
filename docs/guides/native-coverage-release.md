# 原生开发数据、缺带覆盖和冻结测量

该入口面向授权 Linux 容器。工作树须干净，`--expected-commit` 使用实际的完整 SHA。Windows 用于编辑、Git 和传输。运行目录、真实 Bank、旧权重与检查点都在检出目录之外。安装和依赖参见[远程验收指南](remote-validation.md)。验收环境复用 Python 3.10、PyTorch、NumPy 和现有科学计算依赖。需要本机复核版本时，在容器执行 `python -c 'import torch, numpy; print(torch.__version__, numpy.__version__)'`。

固定支持读出、中心光谱旁路、空间错位控制及真实四臂预检见[生产候选与准入指南](native-production-admission.md)。

## 从主开发血缘导入

在已有 `9431fece685b0a729e4ffea065fbf7c2ab5b657d` 的项目克隆中执行以下命令。从单一检出启动脚本，无需添加其他检出到 Python 导入路径。

```bash
git bundle verify /path/to/native-release.bundle
git fetch /path/to/native-release.bundle HEAD
git switch -c codex/native-coverage-review FETCH_HEAD
git status --porcelain
```

## 合成队列

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
repo_sha=$(git rev-parse HEAD)
runs=/home/lbx/hsi-pioneer271828/runs
work="$runs/native-coverage-review"
python_bin=/home/lbx/hsi-pioneer271828/envs/hsi/bin/python

"$python_bin" -B scripts/run_native_released_points.py \
  --expected-commit "$repo_sha" --runs "$runs" --out "$work" --mode coverage-register
"$python_bin" -B scripts/run_native_released_points.py \
  --expected-commit "$repo_sha" --runs "$runs" --out "$work" --mode coverage-run --workers 2
```

`coverage-register` 固定 24 个 Bank 和 120 个案例。PaviaU、徐州和洪湖是新生成的三类工程场景，原名仅用于选择头和类别数。每景包含两个支持种子、K1/K5、两种带缓冲的空间布局，以及 none、anchor、binary、continuous、shuffle 五种覆盖调制。每个组使用同一份 Bank、静态源共享权重和模型种子。模型种子按场景和分区选用 64、512、2048。合成洪湖第 21 类无释放点，分母保留为零，召回留空。

每案 12 次更新，依次遍历全带和中心缺带下的六块训练空间。固定开发验证涵盖八种波段状态，并在第 0、6、12 步评估。保存首个最低融合 NLL 模型。完成独立的 12 步和 4+8 步两条执行路径，比较完整状态，再从终点恢复零步。相同命令可续跑。已接受阶段须符合当前登记、前驱和检查点哈希，有检查点却无阶段回执的中断文件会保留在该案例目录。不同案例可由独立工作进程执行。

## 冻结兼容和 WP12

已发布真实 Bank 的只读预检继续使用 `--mode preflight`。PaviaU 需要原发布路径及 SHA，洪湖和徐州沿用已冻结的授权入口。旧 `--mode synthetic`、`--mode real-stage` 及其发布、预检和授权校验保留，新的五组队列仅接受重新生成的合成 Bank。

冻结回放使用三个已批准并固定 SHA 的带尺度历史回执，每景从首个具有预定类别的开发验证块读取全带和中心缺带两个原生视图。回放对照输入、检查点和概率的 FP32 字节以及稀疏 Q/A。WP06 七个探针只导出尺寸和哈希。冻结回放的结论范围是工程兼容，独立真实支持缺失时不得外推。

```bash
"$python_bin" -B scripts/run_native_released_points.py \
  --expected-commit "$repo_sha" --runs "$runs" --out "$work" --mode coverage-replay \
  --archive "$pavia_archive" "$pavia_sha" --archive "$xuzhou_archive" "$xuzhou_sha" \
  --archive "$honghu_archive" "$honghu_sha"
"$python_bin" -B scripts/run_native_released_points.py \
  --expected-commit "$repo_sha" --runs "$runs" --out "$work" --mode coverage-audit
```

各 `*_archive` 变量指向已授权 `runs` 中的回执，精确 SHA 见 `EXPECTED_ARCHIVES`。WP12 验算逐类分母、混淆计数、选择目标、阶段更新数和恢复状态。产物含真实兼容和合成逐类表、配对差异、阶段成本及设备回执。训练成本包含该步触发的验证，转化、前向和构图是重叠计时项，不应累加。

## 设备与缓存

CUDA 资格短测为注册合成案例。每次指定一个物理 UUID，至少空闲 12,288 MiB 且利用率至多 60%，按同一物理锁核准。不同 GPU 可分别执行不同案例。只读跨设备装载使用固定检查点中的已选模型，优化器恢复继续受登记设备约束。

```bash
export CUDA_VISIBLE_DEVICES="$gpu_uuid" CUBLAS_WORKSPACE_CONFIG=:4096:8
"$python_bin" -B scripts/run_native_released_points.py \
  --expected-commit "$repo_sha" --runs "$runs" --out "$work" --mode coverage-run \
  --case paviaU-s512-k1-a-binary --gpu "$gpu_uuid"
"$python_bin" -B scripts/run_native_released_points.py \
  --expected-commit "$repo_sha" --runs "$runs" --out "$work" --mode coverage-reload \
  --case paviaU-s512-k1-a-binary --from-device cpu --gpu "$gpu_uuid"
```

可反向装载到 CPU，也可指定另一张符合资源门槛的物理卡。冻结图的同视图缓存要求模型状态、设备、空间、角色、波段与构图特征一致。训练中的原生图随特征构建。既有实验性跨视图固定图介入仍需自己的合法性验证。

候选保留主线模型默认值，集成 Bank 发布与读取、WP05 原带与统计切片、覆盖五组、原生构图和设备恢复、WP06 冻结探针及 WP12 导出。历史候选假设、其余整条实验分支和 WP04 未并入这次运行。真实训练、原始真值和 query 保持关闭。
