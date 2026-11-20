# WP11／WP12 冻结路径部署成本

在授权 LXC 从同一份干净检出运行。完整提交 SHA 与输出均由正常入口校验。旧的 `register` 和 `run` 子命令仍消费原 WP11 冻结清单，默认完整 FP32 路径保持原样。

## 预登记和执行

```bash
repo_sha=$(git rev-parse HEAD)
python_bin=/home/lbx/hsi-pioneer271828/envs/hsi/bin/python
runs=/home/lbx/hsi-pioneer271828/runs
out="$runs/wp11-deployment-review"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
"$python_bin" -B scripts/run_wp11_execution.py --expected-commit "$repo_sha" \
  deployment-register --runs "$runs" --out "$out"
plan_sha=$(sha256sum "$out/measurement-plan.json" | cut -d' ' -f1)
"$python_bin" -B scripts/run_wp11_execution.py --expected-commit "$repo_sha" \
  deployment-measure --runs "$runs" --out "$out" --plan-sha "$plan_sha"
"$python_bin" -B scripts/run_wp11_execution.py --expected-commit "$repo_sha" \
  deployment-synthetic --runs "$runs" --out "$out.synthetic"
"$python_bin" -B scripts/run_wp11_execution.py --expected-commit "$repo_sha" \
  deployment-audit --runs "$runs" --out "$out" --plan-sha "$plan_sha" --synthetic "$out.synthetic"
```

登记固定三景已授权原生开发 Bank 的中位面积块，每景全带和中心缺带两个请求。另封存徐州已有匹配轻/重模型的相同五个无标签模型输入，沿用历史比例、原始波形和开发角色。选择依据只用块几何和已发布资格。真实路径均为固定只读推理。Issue 249 的残差头与上面这些轻重模型来自不同合同，保留只读引用。历史 54 例纯常驻装载/计时仍作为旧统计口径单列。

| 范围 | 所计工作 |
|---|---|
| 冷启动到首个可用预测 | 全新进程、导入、入站权限核查、模型载入、输入 I/O、必要预处理/构图、同步前向、CPU 预测物化。另记进程退出耗时 |
| 暖态端到端 | 模型驻留，重新读取登记输入、共享预处理、设备传输、实际路径和预测物化 |
| 纯常驻前向 | 模型和输入驻留，包含模型内计算与原生图构建。移出输入 I/O 和输出校验 |
| 并发吞吐 | 两个独立进程均准备完成后同时放行。以到最后预测物化的壁钟计算吞吐，另保留逐请求时延 |

“冷”指新进程。OS 页缓存不人为清理和控制。分位数由原始重复样本计算。每条数据保留批量、总像素、同步、预热、模型/图调用和各自设备。并发的 RSS/显存峰值按单工作进程取最大。CUDA 另用事件记录同步 GPU 工作，和进程端到端计时分别汇报。轻路完成时图构建为零，完整路按实际请求构图。图缓存只用于经验证模型、完全相同原生请求与生成特征的冻结测试，不作为真实计时默认路径。

CUDA 用同一物理 UUID 映射和资源锁。样本准入需至少 12,288 MiB 空闲且利用率至多 60%，确定 UUID 后可给 `deployment-measure` 和 `deployment-synthetic` 加 `--gpu "$gpu_uuid"`。设置 `CUDA_VISIBLE_DEVICES` 为该 UUID。CUDA 前向使用已登记的确定性稀疏计算协议。新进程继承同一租约和算术合同。设备各自发布，切换物理卡产生独立证据。若阶段失败，保留 attempt 日志并再次运行同一计划。已接受的阶段复用，未发布的阶段重新测量。并发采集到预定的阶段锁后，禁止覆盖已有记录。

## 校准和确认边界

合成模式复用独立 Bank 的审批接口，以校准角色拟合固定 margin 网格，写入冻结记录后才从独立父块读取确认请求。两组覆盖与回退预算共享相同输入、模型种子和支持种子。达到预算以外仍安全回退并记超额。匹配随机控制在波段、块大小和覆盖分层内匹配实际轻、重调用次数。确认只消费冻结阈值。独立真实校准与确认尚待数据授权方发布，真实动态收益表保持空缺。

WP12 发布包含分设备的真实四范围计时、旧版引用和单独的合成路径质量。探针、温度介入和可变光谱缓存不进入部署计时。源码实例中使用现有常驻语义，没有开真实训练、原始真值或 query。

### WP10 冻结压缩生产暖态

同一授权 LXC 的干净检出，沿用上文 CPU 变量。供给已发布的 WP10 低秩清单和旧独立 CPU 成本回执的绝对路径及 SHA。候选为原三恢复种子各自的 dense 与 r16，端到端暖态每次重新读取已发布的观测包。

~~~bash
entry=scripts/run_frozen_cpu_deployment.py
"$python_bin" "$entry" --expected-commit "$repo_sha" warm-register \
  --input "$manifest" "$manifest_sha" --reference "$cpu_receipt" "$cpu_sha" --out "$runs/wp10-warm-plan"
plan="$runs/wp10-warm-plan/plan.json"
plan_sha=$(sha256sum "$plan" | cut -d' ' -f1)
taskset -c 0 "$python_bin" "$entry" --expected-commit "$repo_sha" warm-cost \
  --input "$plan" "$plan_sha" --out "$runs/wp10-warm"
~~~

WP12 的 `scripts/ingest_wp12_evidence.py --wp10-deployment-review` 接收 `--expected-commit`、新仓外 `--out`，以及 `--warm`、`--cold`、`--cuda`、`--cuda-audit`、`--onednn-quality`、`--onednn-cost`、`--wp11-cpu`、`--wp11-cuda` 各自的绝对路径与 SHA。实际暖态工作提交、旧清单生产提交、冷启动或历史 CUDA 的不同计时范围均在结果中保留。[实际数据与护栏](../../research/reports/WP10_WP12_冻结低秩暖态生产与配对成本_20261120.md)
