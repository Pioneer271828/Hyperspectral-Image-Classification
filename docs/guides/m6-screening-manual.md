# PaviaU B0 / Add / Gated：手动运行第一轮完整预算初筛

日期2026-10-30，实验准备 Issue #23。长训练由用户手动启动，本任务仅准备、测试与30步测速。不启动 M7。

## 预先固定的比较

三组配置为 `configs/experiments/screen_m6_{none,add,gated}.json`。每组 `(99+1)*100=10000` 实际优化更新，每100步验证一次，共100次，按验证前景NLL选最佳checkpoint，最终test一次。固定原lr/head_lr、损失、graph_update=none、group_norm、K64/embed32、tile32/gap2、split/model_seed2026。采样仍为有放回抽取有标注训练块，100步不是整景遍历一遍。三个组顺序运行，不并发占 GPU。

这是预先固定的完整初筛预算，不保证10000步已收敛。先依据验证曲线判断是否需要另立延长预算；三组必须一起调整，不能依据test精度挑预算或模型。测试缺类3/7，AA仅支持类，边界N/A；正式结论还需至少三个配对重复及其他数据集。

相同种子本身不能保证不同结构的共有参数相同。脚本从无交互模型生成共同state，逐键加载、哈希校验，共有参数与缓冲一致；仅交互模块新增参数由各组构造初始化，然后重置训练RNG。训练块抽样使用相同独立NumPy种子。共同state、campaign.json、配置哈希和完整代码SHA写到checkout外，变化时拒绝复用。

## 容器命令

先在 Windows 终端执行 `ssh rmvision`，进入训练容器。此实验使用专属 `/workspace/hsi-m6-screening`，不要更新正在运行的其他checkout。实际发布SHA及本次预检命令见对应实验准备报告。

新建专属tmux会话：

```bash
tmux new -s hsi-m6-screen
cd /workspace/hsi-m6-screening
PY=/home/rmvision/venvs/hsi-baseline/bin/python
SHA=$(git rev-parse HEAD)
# 启动前将SHA和交付报告核对；输出目录必须使用交付的专属路径。
OMP_NUM_THREADS=1 "$PY" -u scripts/run_m6_screening.py run \
  --expected-commit "$SHA" \
  --data-root /home/train/hsi-pioneer271828/datasets/current \
  --out-root /home/train/hsi-pioneer271828/runs/m6-screening-20261030
```

命令本身不依赖终端持续连接；按Ctrl+B再按D退出tmux，重连用 `tmux attach -t hsi-m6-screen`。不要使用其他项目的 `train` 会话。

建议先在正常shell执行 `prepare` 或 `benchmark`（替换上方 action）：prepare只创建并验证共有初值/配置身份，不读数据训练；benchmark仅none组30步并做原训练入口的验证/测试，结果只用于计时，不参与正式精度排名，产物在benchmark子目录。启动脚本拒绝错误SHA、脏工作树、无CUDA与并发campaign。

## 进度与结果

`<out-root>/progress.json` 是当前组最近一次进度，`progress.jsonl` 跨组按时间追加；每10步与每次验证更新。每条包含variant、实际更新步数、loss或val指标、UTC时间与耗时。所有epoch的详细进度在各attempt的training.log；run脚本每组完成打印该组状态和用时。`{none,add,gated}-result.json` 记录组结果位置。

```bash
cat /home/train/hsi-pioneer271828/runs/m6-screening-20261030/progress.json
tail -n 5 /home/train/hsi-pioneer271828/runs/m6-screening-20261030/progress.jsonl
```

每组权重、model.pt.json、protocol/statistics、metrics.json与manifest在 `<out-root>/training/<run_id>/attempts/0001/`。完整metrics在完成后写入；训练期间主要看progress日志与已有验证checkpoint。进度总数10000表示当前组；三组合计30000步。

## 中断与重试的准确含义

SSH断开但tmux仍在时只需重连，不另起训练。正常失败或Ctrl+C后，同一run命令加 `--resume`，验证完成组后跳过；失败组创建新attempt，从共同初值重新开始，该组之前的更新不续接。当前没有保存optimizer/scheduler/RNG中间状态，不能称断点续训。

每个campaign及C0运行有独占锁。若进程被SIGKILL或机器重启，锁可能残留；先确认原PID与tmux任务已停止，再检查锁内容及对应attempt，不能直接删除活跃锁、reset或clean。不要在训练期间改代码、配置或共同初值。未完成训练不能根据launcher退出或GPU有占用判作完成。
