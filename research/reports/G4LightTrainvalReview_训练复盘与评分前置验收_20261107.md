# G4 轻量适配训练复盘与评分前置验收

对应研究主清单 P4 §10.3、实施计划 §19、Issue249。新增只读训练复盘。复用已认证的无标签冻结清单与逐例审计，重放验证选步和学习率，核对51个事件三头混淆及逐类召回，记录末段验证变化、可训练参数更新和分阶段记录成本。输出全54病例的曲线 CSV、PNG、逐例明细和18组的三个种子汇总。评分入口新增冻结、验收和复盘回执绑定，逐一核对108个导出文件后才可读取 query。复盘保留事件计时排除项、续跑可能重复的工作，以及控制器实耗和分阶段计时的不同口径。

实测分支 `codex/g4-light-trainval-review`，完整 SHA `44696a60d68baffff5b636d2e107976b531abc3c`。授权 LXC `/home/lbx/hsi-pioneer271828/checkouts/issue249-trainval-review01`，Python 3.10.20、PyTorch 2.7.1+cu128。CPU 复盘专项2/2（0.032秒），评分6/6、冻结6/6、判定5/5共17项回归通过。两个入口帮助及仓库检查421 Python、131 JSON通过。

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_repository.py --expected-commit 44696a60d68baffff5b636d2e107976b531abc3c
python -m g4_light_trainval_review --help
python -m g4_light_ordered_query --help
for suite in test_g4_light_trainval_review test_g4_light_ordered_query test_g4_light_frozen_boundary test_g4_light_paired_analysis; do
  python -m unittest discover -s tests -p "$suite.py" -v
done
```

用实际已审计的 HongHu64 容量仅头、Xuzhou512 容量 VCA 加头两个案例验证复盘字段、输入哈希与导出。分别选中1500、391，均导出51个事件、PNG 和 CSV。HongHu 图已目检。证据根 `/home/lbx/hsi-pioneer271828/runs`，专项日志 `issue249-trainval-review01.log` SHA256 `09277603899b5228dbfa0b9df4f5ea85370a979ab711741a02d49596cbcc51f2`，两例复盘日志 `issue249-trainval-review-partial01.log` SHA256 `ba51c81b1e1f24f420f1a23456ca2293b90ff08bf076f26bb4fbdb1a71d17e8a`，示例输出 `issue249-trainval-review-partial01/`。两例导出属于工程接入验证。未发布全54真实复盘回执，未做 GPU 新专项、全仓回归或真实 query。

## Issue249 现场异常

2026-11-07 18:55左右（北京时间）观察到控制器停止派发。已验收35/54，GPU1 的在途 Xuzhou 仅头任务继续收尾。GPU5 的 `whu-honghu-s64-none-source_pretrained_adapter_heads` 在首步 VCA 值投影请求540 MiB时 OOM。日志显示 PyTorch 已分配约21.38 GiB、GPU 仅余70.31 MiB，另有约1.53 GiB的共存进程占用。失败例只有初始化记录，缺少持久训练游标。已核实 `vca` 暂停。保留原登记、源码和产物。

19:07复查时控制器已退出，状态为 failed，验收数仍为35/54。GPU1 在途案例的训练日志显示第1500步及完成，失败排空后该例尚未通过独立验收。没有全批次接受回执。

失败证据 `g4-light-full-issue249-registered01/logs/whu-honghu-s64-none-source_pretrained_adapter_heads-first2-zpzlzcio.log` SHA256 `a73da54c51a430085c36a236e9bb1c5be6de467e9c6075bd123a60b9406e3349`。日志显示进入 VCA 分块值投影。训练的骨干保持 eval，现有值投影分块仅在训练态启用重计算。共存进程与算子存储峰值均需纳入诊断。下一模块独立验证显存路径的数值与权限一致性及实际容量余量，再登记恢复边界。真实复盘和冻结评分继续等待全部训练验收。本聊天2026-11-07交付工程及异常进度。
