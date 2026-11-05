# G4完整54目标配对训练记录

本项对应研究主清单P4 §10.3、模块计划§19，Issue184 / Draft PR185，依赖已完成工程182。预登记见[G4完整54目标配对预登记](../experiments/G4MatchedFullTarget_完整54目标配对预登记_20261106.md)，静态运行身份见[登记摘要](../../docs/architecture/g4-matched-target-registration-results.json)。当前仅队列启动和首步前冻结，不是完整实验完成或精度结论；各case完成只算批次进度。

生产/实际测试SHA `331d718567b1a9726934f445665999df972392b1`，独立LXC `checkouts/g4-matched-target-issue184-production01`、分支`codex/g4-matched-target-issue184-production01`干净检查通过357Python/81JSON。解释器`/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，Python3.10.20/Torch2.7.1+cu128/CUDA12.8，GPU1 RTX4090。工程完整CPU/GPU/回归、CLI合成1500和两环境结果按[182工程报告](G4MatchedFullTarget_完整协议工程阶段记录_20261106.md)复用；本项不将文档SHA写成重新测试生产代码的SHA，CI未注册。

独立预登记文件版本`b2912ce2d22f40d52f1cd353c16400a4f32c3c4b`已推送、rmvision干净checker357Python82JSON通过。主要残差微调−容量微调，独立次要残差微调−none微调及三个模式的预训练收益；配对宏OA>0且3/3种子正、各目标AA下降≤1pp、HongHuOA下降≤1pp、每类均值召回下降≤2pp。全部正负和三头/逐类保留，不按query选配方。

固定54×1500=81000更新：none27000、容量/残差活动54000；每例2+1498，完成恢复0。支持种子64，模型种子64/512/2048，K10训练另V5验证，新目标头/DomainBN、统一全图无标签观察/图、原损失/调度与首次最小验证NLL选模不变。训练/审计query0；全54接受、独立审计、拟合/小类/成本复盘并创建全部模型冻结边界后才评价query。

## 启动与冻结身份

2026-11-06T03:15:09Z启动队列。外部registration为`runs/g4-matched-target-issue184-registration02/registration.json`，hash `0957aa3c5177be4faa38d6c68576a1db8e12679ab0d3de0bca67a19db2245017`；analysis-plan hash `0bd71779546fd5ee0cbaab57b7b3639a1c378bcf9d1379f8afb1a80231a92dee`，工程receipt hash `80a18ffb42d9b7de0c4b1acc20c48e4e37ce04ef398e0c16137c16e5fb766715`，原54零更新preflight hash `2dacef05240b2d41854c4e70f97e7e3887a3b64bb2e1c3872ffead9bb97d82a5`。运行`runs/g4-matched-target-issue184-full01`、日志同名`.log`；wrapper PID273455，实际Python supervisor须读实时进程，不能把启动shell当训练进程。启动前GPU1 free21748MiB/util41%，磁盘318GiB可用。

实际命令（在上述干净331 checkout，单线程、无bytecode）：

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:tests python -u research/reproduction/m4g4b/matched_target_campaign.py \
 --expected-commit 331d718567b1a9726934f445665999df972392b1 \
 --registration /home/lbx/hsi-pioneer271828/runs/g4-matched-target-issue184-registration02/registration.json \
 --registration-sha256 0957aa3c5177be4faa38d6c68576a1db8e12679ab0d3de0bca67a19db2245017
```

每个子操作free>=12288MiB/util<=60%才执行，不满足等待，不动其他项目。首步前重新核54源/目标身份与9组共同fresh，生成新的executable launch和保护清单，原preflight及源/短测不覆盖。先估12–24h加资源等待，registration冻结estimated_seconds86400，按实测吞吐修正预计时间，不改81000预算。启动/退出总墙钟在父日志记录，controller elapsed不含启动freeze，两者分开。失败保留并停下诊断，不放宽或重跑择优。

## 保留的失败与未完成

registration01抄录工程receipt哈希多写一位，在任何训练/registration写入前被断言拦截；只留下旧analysis-plan副本，保持原样。随后从已提交工程摘要读取真实hash，修正文档并独立registration02；真实工程receipt、源权重、生产代码和数值门槛未改。该问题不计作模型数值失败，也不隐藏。

实际54完整训练、324头指标/4320召回/162独立精确重载、完成0恢复、独立全批次零更新audit、曲线/逐类复盘、全冻结query与效果判据尚待完成。非选中raw事件核hash/history/选步/LR，不扩称全轨迹独立概率复算。负收益也可完成实验，未测不得删范围。本项仅固定支持集上的少样本传导式适配，不称纯源TTA/零样本/未见传感器泛化；其他支持种子/K/空间/传感器及head-only/light-module继续保留。main未合并。
