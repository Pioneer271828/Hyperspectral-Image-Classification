# G4完整预算轻量适配入口验收

Issue245 / PR246。对应研究主清单P4 §10.3和实施计划§19，承接Issue192短程权限工程和Issue243的真实输入、初态一致性证明。接入仅训练三头与VCA加三头两种完整源控制，保持Issue184的K10/V5条件、模型种子、初态、训练配置、更新预算与验证节奏。两种适配方法自身的BN行为按原Issue192分别保留。

## 实现

公开入口为 `scripts/train_g4_light_target_full.py`，每次执行一个登记角色。输入登记固定代码提交、Issue184参考计划、工程验收、分析计划、完整54×1500预算和2～3个不同物理GPU。分配由规范角色顺序和登记卡列表确定。worker核对可见GPU与分配一致、空闲显存至少12288MiB且利用率不高于60%，登记拥有独立输出目录，同一登记内的同卡并发由文件锁拒绝。训练继续复用原有持久状态、选模、守卫和只读完成检查。

短程入口保持12步合同，明确拒绝完整预算。完整入口实际消费原184参考角色的训练和验证数据，重用其配置并登记本方法的参数权限。每例1500更新、51次验证，首次最小融合验证NLL选模。仅头组使用冻结特征库，VCA加头组只允许适配器及三头训练。训练阶段其他参数、梯度及BN缓冲受核验。

## 实际验收

授权LXC `/home/lbx/hsi-pioneer271828/checkouts/issue245-worker02`，分支 `codex/g4-light-full-target`，测试提交 `228632c3dfd043569dcf7e35e79c3c5adaa3bfe4`。沿用Python 3.10.20和Torch 2.7.1+cu128。完整GPU专项使用物理GPU1，启动门槛检查为21748MiB空闲、45%利用率。公开入口的合成2步和锁冲突验证在物理GPU0执行。

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/lbx/hsi-pioneer271828/envs/hsi/bin/python
$PY scripts/check_repository.py --expected-commit 228632c3dfd043569dcf7e35e79c3c5adaa3bfe4
$PY scripts/train_g4_light_target_full.py --help
CUDA_VISIBLE_DEVICES= LIGHT_FULL_TEST_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu \
  $PY -m unittest discover -s tests -p 'test_g4_light*target.py' -v
CUDA_VISIBLE_DEVICES=1 LIGHT_FULL_TEST_DEVICE=cuda:0 LIGHT_TARGET_DEVICE=cuda:0 \
  $PY -m unittest discover -s tests -p 'test_g4_light*target.py' -v
```

| 验收项 | 结果 |
|---|---|
| CPU专项 | 7/7，142.722秒 |
| GPU专项 | 7/7，265.315秒 |
| 仓库检查 | 410个Python文件、129个JSON文件通过 |
| 完整合成训练 | 两种方法在CPU和GPU均完成2＋1498，完成后恢复0更新，逐例51次验证、首次最低NLL、冻结权限、选中预测精确重载均通过 |
| 旧12步回归 | 既有六角色、2＋10恢复、完成后不变及损坏特征库拒绝在CPU/GPU通过 |
| 公开入口补验 | 合成登记下的同卡锁冲突拒绝，释放锁后实际返回interrupted与2更新 |
| 真实零更新预检 | 三景54角色，98个保护文件，99.699秒，配置与初态一致，模型前向、优化和新增query均为0 |

完整预算的合成网络覆盖none结构的两种方法。六角色短程回归覆盖其他结构。真实预检覆盖三种结构的54角色，仍属于非执行预检。未重跑全仓模型回归。首轮 `63f3b6bc91de28809cd24351e2163b34ce606908` 的短入口拒绝负例触发字段缺失异常，增加显式预算边界后在228632c复跑通过。原失败日志保留。

## 归档与后续

工程验收回执 `runs/g4-light-full-issue245-engineering-acceptance02.json`，SHA256 `57720f99eb3295e8d585b91d1960efe6239785467755fd8ddaf5a288c86e9f9b`。真实零更新回执 `runs/g4-light-full-issue245-zero-preflight02.json`，SHA256 `2b04933aeeea6c262b0619d5538ccc1c644dc48a44797ffc20051915a512d118`。日志和回执副本位于 `F:/E2E_VCA_worktrees/issue245-export02`，归档SHA256 `9048e41153135fac979c5688df118832b08306568955061df7c2a9ea5ea5069d`。

Issue245验收范围为登记、单例执行、恢复和权限。Issue247继续2～3卡队列与独立完整性验收，随后登记真实对照的预算、估时和指标。真实长训练启动后交定时任务值守，普通开发和测试继续。轻量适配效果、独立支持种子、K1/5和M8对照待后续实验。
