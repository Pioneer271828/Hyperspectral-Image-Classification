# WP12 冻结54成本观测准备与运行时准入

对应 Issue365、主计划 §20/§22、实施计划 §19.13，依赖 Issue363/364。分支 `codex/wp12-cost-preparation`，实际测试提交 `e9a091b27498c386323643d97df57ba2c62e882a`。该模块准备真实成本测量的输入和案例绑定，不训练、不计时、不解码标签家族或原始query标签，不读取权重张量、不进行模型前向。

## 实现

新增 `evaluation/wp12_cost_preparation.py`、`scripts/prepare_wp12_cost.py` 和专项测试。入口接受固定SHA的精度分析、分析回执、全54冻结边界及原launch，复用原精度/边界核对。逐例核对案例摘要、检查点/sidecar、场景与评估身份；同一场景的所有种子/模式必须绑定同一观测记录。只解码原有无标签观测文件，完整核对原字段指纹、几何和元数据，拒绝额外标签字段，导出六个允许观测字段并重新加载验证完全一致。

导出清单保留54例和三场景工作量、原始文件/新文件/张量三个层面的身份。输出为新同级目录，不覆盖模型、输入和旧产物；任何最后检查失败均不会写成功清单，已经产生的失败导出文件保留。

运行时准入核对原生产清单的全部既有Python/config文件，允许只添加新文件。发现旧文件变化则明确 `compatibility_pending`，列出路径及原/当前SHA。源码变化本身不是数值错误证据；该模块既不认定不兼容，也不自动放宽。即使文件一致，仍不自行登记硬件预算或生成执行作业。

## 验收

环境为授权 LXC `lbx@10.69.37.243:1222`，Python `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，干净checkout `checkouts/wp12-cost-preparation01`，Python3.10.20/PyTorch2.7.1+cu128。全部CPU执行，GPU和真实训练更新0。

```sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests PYTHONDONTWRITEBYTECODE=1
python scripts/check_repository.py --expected-commit e9a091b27498c386323643d97df57ba2c62e882a
python -m unittest tests.test_wp12_cost_preparation tests.test_wp12_evidence tests.test_unified_cohort tests.test_repository_delivery -v
python scripts/prepare_wp12_cost.py --expected-commit e9a091b27498c386323643d97df57ba2c62e882a \
  --registration <runs/wp12-cost-preparation-engineering01/real-registration.json的绝对路径> \
  --registration-sha256 <该登记文件实际SHA256> \
  --out /home/lbx/hsi-pioneer271828/runs/wp12-cost-frozen54-preparation01
```

22/22 CPU测试通过，19.606秒，无跳过。包括全54案例绑定、三观测序列化、禁止前向、错案例/检查点/观测SHA、带标签文件、重复输出、收尾失败不写成功清单，以及新增源码允许/旧源码变更保留/脏代码拒绝。仓库检查585 Python/198 JSON通过。

真实CLI退出0，17.797秒完成已有54案例的只读绑定，导出3个观测文件：PaviaU207400像素、Xuzhou130000像素、HongHu446500像素。总输出976381342字节，数据均在checkout外。原始观测与记录共6个文件再次核SHA保持不变，新导出3文件SHA重新核对通过。没有新精度/成本/逐类结果。

实际清单：`/home/lbx/hsi-pioneer271828/runs/wp12-cost-frozen54-preparation01/preparation.json`，SHA256 `c32ff8865bfce91c40a3c19305e9d3f5c454568f6556a9b93cbb12a4ffa86ee4`。工程日志和真实CLI登记/日志位于 `runs/wp12-cost-preparation-engineering01/`。

## 真实准入结论与剩余事项

输入准备验收通过，**运行时等价性仍待验**。实际发现8个原有文件不同：`model/VCASAHGCN.py`、`repository_delivery.py`、三个测试文件，以及 `transfer/development_training.py`、`transfer/m4_ablation.py`、`transfer/observation_policy.py`。模型文件差异主要是后续增加的默认关闭图路由；仅凭默认关闭不能把当前版本计时自动配给原始331精度。清单不生成可执行成本作业，不读取query进行追调。

下一依赖是固定历史推理运行时的隔离采集，或完成明确登记的等价性核验；之后在资源门槛满足时补GPU实际计时和全54成本。Issue364的CPU采集工程已经交付，PR366以 `codex/wp12-evidence-ingest` 为基线，不代表main已整合。WP04真实短测继续按用户指令后补。当前WP12整包仍未完成真实统一精度/成本验收，不能标为最终模型交付。
