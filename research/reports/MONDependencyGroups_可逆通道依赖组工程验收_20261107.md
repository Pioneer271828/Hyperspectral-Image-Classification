# MON 通道依赖组可逆干预工程验收

对应全量必测清单 MON/MASK、实施计划 §19 和统一训练监测计划 MON-3/4。新增 `monitoring/dependency_groups.py`，消费持续窗口筛出的训练侧候选，将候选锚点和显式依赖位置绑定到同一冻结验证探针。按预登记顺序执行原模型、每个单组和指定组合。每次调用既有隔离探针，用独立模型副本在指定输出处置零。输出三头 OA、AA、混淆及逐类指标。融合头按原始计数比较 OA、AA、逐类召回下降和小类新增错误，组合同时需要其单组通过。

实测 SHA `ee2c3d54283c9c26746adf8360f776709773a79d`，分支 `codex/mon-dependency-interventions`，授权 LXC `/home/lbx/hsi-pioneer271828/checkouts/mon-dependency-groups01`。Python 3.10.20、PyTorch 2.7.1+cu128。CPU 监测40/40（新增3项）7.951秒，仓库检查324 Python、128 JSON通过。

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_repository.py --expected-commit ee2c3d54283c9c26746adf8360f776709773a79d
python -m unittest discover -s tests -p 'test_monitoring*.py' -v
```

合成双分支示例中两个单组均通过，组合屏蔽导致类1新增一个错误，保护规则拒绝组合。冻结计划改动、锚点缺失、未筛选通道、检查点或验证标签身份不一致均被拒绝。日志 `/home/lbx/hsi-pioneer271828/runs/mon-dependency-groups01.log` SHA256 `0cbb88b782e10e18b9ec0f78569780219363fd1226e9cb8f88b68717b69992e0`。示例 `/home/lbx/hsi-pioneer271828/runs/mon-dependency-groups-example01/receipt.json` SHA256 `e7611173f9f1e8fe4adf973e10f9aa1506ca3faa8f06bce4512d5a34077393c6`。

工程接入通过。依赖位置由调用方显式声明，真实 G4 拓扑闭包、候选真实干预、GPU 专项及全仓回归待验。无实际结构裁剪或推理速度结论。本聊天交付工程进度。
