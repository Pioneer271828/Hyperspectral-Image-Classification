# MON-4 结构通道新旧映射工程验收

对应全量必测清单 MON-4／结构剪枝及实施计划 §19。新增 `compression/channel_mapping.py`。消费固定可逆干预计划与已通过的**同一**单组或组合，核对参考权重身份和独立模板。按明确绑定到依赖位置的张量轴切片，记录旧到新、新到旧及被删除索引。所有张量名、dtype、device 和预期形状一致才发布紧凑状态，验证不全、别名、零宽、失败组合与漏改下游均拒绝。输出不包含优化器状态。

实测 SHA `2128a1cb4f774e987654132b1b0e9e6dc208da67`，分支 `codex/mon-compact-channel-map`，授权 LXC `/home/lbx/hsi-pioneer271828/checkouts/mon-compact-channel-map01`。Python 3.10.20、PyTorch 2.7.1+cu128。CPU 新专项3/3（0.064秒）、监测40/40（7.862秒）、掩码6/6（0.996秒）。仓库检查326 Python、128 JSON通过。

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_repository.py --expected-commit 2128a1cb4f774e987654132b1b0e9e6dc208da67
python -m unittest discover -s tests -p 'test_compact_channel_mapping.py' -v
python -m unittest discover -s tests -p 'test_monitoring*.py' -v
python -m unittest discover -s tests -p 'test_weight_masks.py' -v
```

合成投影、BN、三头分流模型真实从3通道缩为2通道，参数39降至28。紧凑模型严格加载并实际前向，三个头相对可逆屏蔽模型最大绝对差均为0。参考模型与模板在状态构造期间保持哈希一致。专项验证了遗漏下游轴和用两个通过的单组替代失败组合均被拒绝。

日志 `/home/lbx/hsi-pioneer271828/runs/mon-compact-channel-map01.log` SHA256 `f28169da90efa84f1b6c06ab5b8037fb4fb3bfa71d157b9d7c2323bfd6004d63`。示例 `/home/lbx/hsi-pioneer271828/runs/mon-compact-channel-map-example01/receipt.json` SHA256 `66be8590208b94769efc363e785a89275aebb8daa01a70b0f043622d4ae26289`。

紧凑张量映射工程通过。依赖位置和切片轴由调用方明确声明，完整状态形状约束不证明真实 G4 图的闭包。真实结构剪枝、恢复训练、精度、GPU 开销与全仓回归仍待验。下一步选定一个真实可裁剪层并核完依赖闭包，再接冻结前向复验和等预算恢复。本聊天交付工程进度。
