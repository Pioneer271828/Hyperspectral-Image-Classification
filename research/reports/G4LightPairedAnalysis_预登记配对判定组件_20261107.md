# G4 轻量适配预登记配对判定组件

对应 P4 §10.3、实施计划 §19 和 Issue249 的固定分析计划 `c666ca8b0c1d78471b662ef900e0c6302fb8d7de3cbd2761eed42bf6da7c1d00`。新增 `g4_light_paired_analysis.py`，分别判定仅三头与 VCA 加三头对同角色全量微调的两个主比较。保留两个对从头训练的比较与 VCA 加头对仅头的描述性读数。

组件校验完整的两份54角色指标与三头类别支持，以混淆计数的有理数重算阈值，按结构、场景等权和模型种子配对。逐景、逐结构保留三头指标差值和召回差值。合成例确认三个种子总体 OA 都提高时，单个类下降超过2个百分点仍使对应主比较失败；恰好下降2个百分点通过该保护。两项主比较独立判定。

授权 LXC `/home/lbx/hsi-pioneer271828/checkouts/issue249-paired-analysis01`，分支 `codex/g4-light-query-analysis`，实测提交 `311da2f67e5c965b6b6f0c6ac9ef07e1306910c6`。CPU 新专项5/5（0.851秒），Issue184 配对分析回归11/11（16.079秒），轻量冻结门禁回归6/6（0.788秒）。仓库检查417 Python、131 JSON通过。

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_repository.py --expected-commit 311da2f67e5c965b6b6f0c6ac9ef07e1306910c6
python -m unittest discover -s tests -p test_g4_light_paired_analysis.py -v
python -m unittest discover -s tests -p test_g4_matched_target_analysis.py -v
python -m unittest discover -s tests -p test_g4_light_frozen_boundary.py -v
```

证据 `/home/lbx/hsi-pioneer271828/runs/issue249-paired-analysis01.log` SHA256 `1863f88162b80fcca11cef584ad729d4180195b6b73a421209482d9d1c54f9cb`。本组件输入是上游认证后的已评分行。相同类支持的检查只覆盖计数一致性，像素队列身份和选中权重认证由冻结评分入口承担。CPU 组件和邻近回归通过，GPU、全仓回归、真实结果和成本统计未验收。后续接入同协议只读评分、真实训练复盘与全54评估。本聊天2026-11-07报告组件进度。
