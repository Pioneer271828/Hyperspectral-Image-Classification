# G4 轻量适配同协议只读评分工程验收

对应研究主清单 P4 §10.3、实施计划 §19、Issue249 预登记。评分入口先绑定登记、完整训练验收、54份独立审计与冻结清单，随后校验固定的 Issue184 冻结及评分回执。沿用 Issue184 的标签释放与有序前向，每景以原18个结果的释放哈希验证 query 像素顺序、标签和类别身份。轻量两方法以各自选中权重、侧车和原始验证状态重载，对同一像素队列评分。记录三头指标、分支诊断和预登记配对判定。读取前核验完整输入，评分中检查选中模型的参数和缓冲区身份，评分后核对全部保护文件。逐例结果和成功回执只写新同级目录。

实测分支 `codex/g4-light-ordered-query`，完整 SHA `4be3db3db6b8de5983a761b33c9c5e46d98ae881`。授权 LXC 独立检出 `/home/lbx/hsi-pioneer271828/checkouts/issue249-ordered-query02`，Python 3.10.20、PyTorch 2.7.1+cu128。CPU 专项6/6、81.175秒，相关回归19/19（冻结门禁6项0.696秒、配对判定5项0.846秒、Issue184 审计与查询边界8项33.676秒）。CLI 帮助及仓库检查419 Python、131 JSON通过。

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_repository.py --expected-commit 4be3db3db6b8de5983a761b33c9c5e46d98ae881
python -m g4_light_ordered_query --help
for suite in test_g4_light_ordered_query test_g4_light_frozen_boundary test_g4_light_paired_analysis test_g4_matched_target_audit; do
  python -m unittest discover -s tests -p "$suite.py" -v
done
```

两种方法各用一个合成案例完成1500步，随后用实际选中态和有序前向做三头评分，确认训练目录的全部文件哈希一致。合成全54清单覆盖运行中拒绝、缺失角色、选中步篡改及像素顺序变动。独立读取了既有 Issue184 回执和冻结门禁的实时边界文件，54行原结果身份核对通过，原始 GT 读取0。这个检查使用实时边界文件，完整保护清单重哈希留给实际评分入口。

证据根 `/home/lbx/hsi-pioneer271828/runs`。专项及回归日志 `issue249-ordered-query02.log` SHA256 `6051875f0e229afe3e67e6be79820266da998d13968f4471683c898f1e2d38c7`。原参考回执核验日志 `issue249-ordered-query-reference02.log` SHA256 `ab1cff164d55f72bb7d82b3af3b99d32ac116149d6c9f1cd405cdfc732ce6f4f`。

评分入口工程验收通过。GPU 评分专项、全仓回归和 Issue249 全54真实评分未执行。下一步完成训练/验证复盘及成本导出，真实批次完成、独立接受、复盘并冻结后再执行评分。本聊天2026-11-07交付评分工程进度。
