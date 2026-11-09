# WP06：冻结模型的匹配删边复盘

Issue349 / PR350；主计划§16/§22、实施计划§19.7。分支`codex/wp06-matched-edge-review`，实际tested SHA `bfcd14c04662bf57a1ce1388fb4185a9aa1fe42c`；基准main `e7115cebd7ad70af3cf5110cd6e32a5041b7cc7a`。

## 用途和实现

将Issue347的度数/距离匹配控制接入已完成GraphRouteRun的选中态，冻结合同、选步、权重、全部开发验证块及掩码。提供none、important、low_weight与random64/512/2048三头OA/AA/NLL、混淆矩阵、逐类召回及实际图/概率身份。任何块没有合格替代掩码时，整个对照不可用，不能减少验证人群。保留原CNN输出、原训练数据/预算/选模，不触及query。

距离使用固定种子网格中心的像素坐标，不是学习后区域重心或地面物理距离；继承逐行/逐列度数与全图距离分箱约束，不能扩称精确距离/连通性/权重质量匹配。随机线性目标不是均匀掩码抽样。原模型、训练器和默认配置未修改。

## 实际验收

授权LXC `lbx@10.69.37.243:1222`；Python `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`；干净`checkouts/wp06-matched-review01`，CPU单线程、禁用CUDA。

```sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests
python -m unittest test_matched_edge_review test_matched_edge_controls test_graph_edge_intervention -v
python scripts/check_repository.py --expected-commit bfcd14c04662bf57a1ce1388fb4185a9aa1fe42c
python /home/lbx/hsi-pioneer271828/runs/wp06-matched-review-evidence01/audit.py
```

CPU15/15通过、无跳过，8.911秒；新4项含实际训练网络三头、原none/important精确对照、优化器/RNG/梯度不变、零删、单块超时导致全控制不可用、身份与验证范围负例、完成态精确恢复。checker561 Python/187 JSON通过。未跑GPU或全仓回归。

独立留存：一个合成数据实际网络24更新，路由预算4；之后6组冻结对照全部可用。逐像素循环独立复算1458混淆矩阵格和162召回，核对三头OA/AA/NLL与48项度数/距离约束。保存模型和计划再加载，结果精确一致；checkpoint、源模型、训练模型及history不变。复盘更新0/query0；合成初始化训练24步不当作真实训练。

证据`runs/wp06-matched-review-evidence01`，含保留的audit.py、trained.pt、plan.json、review.json与receipt.json，3.527918秒。receipt SHA256 `5fddea79378a0899b9c0dafe721006a6d7f4a0e576698603c74ec94bda2abec3`；同字节副本`docs/architecture/wp06-matched-edge-review-engineering-results.json`。测试日志`runs/wp06-matched-review01-tests.log`，固定提交检查日志`runs/wp06-matched-review01-checker-pinned.log`。

## 结论与未完成项

CPU接入工程通过。该接口复用选模开发验证集，不能视为独立确认、因果关系或真实分类收益；本轮没有HongHu或任何真实场景新指标。小类保护仍需真实固定协议逐类核对。无实际测试失败；超时/错误身份为预期拒绝。第一次SCP连接超时结束，重传exit0后才使用bundle，未影响输入。

下一项沿WP06完成冻结删边图的等预算恢复训练工程：在所有开发训练/验证上下文登记固定支持，以相同初始化、曝光步数和选模规则比较原图、重要边、低权重及随机控制；继续保留不可行对照。WP06完整真实实验仍未完成。
