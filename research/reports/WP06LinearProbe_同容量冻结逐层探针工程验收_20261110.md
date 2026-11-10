# WP06：同容量冻结逐层线性探针

Issue345 / PR346；主计划§16、§22 WP06，实施计划§19.7。分支`codex/wp06-linear-probes`，tested SHA `b620ac1f69771400653e82938f4f7185b383d2f6`，基准main `93dbdc22c72554c4b7eb14afdd313882b6d7faec`。

## 实现与范围

新增冻结特征探针、外部CPU CLI和专项测试；复用原四阶段特征捕获与区域广播。每层固定列抽样到共同维数，参数统一为C×(d+1)，训练样本独立拟合均值/尺度，零初始化、全批SGD、固定步数/LR/L2、首次最低验证NLL选步。验证数据不进入优化或标准化。固定维数会丢失部分信息，不能当作全维最优线性分类性能。

输入仅有train/val特征、标签、互斥行号与身份；文件SHA与恢复状态校验、计划变化/覆盖输出/query字段拒绝。CLI不加载原始数据或主网络，不更新骨干。输入哈希只认证字节，**不能单独证明上游模型、划分与标签权限**；真实使用必须另接已接受的冻结捕获边界。当前调用层捕获只声明调用者提供的train/val身份，不冒充完整真实数据准入。

参考Alain/Bengio arXiv1610.01644v4已读§1、§3.1—3.4；有界检索未核实专用作者代码固定SHA和文件许可，未复制或执行外部代码，按公式独立实现。详见`docs/references/wp06-linear-probe-reference-review.json`。

## 验收环境与命令

授权LXC `lbx@10.69.37.243:1222`，Python `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，干净`checkouts/wp06-linear-probes01`。CPU FP64、单线程、CUDA_VISIBLE_DEVICES为空。

```sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests
python -m unittest test_linear_probe test_frozen_representation_capture.FrozenCaptureTests.test_actual_six_synthetic_roles_original_stages_predictions_state_rng_and_cleanup test_frozen_representation_capture.FrozenCaptureTests.test_observer_exception_and_policy_refusals_remove_temporary_attributes test_frozen_representation_capture.FrozenCaptureTests.test_changed_consumer_rejects_historical_capture_before_model_load test_representation_redundancy test_monitoring_probes -v
python scripts/check_repository.py --expected-commit b620ac1f69771400653e82938f4f7185b383d2f6
python scripts/run_frozen_linear_probes.py --help
```

最终23/23通过、无跳过，27.786秒。含独立NumPy梯度及有限差分、3/9/22类、同参数量、连续12与2+10及完成恢复0、身份篡改、train-only统计、验证扰动不改变训练参数、原网络四层捕获/状态/RNG不变与CLI实际恢复。不是全仓回归。首版7项10.788秒、相关16项20.124秒通过；增加并列分数诊断后重跑上述最终23项。GPU与真实训练未执行。

最终日志`runs/wp06-linear-probes-tests01/final01.log` SHA256 `58a68473eca9429cd9b62f7e1c631482591bd9a8f1b28d0ead1e51a00c67b9ab`。

## 独立留存结果与失败

`runs/wp06-linear-probes-evidence02`：3/9/22类合成数组各两探针，另一个初始化合成Xuzhou网络四层，共10探针。连续组120探针更新，恢复组另120，NumPy参考优化另120；骨干/真实训练更新0、query0。10组连续/恢复状态完全一致、完成重入0；11输入文件不变、32输出文件登记。耗时1.129626秒。

独立NumPy优化权重在既定容差内一致，选步相同；基于权威选中态log概率独立核对20组指标、2944混淆格、208逐类召回，NLL最大误差4.440892098500626e-15。该指标oracle没有独立重做骨干前向。

**跨NumPy/PyTorch的严格分类零差异未通过。** 初次留存脚本在混淆断言失败，目录evidence01与日志原位保留，日志SHA256 `9ce5ec4961fd7fa33a527237f2b5c206836c1138002789b38f93f52841c3f9a9`。定位发现初始化图特征存在并列/近并列分数；即使使用同一权重，浮点矩阵运算与log-softmax也可改变argmax。最终独立重训比较仍有stage1_graph训练11/验证3、stage2_graph训练23/验证10个决策差异，receipt明确`cross_backend_strict_decisions_passed=false`，没有改变门槛或删除负结果。

修订仅增加exact_tie_rows、最小log概率间隔及roundoff_sensitive_rows诊断，并纠正比较脚本采用log概率argmax。敏感行标记不改变预测、不四舍五入、不用来豁免分类差异。receipt的`status=passed`只表示所列工程与权威概率指标核对通过，不代表独立概率或跨后端分类一致性通过。

四层初始化网络的合成验证OA依次为CNN1 22.2222%、graph1 31.1111%、CNN2 40%、graph2 31.1111%；选步4/12/12/12，各585参数。全部探针存在零召回类；这些数字不是数据集分类结果、充分收敛或层间语义收益。真实训练模型、空间重复、GPU、丰度探针与因果干预仍未完成。

回执`receipt.json` SHA256 `e254bfc814e3b1dc0c2a25145bde65938ba4c047c3f4c7cbf884821e472904a2`，同字节副本`docs/architecture/wp06-linear-probe-engineering-results.json`。外部脚本`transfer-records/wp06-linear-probes-evidence02.py` SHA256 `ba01a5f9ff9898f81abe58060350a669b0087c6568ff2542d4941a60db69b3dd`。

## 交付结论与下一项

冻结探针CPU工程通过，跨后端严格决策一致性负结果保留；WP06真实机制实验未完成，不作为真实收益准入。下一项按实施§19.7补重要/低权重/随机删边的度数与距离控制，先核现有删边接口与可行性，再登记工程范围。当前GPU空闲显存均未达12288MiB，本次无长训练、定时值守未启用。报告在本聊天交付一次后继续计划。
