# WP09 Tent 同协议适配与 OpenOOD 冻结评分

关联 [Issue #473](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/473)、[PR #474](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/474)、原 #415/#325/#327/#329。主计划 §13/§22.5，实施 §19.10/§19.15。独立分支 `codex/wp09-oss`，基线 `693a792881006327f1a45b02033156a08b482f00`。

## 固定上游和职责

Tent 的原论文以预测熵优化归一化仿射，并在测试批估计统计量。固定 `e9e926a668d85244c66a6d5c006efbd2b82e83e8`，执行原 `softmax_entropy`、`collect_params`、`configure_model` 以及 `Tent.forward/reset`。收集器遍历全部 BN2d（含非活动域），不含区域 BN1d。配置关闭 running stats 并清空均值/方差，启全模型 train。上游 episodic 每个调用复位；两步返回的是第一次更新之后的预测。WP09 需要先保存三头、只一步 SGD、按场景/段复位、独立统计臂及原有效归一化白名单。直接接入 Tent 包装器拒绝。只为仿射与 Adapter 的融合 raw logits 调用熵组件，任一概率低于原下限 1e−8 则沿原夹紧公式。原生 `_Session` 管理其他前向、模型模式、统计和优化。 [Tent 论文](https://arxiv.org/abs/2006.10726v3) · [作者固定源码](https://github.com/DequanWang/tent/tree/e9e926a668d85244c66a6d5c006efbd2b82e83e8)

OpenOOD `3c35632ee91b54b09d1f085d04f94744cece7d0b` 的真实 MSP 在 `base_postprocessor.py`，energy 在 `ebo_postprocessor.py`，两个另名 `msp_postprocessor.py`、`energy_postprocessor.py` 在固定树中不存在。使用未改动的 `BasePostprocessor.postprocess`、`EBOPostprocessor.postprocess` 及 `metrics.auc_and_fpr_recall`、`fpr_recall`。MSP 将原始 logits 按冻结温度缩放并转换为 1−confidence，EBO 取负得到原负能量。两者要求明确的原始 logits 类型及冻结列序，概率和 logprob 不进入 energy。纯 IEEE 极端输入会按有限值门槛拒绝。相同输入数值和众数语义等价；KNN 和其他分布距离不承担原源训练均值原型的最小平方欧氏距离职责。保持原型公式及全部原阈值。 [OpenOOD 论文](https://arxiv.org/abs/2210.07242) · [固定 OpenOOD 源码](https://github.com/Jingkang50/OpenOOD/tree/3c35632ee91b54b09d1f085d04f94744cece7d0b) · [MSP 原论文](https://arxiv.org/abs/1610.02136v3) · [Energy 原论文](https://arxiv.org/abs/2010.03759v4)

OpenOOD 的 `auc_and_fpr_recall` 使用 −confidence、未知为正、选择第一个可达 TPR≥0.95 的 ROC 点，按相同分数成组，梯形 AUROC 对并列计半。语义转换将显式未知映射 −1，0 先剔除。另一个 `fpr_recall` 以已知为正，在案例上得到不同 FPR95。缺任一组由项目返回 N/A，上游原函数的指标无需充当空组默认。原已知校准采用保守经验分位数，`score > threshold` 才拒绝，并列保留；ROC 的 95% 仅用于报告。原始 logits 的公共偏移保持 MSP、改变 energy；极小/极大温度、非连续列、相等分数和错误值均覆盖。

Tent 根 MIT 覆盖固定 `tent.py`。OpenOOD 根 MIT 用于审查表列出的 Base、EBO 和 metrics。OpenOOD 的 `comm.py` 保留 Facebook 版权头。固定路径历史起于 `314e154f66d6c360d6f0d46531feac0916ae80a3`，后经 `aa08687e560711a387e2407b25bde40e7f80a7db` 重组。与五个同期 Detectron2 官方版本比对均有字节差。来源及实际文件许可链标记 unresolved。从候选加载器移除该文件的认证和导入。为满足 Base 原导入绑定，注册项目自写的占位模块。任何分布式 API 访问立即失败，原评分及指标函数保持字节相同。此前数值证据保留，新补充归档仅含许可已核实组件。 [文件和许可清单](../../docs/references/wp09-tent-openood-review-20261114.json) · [OpenOOD 路径首次提交](https://github.com/Jingkang50/OpenOOD/commit/314e154f66d6c360d6f0d46531feac0916ae80a3) · [同期 Detectron2 官方许可](https://github.com/facebookresearch/detectron2/blob/7801ac3d595a0663d16c0c9a6a339c77b2ddbdfb/LICENSE)

## 工程和源准入

原在线执行器只增加可选 session 工厂及 provider digest。默认工厂与绑定 schema 沿用原版；可选输出的根绑定加 provider，原生与可选不能串接恢复。现有持久化、原子 journal、种子、统计白名单、上层 CLI 的默认入口和模型无变动。冻结头、非活动 DomainBN、原统计缓冲及禁 VCA 缓存通过原生约束验证。专项 9/22 类走原三头四臂，连续和到 1/2/6 的恢复，适配臂跨进程，完成重入 0。注入故障在首帧预测落盘、更新前发生，原目录拒绝重试；GroupNorm 统计臂运行前拒绝。Tent 专项 3/3，116.670 秒；原生相关回归 65/65，462.931 秒。首次运行引用层级过深的 Git 备用对象，没有产生 checkout，执行 0；第二次在自身运行中遭候选快进，精确 SHA 门槛拒绝余下四个子场景，最终在不变的独立 checkout 完整通过。失败及执行原日志保留。上游配置及多步语义差异用小张量模型保留。补验源码 `17fe302acf498214f75ea06736659eb72eb362ca` 在可训练仿射会话上覆盖 1e−8 两个紧邻侧与极端有限 logits。低于下限时上游熵调用为零，损失、参数梯度及一步 SGD 更新逐位匹配原生。上侧调用一次上游熵，损失、梯度和更新在预设 FP64 容差内一致。NaN、正负无穷都在梯度和 SGD 前拒绝。原生概率接口可表达负无穷的零质量，候选原始 logits 接口要求有限值。OpenOOD 两个专项也在占位模块下通过，并直接验证真实 `BasePostprocessor.inference` 访问分布式方法时失败。补验合计 3/3，28.768 秒。真实回放证据沿用已核实版本。

OpenOOD 通过原可信源 v2 的授权根、原准备/归一化和源模型末步回执，确认训练标签与统计仅来自已知源支持。沿现有 E4c 特征/原始 logits 生产、9 个已知校准阈值冻结和后置标签发布。按三个冻结头与原列 2/7/19，合成显式未知 31。源 96 个完整标注、57 个监督点，校准矩形 24 个像素抽取 12 点，评分 12 样本含两个未知、两个未标注，源/校准/评价三个 ID 集互斥。温度融合/图/CNN 为 1/2/0.5，覆盖率 95%。最终合成源训练严格 2 步，冻结期间标签尚未发布，读取哨兵无命中。该专项的首个已执行迭代因 CLI 参数格式在 OpenOOD 准备前失败，已执行 2 源步；第二个完成迭代用无未知正例做空组，已执行 2 源步；最终迭代改为两个未知正例。专项三轮源 Adam 共 6 步，原生可信源回归另计。错误源、provider、语义列、冻结包、生产包在标签读取前拒绝。原 v1 无可信根且不获评分准入。

OpenOOD 专项 2/2，27.355 秒。使用实际发布合成标签的末次 9 个三头×方法报告，独立复算混淆、OS*、已知 macro-F1、HOS、未知逐类召回与 AUROC/FPR95；前六个 OpenOOD MSP/energy 的概率评分最大差 1.11×10⁻¹⁶、冻结拒识决策相同。对齐的 OOD-positive AUROC/FPR95 与原指标匹配。已知为正的口径另记。下表为合成三头各方法的 AUROC 和 FPR95，结果仅用于验证接口。

| 方法 | 融合 AUROC / FPR95 | 图 AUROC / FPR95 | CNN AUROC / FPR95 |
| --- | --- | --- | --- |
| MSP | 0.6250 / 0.5000 | 0.3125 / 0.8750 | 0.2500 / 1.0000 |
| Energy | 0.5000 / 0.6250 | 0.2500 / 0.8750 | 0.3125 / 0.7500 |
| 原型 | 0.3750 / 1.0000 | 0.7500 / 0.3750 | 0.1875 / 1.0000 |

## PaviaU 已见训练区无标签诊断

实际执行 `09e7e68a5e220b1c4a733645c114d2fe0ea16480`；源码与已有历史注册的模型、旧会话接口保持相同。登记复核原认证 PaviaU 第 7 步源、两个原训练区观测、24 组 scope `36b42c6377357f7b93057adcbc25dc6ce5335e938eebf8ace8da79df1c90688d` 与原接受回执。旧登记冻结原观测和科学字段，只为候选更新代码身份。授权容器北京时间 08:36 开始，四臂×两复位×三顺序，各 6 连续及 2+4，288 个三头预测，144 次无标签 SGD，72 次 BN 统计前向，此外适配前向总计 216。24 次完成重入 0。历时 663.338 秒（含对照）。24 条连续与分段候选逐帧事件、权重、优化器、RNG、游标相同。原活动与冻结范围一致。对原历史逐元素比较 432 份三头数组、3,048,192 个概率，最大差 2.98×10⁻⁸，最大容差比例 0.015109，分类变化 0。504 个梯度张量最大差 1.46×10⁻¹⁰，232 份有数值差的状态张量最大差 5.82×10⁻¹¹，改变参数集合一致。原 82 个输入保护 pin 前后相同。

独立审计 `7738174832cae56e4d2670a966f0dacbd7857e6f` 加载认证状态并重新前向 144 帧，匹配候选三头和原容差，优化与统计更新 0。原 GT/query、真实标签读取、真实开放集评分和准确率结论均为 0。该数据由同场景已见训练区组成。真实开放集仍缺独立 v2 已知源、开发释放和研究确认。

## 成本与验收

单线程 CPU，原生和候选每项 3 组预热、12 组 AB/BA，p95 采用 inclusive 插值。旧成本包含 SHA 认证和原生模型检查点。表中读观察、预测、适配和 checkpoint 写入来自 `full(s)`，生产 `prediction.npz` 和完整原子 journal 事务在计时边界之外。OpenOOD 两行包含原评分再加 oracle 对照与审计，只表示附加审计成本。运行成本和首次源加载分别报告。合成 9 类仿射成本共 30 次 SGD，原生与 Tent 各 15。两轮固定 Tent 项目夹具各 144 预测、72 SGD、36 统计更新（9/22 类 × 四臂 × 原生、候选连续和候选分段），合计 288/144/72。第二轮故障测试另完成 1 预测、0 SGD。上游两轮玩具手工构造各 5 次 SGD，合计 10。原生回归内的其他独立 fixtures 单列，不计入可选路径或真实。

| 单线程操作 | 原生 p50 / p95 ms | 适配或审计 p50 / p95 ms |
| --- | --- | --- |
| 读观察、预测、更新及状态写入 | 28.66 / 41.60 | 29.25 / 45.22 |
| 读并恢复检查点 | 15.74 / 16.05 | 15.65 / 16.32 |
| 独立持久化状态 | 6.56 / 6.96 | 6.36 / 7.09 |
| 熵原语 | 0.041 / 0.060 | 0.035 / 0.059 |
| 六个 raw-logit 评分及对照 | 0.237 / 0.300 | 1.380 / 1.843 |
| 九法评分和指标复算 | 5.33 / 5.75 | 7.41 / 7.87 |

Tent/OpenOOD 新进程中位 5.49/6.01 秒，认证和导入约 35/55ms，当时加载的上游源码和许可证共 5,490/12,165 字节，OpenOOD 字节数含历史 `comm.py`，修订路径未重新测成本。该成本进程测得峰值 RSS 622,084 KiB，原合成完整恢复检查点 342,781 字节。原生执行及接受的证据均留存原始样本。CPU 工程不替代 GPU 资格，本包没有 GPU 验收。

P4 原范围 `a827dec0a6087d1eaf990ca086f1c28dd27e501ba0ae81c3b2c18434672e9ce5` 与生产 `3a6381cb4b0c089a10123292bc831a3a613e6106` 保持认证。阶段边界只有 GPU 6 达标，GPU 1 空闲 375 MiB。原 0/2,400；13,500、180,000 保持暂停。

## 复现和交付

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest -v tests.test_wp09_tent_oss tests.test_wp09_openood_oss
python scripts/review_wp09_tent.py --preflight-only --expected-commit "$SHA" --runs "$RUNS" --tent "$TENT" --out "$OUT"
python scripts/review_wp09_tent.py --expected-commit "$SHA" --runs "$RUNS" --tent "$TENT" --out "$OUT"
python scripts/audit_wp09_tent_oss.py --candidate "$OUT" --original "$OLD" --out "$AUDIT"
python scripts/review_wp09_openood.py prepare --expected-commit "$SHA" --authority "$AUTH" "$AUTH_HASH" --source-receipt "$SRC" "$SRC_HASH" --openood "$OPENOOD" --out "$APPROVAL"
python scripts/check_repository.py --expected-commit "$SHA"
```

真实模型与观测、梯度和预测的原始数组留在容器 checkout 外。打包 `wp09-oss-evidence01.zip`，含固定源码与许可、核验的原始/失败日志、候选逐帧事件、独立审计、配对成本、最终候选与主入口 checker。源码与算术比较均可复验。真实 receipt SHA `98a392bbfb417e0d948b6c47bba53c56446846458e83ab0de4fae7ee79443effd`，逐帧独立审计 `90b3dc24ffbcae274ff4468d870ded36f4038e164aa3ac4f002d55c94ed5c6b6`，OpenOOD 合成审计 `7ae8effa5ddff2c78b6ef1091e552b19e14adb33603d3d9a2e248963924a72198`，成本 `82dadf10038d76901c88b320c45518be476d37e6aa256c004990f2601d778326`。最终工程源码 `7738174832cae56e4d2670a966f0dacbd7857e6f`，OpenOOD 合成 `a5af9f41d06e4d62ac6967e19a7aaae5f864dd72`，成本 `3bfb558729cf7790b883cd79bbd0ced9998567a3`。Tent 只保留可选熵研究对照，OpenOOD 只保留冻结评分 oracle。现有默认、训练流程、评分和阈值继续使用原实现，真实收益须按独立数据前置条件另验。
