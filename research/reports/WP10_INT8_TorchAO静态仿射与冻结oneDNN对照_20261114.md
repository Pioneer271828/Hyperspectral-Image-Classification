# WP10 INT8 静态仿射与冻结 oneDNN 对照

承接 #271、#417，研究主计划压缩补充与 §22.5、实施 §19.11/§19.15。候选 [PR #472](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/472)，基线 `780e8ea18de452040369ff063569c7d8bac4326b`。执行代码 `4bec8a2537b76ab9e8f4d068b9952bdf4e367b8a`。生产保留完整 FP32 默认。

## 来源、许可及适用范围

Jacob 等论文 §2–3 描述整数累加、激活/权重仿射尺度与 zero-point、偏置和重新量化。WP10 使用其中的算术关系作为参照。独立参考先以 int64 对每像素计算 `sum((qx-zx)*qw)`，乘输入和逐输出通道的权重尺度，加 FP32 bias，再计算输出仿射量化。WP10 只量化原两个独立 1×1 point convolution，输入 uint8 0–127，权重 qint8 −127–127、axis=0/zero-point=0，输出 uint8 0–255。保留原输入乘倒数和舍入路径，包括边界。整个模型仍有浮点、图构建和三头。 [Jacob 等](https://arxiv.org/abs/1712.05877)

| 固定来源 | 核查及实际使用 | 许可 |
| --- | --- | --- |
| pytorch/ao `4ab9d61da2937831743bc3f7904975fe1b32de7f` | 官方 TorchAO，核对原 QAT Linear、仿射 primitives、发布说明与依赖。不改公共运行环境。Python-only wheel 0.19.0 能构建，在 Torch 2.7.1+cu128 中导入缺 `ScalingType`，该提交在冻结环境不兼容。 | 根 BSD-3-Clause，`LICENSE` SHA `67901e351baeaa5aef967cccd0886ba679f085df8673d1ecd49592bf2844115af` |
| pytorch/ao v0.15.0 `9338966da58ec44b60f0e0b173cabab08f942ed0` | 官方兼容表包含 Torch 2.7.1，隔离 Python-only 源码 wheel `0.15.0+git9338966` 实测可导入。核查 `quant_primitives.py`、`pt2e/quantizer/x86_inductor_quantizer.py`、`quantize_pt2e.py`、静态 Linear 教程、`kernel/intmm.py` 及对应测试。复用静态仿射 quantize/dequantize 作算术参照。 | 根 BSD-3-Clause，`LICENSE` SHA `cbede3898d33d186efffb51c40200f7136170089ef1adba4dffc3f0a2abcea3a`。CUTLASS 子模块未展开或编译，其许可证不用于本接入。 |
| PyTorch `e2d141dbde55c2a4370fac5165b0561b6af4798b`，2.7.1+cu128 | 官方框架，核查 `qconv.cpp`、`qconv_prepack.cpp` 与 Inductor `fx_passes/quantization.py`。实测 `onednn::qconv_prepack` 将原 int8 权重打包为 MKLDNN layout，`onednn::qconv2d_pointwise` 以 MkldnnCPU 分派。 | PyTorch 根 BSD-3-Clause，已链接依赖遵循各自许可。 |
| oneDNN 3.7.1 `8d263e693366ef8db40acc569cc7d8edf644556d` | 冻结 PyTorch wheel 构建配置记录的实际依赖；保留官方 LICENSE 和第三方许可清单。 | Apache-2.0，`LICENSE` SHA `9dad27256276bb4fce6725c162ef619b5aa2e8abc8f341a4dab0b870aad55664`。 |

[指定 TorchAO 源码](https://github.com/pytorch/ao/tree/4ab9d61da2937831743bc3f7904975fe1b32de7f) · [官方兼容表](https://github.com/pytorch/ao/issues/2919) · [兼容发布](https://github.com/pytorch/ao/releases/tag/v0.15.0) · [x86 量化器源码](https://github.com/pytorch/ao/blob/9338966da58ec44b60f0e0b173cabab08f942ed0/torchao/quantization/pt2e/quantizer/x86_inductor_quantizer.py) · [PyTorch 卷积源码](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/aten/src/ATen/native/quantized/cpu/qconv.cpp) · [oneDNN 固定许可](https://github.com/uxlfoundation/oneDNN/blob/8d263e693366ef8db40acc569cc7d8edf644556d/LICENSE)

TorchAO x86 PT2E 能标注静态卷积，但默认配置的权重区间含 −128 且 observer 管理输入和输出范围。WP10 直接消费现有整数包及其不同的输入和输出范围。冻结 Q/DQ 编译探针在无冻结选项时虽得到相同整数码，实际执行 `aten::_slow_conv2d_forward`。启用冻结后图中已见 `onednn.qconv2d_pointwise`，但冻结 PyTorch 的 Inductor 编译以 `UnsupportedOperatorException: onednn.qconv_prepack.default` 失败。使用已链接的独立 oneDNN 算子做最小 CPU 部署适配，并通过 profiler 证明真实整数分派。没有平铺 Linear、动态 token 尺度和附加观察器。Triton 3.3.1 来自旧 PyTorch 环境，候选实测未执行 Triton。编译诊断曾用外部隔离 GCC 12.4，部署无编译步骤。

## 工程和预算

- 原生默认路径无需 TorchAO。可选模块校验显式指定的 Git SHA、Torch 版本、已装发行版和所有安装的 Python 源码 Git blob，拒绝额外本地扩展及错源。包 sidecar 绑定原权重 SHA、state、校准计划、engine、PyTorch 和 TorchAO 身份，本身也经 SHA 校验。打包只声明原状态，回读从原整数码创建 oneDNN pack；绝不重读训练块来校准。
- 9/22 类各测第一层、第二层、双层及关闭、重载、错误包、缺依赖、构造失败回退条目。CPU 专项 4/4（7.314s），固定上游仿射 tests 4/4（0.055s）。非连续布局、正负极值、截断、零范围、有无 bias、整数累加及分派均实测。本轮候选仅支持部署，微测和 9/22 类测试实际优化 0。
- 原生 INT8、QAT/三值七臂恢复回归 22/22（191.623s），原回退和成本独立 8/8（2.145s）。最终七臂合成恢复 9/22 类各原 4 与 2+2，共 112 更新，含旧 QAT 梯度、实际变化和跨进程恢复，完成重入 0。其中 QAT 三臂合计 48 更新；FP32 16，三值 48。第一次回归的两个非训练用例因 `PYTHONPATH` 不含 tests 导入失败，9/22 类原生七臂仍完整执行了另 112 次。因此当前交接两个原生批次实际合成总数为 224，首轮和最终轮各 112，新增可选部署夹具优化为 0。构造校准 fixture 按各自训练块完成，与真实数据无关。
- CPU 冻结部署入口为 `scripts/review_wp10_int8_torchao.py`，配套审计 `scripts/audit_wp10_int8_torchao.py`。保持原数据、DomainBN/图、三头、loss、Adam 和选模。oneDNN 为 CPU 算子，本轮 GPU 部署状态为未运行。

## Xuzhou 冻结对照

原恢复生产 `9dea199653079c9c430264b72ac968f1fefd7d7e`，登记 `086eeb643eb1c0b2cea4299732299810bf8a18d9d6da0cba1c8eb2dc538d291d`，旧独审 `9588a95ca3db5ae461e686ae04ec75041f8108de075a248facea20b1d6b8bbb5`。核认证原七个包及原 WP08 off 246 步选中稠密权重，原 155×14=2,170 次更新仅为历史。按原授权 5 块 3,920 像素、K5/V5 各 45 标签，对原未恢复稠密及同预算 FP32、QAT 第一/第二/双层复算。三值只校验包身份。冻结校准使用原 devtrain 92、8 两块，验证只读。收尾 120 项保护记录一致。

四个实际替换点合计 1,505,280 个输出整数码，和原 x86 包逐位相同。TorchAO 算术参照与 x86 输入整数全部一致，双层包第一层有 3 个输出码相差 1，其余与 x86 一致；单级反量化最大差 0.045656。实际 oneDNN 替换在 5/5/10 次声明卷积上均获 profiler 见证，各候选在五块三头概率全部逐位一致。独立整数聚合复查 24 个混淆、27 个保护项和全部成对计时行。

| 冻结候选 | 融合／图／CNN OA 和 AA | 同预算 FP32 三头保护 |
| --- | --- | --- |
| 原 WP08 off 稠密 | 26.67%／44.44%／15.56% | 未恢复参照 |
| 155 步 FP32 | 37.78%／55.56%／17.78% | 同额参照 |
| 第一／第二层（分别） | 37.78%／55.56%／17.78% | 三头通过 |
| 双层 | 37.78%／55.56%／15.56% | 融合和图通过；CNN 第 4 类新增 1 错，召回 −20pp，OA/AA −2.22pp |

原融合选用策略仍选双层，三头保护保留双层失败。全部逐类、逐样本小类新增错误、Kappa 和 macro-F1 均在质量回执。此次仅替换算子，新增真实校准、优化、原始 GT、query 和独立确认均为 0。支持范围为固定开发 V5。

## 成本与资源

单线程 CPU，同角色五块整网，3 次预热后 12 组 AB/BA；另有同预算 FP32 控制。读取验证、预打包、重排、量化与反量化、在线构图及三头均纳入对应时间。以下为常驻 ms，括号为 p95，按旧 x86 / oneDNN。

| 替换层 | 常驻 p50（p95） | 原包 / 加 sidecar 字节 |
| --- | --- | --- |
| 第一 | 393.57（484.21） / 399.61（428.41） | 285,560 / 286,160 |
| 第二 | 384.39（405.95） / 386.74（402.10） | 284,536 / 285,137 |
| 双层 | 412.84（443.83） / 396.38（424.81） | 263,910 / 264,508 |

分别与 FP32 配对时，三个 oneDNN 路径的常驻中位约 409.80、395.36、388.51ms，同组 FP32 约 390.91、379.82、384.24ms。双层的同包常驻中位降低约 4%，12 组成对中位比值为 1.034×。两单层对应比值 0.990× 和 0.994×。未见跨三臂一致的整网加速。相同包在独立进程中，相对原生约增加 3.5–3.9 秒，主要为源码认证和导入。包认证约 29.9–59.4ms，独立进程转换和预打包 9.9–11.7ms，首扫约 412–442ms，峰值 RSS 原生约 666–668 千 KiB、可选路径约 722–725 千 KiB。12 个独立进程，文件系统缓存状态未固定。逐块 oneDNN 内核约 0.11–0.12ms，布局和码重排约 0.08–0.14ms，输入量化约 0.04–0.08ms，反量化约 0.04–0.08ms。原真实校准准备 0.224–0.242 秒，本次新增校准 0。秒级进程额外开销和毫秒级常驻差异不得混用。

锚定 P4 原 scope `a827dec0a6087d1eaf990ca086f1c28dd27e501ba0ae81c3b2c18434672e9ce5` 和生产 `3a6381cb4b0c089a10123292bc831a3a613e6106`。阶段性检查原 GPU 1/6，只有 GPU 6 达到资源门槛，UTC 00:06 与 00:14 两次检查均只有 GPU 6 达标，GPU 1 余 935 MiB。原执行目录未生成，真实 0/2,400 步。13,500 与 180,000 仍暂停。

## 复现与证据

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WP10_TORCHAO_SOURCE=/home/lbx/hsi-pioneer271828/references/wp10-int8-oss01/ao-015
python -m unittest -v tests.test_int8_torchao_reference
python -m unittest -v tests.test_int8 tests.test_int8_model tests.test_precision_recovery
python -m unittest -v tests.test_precision_fallback tests.test_precision_cost
python scripts/review_wp10_int8_torchao.py --expected-commit "$SHA" --campaign "$ORIGINAL" --torchao-source "$WP10_TORCHAO_SOURCE" --out "$OUT"
python scripts/audit_wp10_int8_torchao.py --quality "$QUALITY" "$QUALITY_SHA" --cost "$COST" "$COST_SHA" --out "$AUDIT"
python scripts/check_repository.py --expected-commit "$SHA"
```

真实模型和输入留在 LXC checkout 外。证据包含上游源码和许可、兼容安装与不支持记录、原参数与版本的固定证据、三头质量、整数计数、AB/BA、重载、合成测试及最终两份仓库检查。固定名称 `wp10-int8-torchao-evidence01.zip`。独立低秩续证据 `wp10-low-rank-tensorly-evidence02.zip` 共 15 个成员，SHA-256 `67822f97974ada4f89c45e8e138518b73d971a5d1c8d875d26fc9c8f1851dbde`，清单 `3a131e6d57f65370ae8a09bb3c61b02c0d4a540e4a2fca8498db69f1cb6d595d`。
