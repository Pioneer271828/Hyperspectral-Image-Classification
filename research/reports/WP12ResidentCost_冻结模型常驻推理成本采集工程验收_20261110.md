# WP12 冻结模型常驻推理成本采集工程验收

对应 Issue364，主计划 §20/§22、实施计划 §19.13，依赖 Issue362/363。分支 `codex/wp12-resident-cost`，实际测试提交 `e13dd9514a38d9e4b91a6fd859c635c6741ca927`。仅新增采集、汇总入口与测试；原训练、模型、选模、损失和冻结生产代码未修改。

## 实现和口径

`evaluation/wp12_resident_cost.py` 和 `scripts/measure_wp12_resident_cost.py` 接收独立固定的原生权重、架构 sidecar 与无标签观测。加载前后核对字节哈希，加载后将所有参数和缓冲区与检查点张量逐项指纹核对。登记固定场景、种子、实验臂、精度评估身份、检查点、观测张量身份、工作量、设备环境、原生/ordered 算术以及预热和重复次数。

采用独立新进程的常驻整图三头前向。预热在计时前；CUDA 在计时前后同步；计时排除导入、加载、数据准备、传输、BN校准、指标计算和结果写入。前后要求三头有限且输出指纹一致，权重、缓冲区、观测和输入文件不变；不训练、不校准、不读取 query 标签。额外标签字段或非约定观测类型会被拒绝。CUDA 每进程仅一张可见卡，按 UUID 核对物理卡可用显存至少12288 MiB、利用率不超过60%；不符合则拒绝启动，不操作其他进程。

内存口径不能混用：CPU 是**独立工作进程整个生命周期峰值RSS**，包含导入、加载、前后校验；CUDA 是**计时前向期间峰值已分配张量字节**，包含已常驻模型和输入，不是设备总占用、缓存保留量或模型增量内存。`package_bytes` 是固定权重文件加架构sidecar的实际文件字节，不包括解释器、依赖库或数据。口径写入环境身份；不能将该字段宣称为完整部署镜像大小。

`assemble` 要求完整登记案例、同一环境与范围、每例同检查点/工作量及同评估身份，才生成 Issue363 接受的 `wp12-frozen-hardware-cost-v1` 回执。CLI 只写新的 checkout 外文件，输出存在即拒绝；输入和代码在结束时再次核验。该模块是成本采集器，不替代历史运行时兼容性审计及实际测量计划的事前登记。

## 远程验收

授权 LXC `lbx@10.69.37.243:1222`，目录 `/home/lbx/hsi-pioneer271828/checkouts/wp12-resident-cost02`，解释器 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，Python3.10.20/PyTorch2.7.1+cu128。环境与所有运行记录位于 checkout 外 `runs/wp12-resident-cost01/`。

```sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests PYTHONDONTWRITEBYTECODE=1
python scripts/check_repository.py --expected-commit e13dd9514a38d9e4b91a6fd859c635c6741ca927
python -m unittest tests.test_wp12_resident_cost tests.test_wp12_evidence tests.test_unified_cohort tests.test_repository_delivery -v
```

最终25项中24通过、1项显式GPU实测跳过，39.207秒。CPU覆盖原生网络实际前向、独立CLI进程、输入不变、输出独占、错SHA/环境/步数/工作量/范围拒绝、标签混入、NaN、模型被修改、物理GPU资源门槛边界、全54合成成本记录进入精度消费者并形成18摘要，以及缺例/重复/错配负例。全54单测使用合成计时，不是54真实模型测量。GPU资源门槛测试是模拟资源返回值，不能代替CUDA实际运行。

仓库检查通过582 Python/198 JSON。额外保留独立原生网络合成输入的实际CPU CLI记录：36像素、61696参数、权重及sidecar共272276字节；4次延迟为3.198476、3.153775、3.057507、2.992029毫秒，进程峰值RSS为578379776字节。这是工程夹具，不能当作PaviaU真实整图成本。`runs/wp12-resident-cost01/native-cli02/measurement.json` SHA-256为 `159c49b5718bace836a95237b337e266a53569f1c06a3cb71166506a32d1526f`；对应 `job.json` 为 `119a7472e5cdcb2adc8cd675ed360031c0b6bfe81cd67016b894a5104207ce7b`。

首版 `e7f7f9b33b596a0946a17aea3da2d71cc3f27865` 的23通过/1GPU跳过、40.438秒记录保留；随后增加物理资源门槛和观测类型检查并重验。同步首次从仅含HEAD引用的增量bundle按分支名fetch被拒绝；改用HEAD成功，失败时未执行测试、未改远程旧checkout。Sparse CSR beta提示来自既有PyTorch ordered实现，保留日志。

## 验收结论及下一项

**CPU工程与汇总接口通过。CUDA执行和真实54模型成本未验收。**本轮真实训练更新0、新query评分0，WP04真实短测继续延期。该模块不提供真实分类精度变化或提速结论。

继续推进同一冻结54模型的成本观测准备和历史加载兼容性检查；GPU有条件后再按固定预算补CUDA及全案例成本。真实精度表仍沿用既有冻结评价，不能因新计时切换推理算术或改变模型后继续冒用旧精度。
