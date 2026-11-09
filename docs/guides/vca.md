# M2：VCA 接口与缓存

入口仍是 `model/variable_channel_adapter.py::VariableChannelAdapter`，未重写模型或更改默认 K、宽度、局部窗口、损失和训练预算。输出为 `[H,W,K*value_dim]` 或 `[N,K*value_dim,H,W]`；三种 `wavelength/index/uniform` 模式保留。研究计划中的首轮 K=64、embed_dim=32 是显式实验设置，本模块真实小块验收采用此设置，现有构造器默认值不变。

## 元数据与坏波段

旧调用 `adapter(x, wavelengths, fwhm, return_attention)` 继续有效。新增可选关键字：

```python
y, info = adapter(
    x, wavelengths_nm, fwhm_nm, return_attention=True,
    wavelength_unit="nm", fwhm_unit="nm", band_mask=valid_bands,
)
```

- `wavelength_unit`、`fwhm_unit` 独立支持 `nm / normalized / auto`；默认 auto 保留旧阈值推断。新实验应显式指定单位，尤其 FWHM 小于 2 nm 时。nm 按构造器波长范围归一化；波长仍沿用旧实现的 `[0,1]` 裁剪。
- 缺失波长时使用归一化索引，`wavelength_source=index_fallback`、`wavelength_unit=normalized_index`，不能解释为真实物理波长；显式要求 nm 却缺失波长时报错。`provided` 仅表示调用者提供，不能据此认定已测量或已验证；质量登记仍由数据协议负责。
- index/uniform 模式使用重新排列的归一化波段索引，忽略波长/FWHM 对注意力的影响。`position_source` 描述实际位置来源。uniform 对所有保留波段平均 value 投影，仍有可训练 value 网络。
- `band_mask` 是长度等于原始 B 的布尔向量，同步过滤影像、波长和 FWHM，再投影。掩码后保留至少一个波段，`band_indices` 返回原输入索引。不自动探测坏波段；也可在上游同步裁剪三者。缺失波长的 wavelength 回退保留原归一化位置，index/uniform 在过滤后重新编号。
- 保留波长/FWHM 必须有限，FWHM 非负；长度不匹配、空观测、无效掩码直接报错。影像有效值仍由数据协议检查。没有覆盖的锚点输出零、`anchor_valid=0`，注意力数值保持有限。

已有诊断 `attention[K,B] / anchors[K] / anchor_valid[K] / wavelengths / band_positions / attention_mode` 保留，新增单位、来源、原波段索引和 `fwhm_normalized`。anchors、位置和 FWHM 诊断均使用归一化尺度，attention 仅包含保留波段。M1 的采样策略和 E1 的连续覆盖描述另行实现。

## 缓存契约

只在 eval、禁用 autograd、关闭 autocast、dropout 参数为零时缓存注意力；像素 value 始终重新计算。`cache_attention=False` 可在构造时或运行时禁用缓存，`clear_attention_cache()` 可显式释放。

train/eval 切换、设备或 dtype 应用、已有对象及其父模块的权重加载均清理缓存。缓存键包含有效波长/FWHM 的精确值、注意力设置，以及模型参数和缓冲区的身份/版本；因此常规优化器更新、原位更新和参数替换后会重新计算。仅保留最近一个条目，反复切换波段子集不会累计显存。返回诊断不能原位污染缓存。

eval 不会关闭梯度。eval+autograd 始终重新构建注意力计算图，query、key、锚点和元数据梯度均保留。autocast 推理直接重算，避免缓存混用精度上下文。若模型本身在 inference_mode 中创建、张量没有版本计数器，则直接重算。通过 `.data` 或外部共享存储修改权重会绕过版本计数，此类操作需显式清缓存或禁用缓存；支持的更新方式是优化器或 no_grad 下的正常原位操作。

缓存不会进入 state_dict，旧 checkpoint 键保持严格兼容。不会持久化整个 Python 模型对象；继续使用仓库现有 state_dict checkpoint 入口。

## 远程验收

仅在 `ssh rmvision` 的独立 checkout 执行：

```bash
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python scripts/validate_m2.py \
  --expected-commit <完整SHA> --device cuda \
  --data-root /home/train/hsi-pioneer271828/datasets/current \
  --out-dir /home/train/hsi-pioneer271828/runs/m2-<唯一批次>
```

入口要求容器、正确 SHA、干净工作树、checkout 外的新产物目录，运行完整 CPU 单元测试和 CPU smoke，并对比基准提交的旧 VCA 输出、梯度和 state_dict。指定 `--device cuda` 时另运行 CUDA smoke，真实小块在 GPU 执行；默认 cpu 便于独立复核。真实 PaviaU/Xuzhou/HongHu 各取固定 12×12 小块、全波段，三种模式各一步优化并检查缓存与 fresh 结果。小块内缩放仅用于数值检查，不代表正式空间训练统计或精度评估。标签不参与本验收。失败记录保留，重试使用新目录。

M2 证明接口和梯度链路可用，不证明三种模式的精度优劣。正式精度比较需要冻结同一主干、划分、容量和预算，另立实验 Issue。用户已授权 GPU 小规模验收，长训练不属于本次验收。
