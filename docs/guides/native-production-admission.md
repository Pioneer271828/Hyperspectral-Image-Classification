# WP05/WP06 原生候选准入和最小比较

原生训练入口为 `scripts/run_native_released_points.py`。生产实验共用现有 `native_scale_training.Run` 的三头稀疏监督、首个最低校准融合 NLL 选模、Adam/RNG 状态和前缀续跑格式。配置由 `transfer/native_production_model.Switches` 明确指定。

| 开关 | 值与边界 |
| --- | --- |
| `arm` | `native`、`center`、`spatial_shift`、`pixel` |
| `spectral` | `none` 或 `continuous`，同一对照内一致 |
| `residual` | `none` 或现有 `x_residual`，原生、中心和错位臂可独立开启 |
| `graph_scale` | 登记原生格网和 16/50/100 尺度之一 |
| `axis` | 已确认实际交付带 ID 的唯一升序轴；每个输入保留列必须是该轴的原序子集 |

中心和错位臂用同一个定值线性投影、同样的像素索引和融合头入口。中心臂按原像素，错位臂按登记格网的固定平移。两者的原生谱支路和 CNN/图头照常走。纯谱参考复用逐像素 MLP，按原带 ID 对齐，已删列在零填充容器里没有观测值。最小四臂比较使用 `residual=none`，纯谱参数量接近原生。这套生产工程使用已交付点的局部实现；不依赖外部模型代码。

## 确切发布材料

1. 按[独立光学与标签交接](independent-optical-intake.md)取得真实独立 Bank、清单和单独审批。登记 support、calibration、evaluation 的样本、块、最小缓冲、全类分母和既往曝光记录。真实进场需要 `independent_accuracy` 和 `cost` 资格。
2. 准备合法共享源检查点 `weights`、实际 `.pt.json` 元信息 `metadata` 及模型 `state` 哈希。每个文件用授权 runs 根下的绝对路径和 SHA-256 绑定。源与目标同尺寸张量采用现有 29 项共享清单，图和头按目标重新初始化。旧版带元信息检查点可显式提供原生开关和轴，用 `open_production` 严格导入。只有权重时显式提供原架构，用 `open_legacy_weights` 严格导入。
3. 持有人以 `wp05_wp06_real_production_intake_v1` 提交 `independent_manifest`、`independent_authority`、`source`、`axis`。最低配对为 `model_seeds=[64,512,2048]`、`arms=["native","center","spatial_shift","pixel"]`、`views=["full","central_gap","matched_dispersed"]`，另登记 `spectral`、`residual`、`graph_scale` 和正整数 `updates_per_arm`。预算、曝光、选择规则、算术、设备和评价阈值需在任何目标外推前锁定。现场若有缺元数据条件，另登记保留列和格网，不与单因素删带混报。
4. 预检需要独立的 `wp05_wp06_real_production_scope_approval_v1` 记录，含准确的 `approved_request_sha256` 和 `producer`，用途为 `read_only_preflight` 且 `train_authorized=false`。可信校验值由审批方在授权通道提供。该命令核对角色、源检查点、轴和各臂容量，可生成只读预检回执。

```bash
python -B scripts/check_wp12_production.py real-preflight \
  --runs "$RUNS" --out "$OUT/real-production-preflight.json" --expected-commit "$REV" \
  --request "$INTAKE" "$INTAKE_SHA256" --authority "$REVIEW_APPROVAL" "$REVIEW_APPROVAL_SHA256"
```

真实训练许可需另行签发并登记最终工程版本、实际角色和设备。同一份预检批准只覆盖只读核验。当前可运行的训练 CLI `production-register`、`production-run` 使用合成正式路径。真实源/目标的许可、标签或独立结果不由该命令生成。第一次真实比较保留三个模型种子和逐支持种子分母，报告同点逐类损益及小类新增错误，复用相同支持、波段、原生格网和成本工作量。

部署预测经 `open_production`、`run_forward` 取得导出的开关、真实保留列和检查点，随后投影到原像素。FWHM 缺失或无物理轴走适配器现有回退。成本从已登记观测读取开始，含变换、转移、图、三头、可选中心投影及原格输出同步。
