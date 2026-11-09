# 外部登记的课程开发训练CLI（WP01）

对应主计划§22 WP01、实施§19.2、Issue335。复用CUR-S3/CUR-O2及多类别开发消费者，本入口只扩展外部编排，不改变模型、采样、优化或选模。参考沿用 `docs/references/curriculum-reference-review-20261109.json`：项目独立实现，未复制/执行作者代码，固定上游SHA/许可缺项仍保留。

## 输入与权限

- 使用 `transfer.registered_development.export_development` 已验接口导出开发区：只含devtrain/devval块和已释放10/5每类支持，原GT与其余观测不打包。登记CLI不接收原始数据根目录，也不生成划分。
- 源权重和 `.pt.json` 架构元数据分别提供外部SHA256。源须单任务匹配类别、GroupNorm、未耦合完整模型；当前合同为CPU FP32，不能添加 `--gpu` 假装已完成GPU接入。
- 外部ExposurePlan JSON固定预算、原class_first采样序列和逐类/逐支持最低曝光；不足则拒绝，不重抽、不追加步数。
- 样本课程JSON和观测课程JSON至多提供一种；不提供表示普通对照。文件SHA256与内部策略语义hash不同，均核对。样本评分教师成本与观测条件成本另行登记，不把边际曝光相同当作条件×类别曝光相同。
- 所有路径在checkout外，路径与SHA成对。工作区须干净且完整提交匹配；拒绝Python优化模式。所有声明通过的检查须在授权LXC实际运行。

## 登记、分段运行与完成恢复

下面是参数模板，不是已启动训练。`$p`为授权容器既有Python；完整SHA和所有哈希均从冻结文件/正式登记获得，禁止照抄示例值。

```bash
$p scripts/run_curriculum.py --expected-commit <完整SHA> register \
  --source <外部source.pt> --source-sha256 <权重hash> \
  --metadata <外部source.pt.json> --metadata-sha256 <元数据hash> \
  --development <外部development.json> --development-sha256 <开发包hash> \
  --exposure <外部exposure.json> --exposure-sha256 <曝光文件hash> \
  --sample-policy <外部sample.json> --sample-policy-sha256 <策略文件hash> \
  --lr <登记学习率> --torch-seed <登记种子> --out <新registration.json>

$p scripts/run_curriculum.py --expected-commit <相同完整SHA> run \
  --registration <registration.json> --registration-sha256 <登记文件hash> \
  --to-step <绝对步数> --out <新attempt目录>

$p scripts/run_curriculum.py --expected-commit <相同完整SHA> run \
  --registration <registration.json> --registration-sha256 <登记文件hash> \
  --resume <上一阶段step-N.pt> --resume-sha256 <该权重文件hash> \
  --to-step <绝对目标步数> --out <新的同级attempt目录>
```

观测课程改用 `--observation-policy/--observation-policy-sha256`；普通对照省略两种策略参数。登记执行0次优化，每次恢复重建并核对原合同；`--to-step`是绝对步数，不能回退或超预算。完成后相同步数恢复的 `new_updates=0`，仍核验选中与最后状态，不追加训练。固定Adam、原三头类平衡损失、完整devval fused NLL每步首次最小沿用原合同。

每次调用使用独立目录，禁止覆盖现有目录或与输入、历史目录祖先/后代重叠。每个成功步骤保存不可覆盖checkpoint和哈希；status原子更新。失败保留最后成功前缀及 `resumable=false` 的诊断工件，状态写failed后抛出异常；本入口不自动重试、清理、排队或启动下一实验。

`status=completed`表示此次固定预算执行证据完成，`independent_acceptance=false`仍要求独立审计，不能据此关闭完整研究实验。`query=0`不释放任何最终评价标签。`elapsed_seconds`从输入/恢复核验之后计时，仅含本段执行及收尾；登记、教师评分、恢复检查与总墙钟另记，不充当全流程成本。

## 验收边界

工程检查包括3/9/22类普通/样本/观测三臂、连续与分段精确恢复、完成0、实际曝光/梯度/更新及首次最小选模。3类原24步合成样例曝光不足保留为拒绝负例，独立正例固定48与2+46；9/22类固定24与2+22。仅修订合成测试合同，原运行器、真实预算和门槛不改，不允许运行中自动增步。外部hash错误先于权重反序列化拒绝，篡改、双课程、越预算、脏源码、目录重叠和故障保留另测。独立验收从保存态复算曝光和选中/最后三头混淆与召回。

CPU合成通过不意味着真实HongHu空间可行、真实小类收益、完整课程训练或GPU生产通过。尚未完成的GPU/真实对照按WP01后续及WP02准入单列；本CLI不扩展已冻结数值合同。工程测试与独立审计的实际版本、数量和证据以专属模块报告为准。
