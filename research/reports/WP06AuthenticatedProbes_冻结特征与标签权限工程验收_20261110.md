# WP06：冻结阶段特征和探针标签权限绑定

Issue353 / PR354；主计划§16/§22、实施§19.7。分支 `codex/wp06-authenticated-probes`；tested SHA `4320d40a8a4f15a51d11b7698cbe54b3d221b64e`，基准main `6e198b4181b8a047503f5efb446f0b17089bb06e`。

## 实现和合同

复用原冻结采集、同容量探针和标签家族接口，新增历史阶段特征消费者及CLI。读取任何特征前核对采集版本f090ca302192a065de852c43e07e4f3beaeea6b4、源码文件摘要、原full54冻结/分析/案例、选中步数和全部保护输入；四份阶段产物必须齐备且哈希一致。随后仅释放原家族K10适配与V5验证标签，检查像素索引、类别、场景和互斥；图特征通过原Q映射到相同授权像素。

消费者不加载权重、不执行模型、不读取query、不更新骨干。历史源码记录用于核对旧产物来源，不用于伪称当前模型兼容；当前模型源码已演进，不能绕过旧推理版本检查。原全54区域摘要缺少逐像素特征，不能当作探针输入。仅接入原单例阶段归档，两阶段CNN/图共四层。复用选模验证属于同景传导诊断，不是独立空间确认。

## 实际验收

授权LXC lbx@10.69.37.243:1222，Python `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`；干净 `checkouts/wp06-authenticated-probes01`，CPU单线程/CUDA禁用。

```sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests
python -m unittest test_authenticated_probe_pack test_linear_probe test_target_training -v
python scripts/check_repository.py --expected-commit 4320d40a8a4f15a51d11b7698cbe54b3d221b64e
python /home/lbx/hsi-pioneer271828/runs/wp06-authenticated-probes-audit01.py
```

CPU16/16通过、54.416秒，无跳过；checker566 Python/191 JSON通过。CLI与直接特征行一致，四探针8步与3+5恢复精确；错误采集版本、越权query标记、缺产物、错哈希、后置文件变化、错Q/宽度及已有输出拒绝。未执行GPU或全仓回归。

独立留存合成九类结构边界和一个初始化网络的实际阶段数组，核对51,840特征单元（NumPy映射最大绝对差0）、648混淆矩阵格及72逐类召回；175输入文件保持不变。四探针各8步=32步，恢复回放另32步，完成恢复0；骨干0、query0，10.496214秒。指标核查使用权威PyTorch概率/决策与独立NumPy计数，不能称概率跨后端独立严格复算。

四层验证OA分别15.5556/31.1111/26.6667/31.1111%，零召回类别6/6/6/4，均选8步；仅为合成工程读数，无真实分类或充分拟合结论。结构样例中的54条和selected_step=1500是接口元数据，不是54模型或1500步训练证据。

证据 `runs/wp06-authenticated-probes-evidence01`，receipt SHA256 `907057ab1a639208ac89f1d116fb687da5142aa5ad674911dba769ceb1497f76`；同字节副本 `docs/architecture/wp06-authenticated-probes-engineering-results.json`。外部审计脚本SHA256 `c4dbaab67eb781762a1112aa6fb95069aed576ce22f94bce715d57dd74879294b`，日志 `runs/wp06-authenticated-probes-audit01.log`。所有张量和运行产物checkout外。

## 失败、未测与下一项

首轮8ae46ba49a5fb5e1f67863305179b4afc64bb331：16项中15通过、1错误，49.836秒；旧结构样例遗漏selected_step，正例导出发生KeyError。最终版本补齐结构样例并在解码前明确拒绝缺失或非法选步，未放宽冻结合同。保留 `runs/wp06-authenticated-probes01-tests.log` 和最终02日志。既有探针跨后端并列决策负结果仍保留，不通过本模块抹除。

结论：历史阶段产物与标签权限绑定CPU工程通过；真实模型逐层探针、源初始化对照、空间重复和机制收益仍未完成。实时八卡最高空闲7100MiB，均未达12288MiB启动门槛，不启动真实GPU采集/长训练。下一项沿WP06补受控长程信号与瓶颈干预的工程证据；复用已有诊断，不将WP06整体标为完成。
