# 原生区域支持演化与多尺度实验

入口为 `scripts/run_native_support.py`。在授权 LXC、一个 CPU 工作进程、FP32、GroupNorm、单数据集和已冻结源码下运行。原 VCA、CNN 两阶段、第一阶段 HGCN、共享 Q 回投与三头使用原生产算子。第二阶段 HGCN 使用所选 COO 支持。局部原权重取本次前向 A0，投影输入取当前 CNN 第一阶段的区域池化与 HGCN 第一阶段特征。选边使用首更新前同源权重固定的各开发块光谱和图描述。仅两个 128→16 无偏置评分投影是新增参数。其初始化使用独立种子 7321。

### 登记和恢复

将输出目录设在 checkout 外。核对分支、完整 SHA 和干净工作树后，设置单线程并隐藏 CUDA。按登记中各个 `case` 和 `arm` 依次执行，正式矩阵一次运行。

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/lbx/hsi-pioneer271828/envs/hsi/bin/python
SHA=$(git rev-parse HEAD)
ROOT=/home/lbx/hsi-pioneer271828/runs/<新目录>
test -z "$(git status --porcelain --untracked-files=all)"
"$PY" scripts/check_repository.py --expected-commit "$SHA"
cli=("$PY" scripts/run_native_support.py --expected-commit "$SHA")
pin() { sha256sum "$1" | cut -d' ' -f1; }
"${cli[@]}" register --root "$ROOT"
R="$ROOT/registration.json"
"${cli[@]}" preflight --registration "$R" "$(pin "$R")"
C=wp08-nine8-64
A=joint
"${cli[@]}" stage --registration "$R" "$(pin "$R")" --case "$C" --arm "$A" --phase full
"${cli[@]}" stage --registration "$R" "$(pin "$R")" --case "$C" --arm "$A" --phase prefix
P="$ROOT/matrix/$C/attempts/$A/prefix/status.json"
"${cli[@]}" stage --registration "$R" "$(pin "$R")" --case "$C" --arm "$A" --phase resume --prior "$P" "$(pin "$P")"
P="$ROOT/matrix/$C/attempts/$A/resume/status.json"
"${cli[@]}" stage --registration "$R" "$(pin "$R")" --case "$C" --arm "$A" --phase completed --prior "$P" "$(pin "$P")"
```

每阶段仅认领一次，完成态重载执行 0 更新。失败状态保留在 `attempts`，阶段拒绝重复消费。WP08 的 `fixed` 需要同夹具同种子的 `joint` 完成其四个阶段，登记的固定 donor 绑定连续末权重包和完成回执。固定图只继承支持，重新训练从共同源权重与独立固定 scorer 初始化开始。阶段命令示例只演示一个臂，`freeze` 需要完整 75 个臂。

```bash
"${cli[@]}" freeze --registration "$R" "$(pin "$R")"
F="$ROOT/freeze.json"
"${cli[@]}" audit --registration "$R" "$(pin "$R")" --freeze "$F" "$(pin "$F")"
"${cli[@]}" cost --registration "$R" "$(pin "$R")" --freeze "$F" "$(pin "$F")"
```

若审计或成本代码在单独验收提交上，`--expected-commit` 使用该 checkout 的完整 SHA，另传 `--producer-commit` 指明已冻结训练提交。训练阶段、预检和冻结要求原登记的提交。输出的 `freeze.json` 包含选中与末态的版本化模型包。加载时先从已绑定清单和 SHA 校验源、开发影像、上下文及模型包，再用 `inference.native_support_checkpoint.load_endpoint(..., device="cpu")`。用 `predict_registered_tile` 在匹配的 devval 块获取三个概率张量。旧 off 使用原架构配置，加载时返回隔离副本。直接构造只接受双坐标通道、GroupNorm、半径 1 的原学习图。实验版通过显式封装加载，CPU 入口拒绝 CUDA。

### 开发角色与选模

登记的合成训练块必须覆盖两块全部支持和每类曝光，首步前检查。候选限制在单块同分区，最多 256 区域，描述每路最多 32 维，候选每行最多 32，预算含自环最多 16。全局优化步驱动支持状态，某块跳过数步时按原序补足刷新和随机流。活动图严格使用当前预算，历史快照以其上下文、来源、随机状态及历史预算认证。验证使用每块的第 0 步支持，直到末态和恢复均保持同一协议。每更新一轮完整开发验证，按最早最低 fused NLL 选步。读取本地训练阶段回执和 `audit.json` 时，`pool_sha256` 变化表示候选池变化，`support_sha256` 或与同一块上次访问的支持比较才能判定实际换边。原始遥测记录发送与接收度和网格距离，只有边数一致时报告边数对照。审计的 −2pp 逐类召回比较沿用 #411 的显示阈值，在本批仅作合成读数。OA、AA 另列，小类错分身份与候选准入由后续真实协议决定。

`real-probe` 需要固定的 #411 源登记、#407 开发 manifest 和 WP02 curator 回执，使用 `--prior-wp07`、`--prior-wp08`、`--curator` 逐项传路径和 SHA。真实探针每个合规场景最多读取一块 devtrain 和一块 devval，禁止 query、rawGT 和未释放块，真实优化为零。其输出与原生合成指标通过 `supplement` 分别登记，既有 WP12 表逐字节保留。候选张量合同允许验证独立的小图，真实权限由固定开发发布核验决定。课程、MiTA、原图路由、实验多尺度与支持演化的联合训练未定义。

### 多尺度与成本

WP07 半径 2 使用按配置生成的非持久偏移。原有 9 个偏移是半径 2 的 25 个偏移的子集，新增 16 个偏移由几何生成。图查询、键和共享卷积使用同一来源的张量。可信 nm、FWHM、GSD 和覆盖率来自明确标记的合成条件。缺失物理条件时 metadata 回退 mean，joint 回退 content。波段置换时同步元数据，编码先稳定排序；重复原始 band ID 和全部无效波段拒绝。FWHM 缺失有显式标记。

`cost.json` 含配对的热样本与独立进程冷启动，冷启动含 Python、导入和进程开销，系统文件缓存未刷新。已驻留的源 bank 与模型称热，Q/A0 和当前支持权重仍逐次构建。固定验证块和开发训练块的支持演化回放有各自的原始样本。报告一次性 bank、donor 和摊销成本，区分网络段计时、搜索、保存恢复、反向和端到端。分项嵌套计时不得直接相加。CPU 开发块测量仅适用于登记的块、设备和前向协议。
