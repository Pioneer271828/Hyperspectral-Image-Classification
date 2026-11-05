# G4 none同协议完整源训练与独立验收

批次日期2026-11-05；完整交付记录于2026-11-06（北京时间）。

对应研究主清单P4/§10.3与模块计划§19，Issue178/PR179。用途是为完整M4源→目标比较提供新的同数值协议none源控制；旧none批次在Issue176的完整轨迹兼容门槛失败，原产物和效果结论保留。本项**完整源训练、数值/身份验收与复盘通过**，不包含目标准确率、迁移增益或充分收敛结论。

## 实现与固定合同

复用SourcePretrainRuntime、selective source context与已验OrderedValidation，只在显式none v2的validation_snapshot使用固定行序FP32求和。None模型结构、三头损失、源抽样、前端DomainBN校准、源训练标签图及实际训练更新不变；旧默认native、旧配置与冻结目录不变。新协议JSON在research/experiments独立绑定SHA，不通过改冻结规则绕检查。

全部3留出目标×3模型种子64/512/2048的9例在首更新前冻结：源码/研究JSON/launch、另两源context及输入hash、Q/A、fresh初始完整状态与两活动M4模式的共有初始参数逐元素一致。worker实际构建再次核对初始状态，provenance/checkpoint绑定launch。只读另两个源的payload，留出目标影像/统计/标签/图不加载，群组元数据可见；query0。

每模型1002更新，共9018；原500 epoch、每epoch有放回抽样2源、val_interval10、51次校准验证，first-max宏fused源验证OA选模，原ReduceLROnPlateau。每例2步中断后恢复1000步，总预算不重复；完成后恢复0步且文件不变。保留原rtol1e-5/atol1e-6、分类变化0；独立三次重载另要求逐元素一致。

## 实际环境、代码与工程测试

- **完整生产/测试SHA：**`52f32978419cbeda216a90bfb0ed76c393ef4f5d`。LXC分支`codex/g4-none-source-campaign01`，checkout `/home/lbx/hsi-pioneer271828/checkouts/g4-none-source-campaign01`，解释器`/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`、GPU1单队列。rmvision同名checkout/分支，解释器`/home/rmvision/venvs/hsi-baseline/bin/python`。CPUcontroller2=0.036s、GPU5=10.094s、全量534=174.773s、checker331 Python/76 JSON通过；LXC CPU2=0.074s/checker通过。
- **真实短测producer：**`a44ce12754dd380e6063aa0a4f1138f1cc5335da`。rmvision CPU5=4.348s、GPU5=9.429s、全量532=171.468s；LXC GPU5=10.232s。三例各8步共24，36独立指标480召回、9精确重载、3完成0恢复和3163旧文件输入保护通过，283.577s。完整controller增加文件而不改短测producer已有源码，真实短回执与全部产物hash是启动前置门槛。
- **独立审计SHA：**`6c8a93eb2be0271c37782d254bcc2a8ada55fc20`。LXC分支`codex/g4-none-source-audit01`，checkout `checkouts/g4-none-source-audit01`；rmvision `/home/rmvision/g4-none-source-audit01`同名独立分支。CPU3=0.032/0.072s、全量535=172.543s、checker333 Python/77 JSON通过。
- **曲线导出功能版：**`196242a70fb82a9095436827b03b462481906b8d`，rmvision CPU2=1.373s、全量537=174.176s；只改图例布局的实际导出SHA `e3ef407afb8d7f25841186dff7ced1b199fd535f`，rmvision/LXC独立`g4-none-source-report02`分支和checkout，CPU2=1.348/1.672s、checker335 Python/77 JSON/CLI帮助通过。537对应196功能版，未冒称e3再次全量。

专项命令分别为`python -m unittest test_g4_none_source_launch test_g4_none_ordered_source -v`、`python -m unittest test_none_source_campaign -v`、`python -m unittest test_none_source_full_audit test_none_source_campaign -v`、`python -m unittest test_none_source_report -v`；全量为`python -m unittest discover -s tests -v`。GPU专项使用`G4_NONE_DEVICE=cuda`与单卡CUDA_VISIBLE_DEVICES；OMP/OPENBLAS/MKL线程均1，PYTHONDONTWRITEBYTECODE=1。执行前核对Linux容器、完整SHA及干净工作树；未注册CI，不声称自动CI运行。最终交付检查另记录在版本提交与PR。

## 完整批次与独立回执

原始批次`/home/lbx/hsi-pioneer271828/runs/g4-none-source-issue178-full01`。9/9模型均完成1002、原严格收尾和完成0恢复通过，合计9018。controller耗时**7111.150608秒，1小时58分31秒**，包含逐例资源等待、加载/恢复与复核，不是受控纯训练速度；首步前准备与曲线导出未单独计时，不补造耗时。

控制器逐操作门槛free>=12288MiB/util<=60%，GPU1单队列；利用率超限时只等待，不干预其他项目。首次预估1–2小时，实际落在该范围内。训练期间不改源码、模型、损失、标签/图、预算或选模，不重复训练择优。

|证据|SHA256|
|---|---|
|full01/launch.json|`6e1b979d4c9375f98e5fbcf4871656bb16e59812140746ee3de06a5b101b1a21`|
|full01/training-acceptance.json|`60f70fe924479c20920b9668dc338a6d4faf1446a108e4637baf794566ebd481`|
|full01/protected-files.json|`13b71aa2f5c3418047fb4ccdf75bbe72570cba4ad2453141ea8361181e84b613`|
|independent01/receipt.json|`6ff38d1915c53286c1800832234afd3991cc879f029317c6dc01a70e39d4aa0b`|
|review01/summary.json|`83e3c09cc1fcd47eb10b02e3ffe790f4fe687a5dedcaee6d4fabe3807c5b9abe`|

上述independent01/review01均指同一runs根下的`g4-none-source-issue178-independent01`与`g4-none-source-issue178-review01`。独立审计通过108项头指标、1440项逐类召回复算、27次精确重载，以及初始状态/源context/图、真实更新/曝光、权重/optimizer/scheduler状态、first-max选模与结果文件对核验。**4270个受保护文件与输入不变，新增优化器更新0/query0，110.422900秒**。训练阶段另有3230项保护，范围重叠，不能相加。

实际独立命令（上述LXC解释器、线程环境，审计GPU1）：

```bash
cd /home/lbx/hsi-pioneer271828/checkouts/g4-none-source-audit01
CUDA_VISIBLE_DEVICES=1 /home/lbx/hsi-pioneer271828/envs/hsi/bin/python \
 research/reproduction/m4g4b/none_source_full_audit.py \
 --expected-commit 6c8a93eb2be0271c37782d254bcc2a8ada55fc20 \
 --producer 52f32978419cbeda216a90bfb0ed76c393ef4f5d \
 --run /home/lbx/hsi-pioneer271828/runs/g4-none-source-issue178-full01 \
 --launch-sha256 6e1b979d4c9375f98e5fbcf4871656bb16e59812140746ee3de06a5b101b1a21 \
 --acceptance-sha256 60f70fe924479c20920b9668dc338a6d4faf1446a108e4637baf794566ebd481 \
 --protected-sha256 13b71aa2f5c3418047fb4ccdf75bbe72570cba4ad2453141ea8361181e84b613 \
 --out /home/lbx/hsi-pioneer271828/runs/g4-none-source-issue178-independent01
```

## 逐例拟合、选模与成本

OA为另两个源的宏fused验证OA，不是左列留出目标的精度。Loss为每例首/末30步总监督loss均值；末段增益是最后50 epoch源验证OA变化（pp）。耗时含controller逐例复核与等待。

|留出目标/种子|选中步|源验证OA|首30→末30 loss|末段OA增益|秒|
|---|---:|---:|---|---:|---:|
|PaviaU/64|1002|95.1366%|7.9558→0.4813|+0.6677|960.631|
|PaviaU/512|1002|95.5114%|7.6691→0.5249|+0.6677|1006.203|
|PaviaU/2048|1002|94.0685%|7.9639→0.6452|+0.7961|960.503|
|Xuzhou/64|1002|94.6412%|7.9576→0.5396|+1.0244|816.721|
|Xuzhou/512|1002|94.4671%|7.6715→0.5399|+0.5907|818.252|
|Xuzhou/2048|1002|93.6196%|8.0393→0.6662|+1.1243|849.504|
|HongHu/64|1002|98.5967%|6.3409→0.1614|+0.1437|561.244|
|HongHu/512|982|98.2837%|6.4783→0.1598|+0.4566|566.912|
|HongHu/2048|1002|97.8271%|6.4877→0.1545|+0.4566|558.837|

9例均完成1002次图支路正梯度和实际更新；adapter变化元素10394–10465、CNN20096、graph17088。None不存在M4残差参数，不虚构M4梯度。参与/参数变化证明工程运行，不能当作精度有效性。

8/9选中1002，HongHu/512首次最大值在982。源HongHu的部分训练OA仍约90–94%、末段验证上升，尚不能确认充分收敛；按原配对预算完成，不据此临时延长训练。训练—验证fused OA差在本批次不超过约3.20pp，不能单凭总loss宣称没有过拟合；不同源难度和头表现分别保留。实际LR曲线沿原调度，没有事后改学习率。

## 小类、三头与源验证参照

所有已观察源类的三个验证头均没有0召回；fused全批次最小召回**0.7190**（PaviaU留出/2048的HongHu源），graph最小**0.5000**（HongHu留出/2048的PaviaU源），CNN最小约**0.6694**。总体OA不能掩盖图头个别小类偏弱。小类保护为完整记录与独立一致性核验，不把没有预登记的阈值当成通过/失败新判据。

三模型种子均值±样本标准差；下表三列均为两个源的验证结果，活动源来自已验172批次、同held/seed/源输入/初始化与1002预算，不能直接解释成目标迁移收益。

|留出目标|新none源OA|已验容量源OA|已验残差源OA|
|---|---:|---:|---:|
|PaviaU|94.9055±0.7487%|95.5098±0.2921%|96.0732±0.6041%|
|Xuzhou|94.2426±0.5465%|94.8980±0.6883%|95.3726±0.5706%|
|HongHu|98.2358±0.3870%|98.4359±0.1566%|98.1118±0.2284%|

详细逐源/逐头OA、AA、NLL、各类support与recall、拟合差和文件对hash见`docs/architecture/g4-none-source-full-results.json`。9张真实六面板图与9份CSV在远程review01；只读导出本地副本`F:/E2E_VCA_worktrees/issue178-review01-export`。已检查三方向代表图（包括唯一982选模例），图例和类别柱清晰；合成预览与真实结果分开。图片/CSV/原始权重日志留checkout外，不入Git。

实际CPU导出使用e3 checkout，`none_source_report.py --expected-commit e3ef407afb8d7f25841186dff7ced1b199fd535f --run <上述full01> --launch-sha256 <上述launch hash> --acceptance-sha256 <上述acceptance hash> --audit <上述independent receipt> --audit-sha256 6ff38d1915c53286c1800832234afd3991cc879f029317c6dc01a70e39d4aa0b --out <上述review01>`。独立回执及全部原文件hash在导出前后核验，源验证之外的数据不读取。

## 保留失败、验收边界与下一项

旧native none兼容审计负结果与旧query结论保留；旧M4 v1/native目标失败不改。首43候选把研究JSON放进冻结configs路径被checker拒绝；e8移到research而未改变内容或冻结规则。CLI两次合成fixture错误在真实训练前修复，旧日志留存。真实短测及本次完整批次没有重跑挑选或静默改门槛。

本模块验收结论：同协议9例完整none源控制、严格完成恢复、独立复算与曲线/小类复盘通过，可进入**none完整源→目标绑定与短接入**。源验证结果不证明目标迁移收益、边界收益、严格未见传感器或支持集/空间独立泛化；模型种子不等于支持集种子。

下一项按P4/§19独立Issue/分支建立新none完整源协议绑定，继承fresh目标头/DomainBN、只复制共享参数、K10另V5/support64/无标签目标图，先6例×12=72更新真实短接入及CPU/GPU/回归/独立复核，再预登记none18×1500与活动36×1500完整目标比较。全部比较模型冻结后才同协议query评价，不混旧native分数。后续独立支持集512/2048、K1/5、空间/传感器与其余机制仍按两主计划保留；main另待用户评审。
