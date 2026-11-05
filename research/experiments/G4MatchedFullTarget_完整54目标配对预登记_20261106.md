# G4完整54目标配对实验预登记

对应研究主清单P4 §10.3、模块计划§19，实际实验Issue184。工程Issue182已经验收：生产/测试331d718567b1a9726934f445665999df972392b1；工程receipt在LXC `runs/g4-matched-target-issue182-engineering01/receipt.json`，SHA256 `80a18ffb42d9b7de0c4b1acc20c48e4e37ce04ef398e0c16137c16e5fb766715`。本登记在首个真实更新和任何新query读取前冻结，完整文件hash写入外部registration、launch和checkpoint保护合同。唯一文档写入分支`codex/g4-matched-target-full-study`；实际运行使用独立干净331 checkout，不改已验生产源码。

## 假设、配对和唯一预算

54目标模型＝3目标paviaU/xuzhou/whu-honghu × 模型种子64/512/2048 × scratch/full-finetune × none/capacity_control/x_residual。每例1500更新，总81000，分none18例27000和活动36例54000；每例先2中断后1498恢复，完成后恢复0，不增加更新。3个模型种子是固定支持集上的训练重复，不能当作3次支持集或空间划分重复。

复用已验none9源和活动18源的完整1002步预训练；既有源总27054更新和工程短目标总216更新单列，不计入本次目标81000。源角色/留出目标/模式/种子/预算/图、整批acceptance/独立audit/result/weight/sidecar严格绑定，不用旧native或短初始化器替代。三模式共有fresh参数在9组target×seed中逐元素相同；迁移只复制共享骨干，目标分类头和DomainBN保持fresh，head seed=model seed+1000。

三个模式和两种初始化方法保持相同target数值训练合同：支持种子64，每类K10训练，另V5验证；总目标标签预算15/类，验证不冒充无标签。使用已验G4 label-free全图观察/Q/A，目标无标签输入参与整图构图和统计。epoch1499/每epoch1更新/val_interval30，总1500与51验证事件；保存raw、权威模型/optimizer/scheduler/RNG和最终选中状态。按首次最小fused验证NLL选步，原优化器/ReduceLROnPlateau、三头监督/损失、抽样和训练更新数值路径不改；OrderedValidation用于校准/验证，统一最终query算术。原rtol1e-5/atol1e-6、分类变化0、同路径三次独立重载逐元素一致，不放宽。

## 事前效果判据

主要比较为full-finetune下残差−容量控制，评估残差在相近结构容量控制下的目标效果。先对每个模型种子求三个目标等权macro-scene fused OA配对差，再汇总三个种子的均值及样本SD。收益主张须同时满足：平均宏OA差>0、3/3种子宏OA差>0；每目标三种子平均AA下降不超过1个百分点；HongHu三种子平均OA下降不超过1个百分点；每目标每类三种子平均召回下降不超过2个百分点。所有类均保护，不用总体OA掩盖小类退化。

残差−none的full-finetune比较，以及none/容量/残差各自full-finetune−scratch三个预训练收益比较，分别作为独立命名的次要主张，逐项应用同一组OA/AA/HongHu/逐类保护门槛，不将其中一个通过替换主要比较。scratch下残差−容量、残差−none、容量−none及full-finetune容量−none均描述性报告。补充报告差中差：[(残差−容量)full-finetune−(残差−容量)scratch]，不据此选择新配置或追加预算。

完整报告三头OA/AA/kappa/NLL及逐类计数/召回，主要收益判定使用fused头。先给每目标及模型种子配对表，再给等权场景宏汇总；三种子不宣称统计显著性。任一保护失败时报告局部正差和失败，不宣称净收益。负结果可完成实验；不重跑择优、不更换种子、不按query调参。train/val仅作拟合与选步诊断，不能写成query精度。全部末步选中/末段仍提升时明确未充分收敛，保持1500预算不变。

## 访问边界与完整验收

训练期间只读已释放train/val标签和无标签目标输入，禁止query/test评分和原GT消费者。启动前重新核54身份、同支持/图/观察、9共同fresh、完整源与已验短回执的全部保护文件，冻结后才首更新。工程preflight原目录`runs/g4-matched-target-issue182-preflight01`只读复用，receipt/launch/protected哈希由原文件取得；不能把零更新receipt升级成训练接受。

单卡队列逐例2+1498+完成0，再独立选中态NumPy复算、3重载、模块梯度参与与实际参数改变。全批次须54例81000、324头指标/4320召回/162精确重载，所有已验源/旧结果/输入不变，完成只读恢复0。逐例核51 raw事件完整、first-min选步与逐更新LR重放、权威optimizer/scheduler状态；非选中raw为hash/history/选模/LR检查，不称逐事件独立重算概率。保留三头逐类、首末loss/曲线、选中步、真实模块更新、运行/校准/验证/审计与资源等待成本；不能以有梯度代替精度收益。

Controller接受后，从独立干净331 checkout执行全批次零更新audit，独立核同324/4320/162、源/目标输入与完整训练树不变。先完成拟合/逐类/成本复盘并检查完整receipt，再生成all54冻结边界；边界须绑定全部权重/sidecar/result/launch/acceptance/audit/数据身份，后才允许原GT的final_query释放。所有54在同一个冻结边界下统一ordered query，不混旧native query。只读评价后仍核训练树和输入不变，不做任何模型更新或选择。

## 资源、耗时与失败规则

LXC `/home/lbx/hsi-pioneer271828`、Python `envs/hsi/bin/python`，生产checkout `checkouts/g4-matched-target-issue184-production01`、分支`codex/g4-matched-target-issue184-production01`、完整SHA331d718567b1a9726934f445665999df972392b1。GPU1 RTX4090单队列，每个子操作前free>=12288MiB且util<=60%；不满足只等待，不干扰其他项目。注册冻结GPU1、估时86400秒；启动前估计12–24小时加资源等待，按已完成case的实际吞吐更新预估，不改预算。旧native18目标7746.477秒只用于粗估，不能当新ordered成本。启动前磁盘318GiB可用，旧27目标目录约2GiB，所有旧产物保留。

输出全新`runs/g4-matched-target-issue184-full01`及同级日志；registration和本计划副本在独立`runs/g4-matched-target-issue184-registration01`，不在源码内。独立audit/review/frozen/query均使用新的同级目录。记录启动/退出墙钟，控制器elapsed不含首步前freeze，二者分开。资源等待不算失败；身份/hash/预算/标签/数值/状态不一致或子进程失败时停止并保留原产物，先独立诊断/登记版本，不放宽、不选择性重跑。机器故障后不自动改预算或恢复策略。

本项是历史代号G4的有标签少样本、全图无标签输入参与的传导式适配，不称零样本、纯源无标签TTA或未见传感器泛化。支持种子512/2048、K1/5、独立空间/传感器/波长质量、head-only/light-module及其他两主计划全部必测机制仍未完成，按依赖继续。main保持待评审，不合并。

登记身份校验记录：首个registration01在任何训练/登记文件写入前，断言拦截本文件所抄工程receipt哈希多写一位的错误；旧analysis-plan副本保留。现从已提交工程JSON摘要逐字取得真实hash，登记改用独立registration02，原receipt、数值门槛与生产代码均未改，不重跑预检或模型择优。

