# G4完整配对目标协议工程范围

对应研究主清单P4/§10.3、模块计划§19，Issue182。前置172/178完整活动18源/none9源、174活动12短目标、180 none6短目标及独立验收已经交付。复用既有明确full1500 runtime/源loader，不将旧native query混入新比较，也不把short入口或零更新预检称作完整训练。

唯一写入`F:/E2E_VCA_worktrees/g4-matched-full-target`，分支`codex/g4-matched-full-target`，基于已验adapter交付f240a7e116f16cb0a431813999e86cc4a04e75d7，root唯一负责人。仅新增独立study/launch/worker/campaign/audit/tests、research manifest和状态报告；原模型、loss、source/target训练、校准器、旧入口及冻结configs不改。

工程目标是统一54条件：3目标×3模型种子64/512/2048×从头/全微调×none/容量/残差，完整计划预算none18×1500=27000、活动36×1500=54000，共81000。支持集seed64，K10另V5，无标签目标图；headseed=modelseed+1000。明确full epochs1499/每epoch1/val30/51验证事件，原优化器/scheduler/首次最小fused NLL选模，校准/验证ordered及严格原rtol1e-5/atol1e-6/分类0/精确重载均保持。

逐角色设置保留架构/身份差异后，其余训练与数值合同必须完全相同；配对输入/标签/图不能仅凭文件名推断。首更新前核对全部源acceptance/audit/result/weight/sidecar/seed/mode/graph/预算，并证明三模式共有fresh初始化相同，迁移目标head/DomainBN与对应fresh相同。已验short receipts和其producer文件不变证明先行。全部54 launch须冻结；零更新预检phase独立、execution_enabled=false，不能当完整实验预算或启动凭证。

验收覆盖CPU/GPU/全量、54角色与各种越权/身份/seed/模式/hash/预算/标签/图负例，有界fixture完整1500调度/51选模事件/恢复/完成0，实际54零更新预检和文件输入保护。独立完整接受、逐类/拟合/成本复盘与全冻结评价合同仍需新工程，不把预检工具存在当完成。每卡单队列、启动free>=12288MiB/util<=60%，不动其他项目。

本Issue是工程任务，真实54完整实验另登记Issue/预算/停止条件/资源计划，宣布预计时长后依用户既有授权执行；全54与独立验收完成、模型冻结后才统一query。head-only/轻量模块、支持集/K/空间/传感器与两主计划其他必测机制保留；main另待评审。
