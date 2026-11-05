# G4 none同协议完整源对照预登记

对应研究主清单P4与模块计划§19、Issue178/PR179。兼容审计176已按事前精确分类计数门槛拒绝复用旧none；旧模型与query结果保留。本项建立新的同协议源对照，不能用短工程权重或旧native结果代替。

## 首步前的门槛与身份

独立short/full worker、launch与controller代码提交为`52f32978419cbeda216a90bfb0ed76c393ef4f5d`（运行现场还须核对完整SHA和干净工作树）。先完成实际三留出方向seed64、各8步共24更新的短测及CPU/GPU/全量回归。真实短测生产`a44ce12754dd380e6063aa0a4f1138f1cc5335da`，短测receipt完整通过且全部产物原hash保持后，controller才允许继续。短测producer所有源码在完整候选中须逐字节保持，只允许增加controller/测试。

数据目录只取已验活动源launch，固定其SHA256为`0ab7b356b0dd9c63aee67fb2e6b7aca147c63ec4fbb242e6d9c0089bbc63e260`、producer为`01c29c82e92ce0a529749c53962202e9bf0dc5fe`。准备manifest沿base97_training的固定hash。新研究JSON独立绑定文件SHA、完整提交与干净工作树；不修改configs冻结路径或旧清单。

在首个更新前完成全部9例launch：每例settings、另两源selective source_context/原输入hash、Q/A、fresh完整初始状态、共有参数与容量/残差两个活动源初始逐元素一致。worker重复核验实际输入/图/初始构建；launch hash写入运行provenance和checkpoint。源模型只使用另两个源的payload，留出目标影像/统计/标签/图不加载；群组元数据可见不等同payload访问。冻结旧none源、活动源、短测与准备输入，结束核验全部不变。

## 唯一预算与判据

3留出目标paviaU/xuzhou/whu-honghu × 模型种子64/512/2048 = 9模型。每模型1002更新，共9018；原500 epoch、每epoch有放回抽样2源、val_interval10、原源front-end DomainBN校准、first-max macro fused源验证OA选模、原ReduceLROnPlateau设置。None结构、监督/损失、源图、训练更新数值路径和种子不变。仅validation_snapshot使用与活动源v2相同OrderedValidation。原默认native入口保留。

固定2步中断后恢复其余1000步，总1002；完成后恢复0更新，所有完成产物不变。原收尾rtol1e-5/atol1e-6且分类变化0；独立3重载必须逐元素精确。源抽样允许单例某源在短窗口曝光不同，不改变原schedule。完整trace/history、模型/optimizer/scheduler/RNG、raw验证状态、选中状态和概率均保留。任何失败停止并保留，不改门槛、不重复训练择优。

单GPU单队列，每个子操作前空闲至少12288MiB、util<=60%，不满足等待；不干扰其他项目。所有运行产物在独立checkout之外。启动前告知完整预算和预计耗时，按实际吞吐更新估计；旧native成本只用于估计，不当作新实验成本。

## 完成与复盘

逐例独立NumPy复算train/val三头指标和逐类召回；九模型共108头指标/1440召回、27次独立精确重载。核验first-max选中步、所有真实更新、图梯度与adapter/CNN/graph实际参数变化；复盘首末loss、验证曲线、三头、小类最小召回、选中步和成本，末步选中不称充分收敛。所有旧模型及输入保持不变。Controller acceptance只是批次验收，仍需从独立干净checkout执行零更新全批次audit并读完整receipt，才交付完整源模块。

本项不读query、不产生目标准确率，也不对none优于/劣于M4作效果结论。完整源验收后另登记none完整源到目标18×1500和活动目标36×1500，同协议/同支持集/同预算/同标签权限及fresh头/DomainBN、无标签目标图；全部冻结后再统一query评价。独立支持/空间、K敏感性及其余必测机制继续按两份主计划推进。
