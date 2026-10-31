# B0同预算坐标重训（Issue #33）

三个独立训练条件absolute/relative/zero，均使用B0（保留CNN/Graph与固定融合，关闭交互）、PaviaU、seed2026、原空间块32/gap2、同一初始化/训练采样种子、10000次更新、原损失/优化器及val_loss选模。每100步验证；各自ReduceLROnPlateau使用同一规则但轨迹可能不同。输入通道和参数量不变；zero保留两个零通道，不删除层。relative按每块自身尺寸归一化，既改变位置也改变尺度；动态图值可能变化。默认配置及旧配置hash不变。

SpatialConfig可显式携带coordinate_mode，写入checkpoint的training_config。train_spatial的训练、验证、选模后测试统一使用对应模式；evaluate未指定时自动读取checkpoint，旧checkpoint默认absolute。底层forward_partition仍接受已构造观察，其他低层诊断入口不会自动转换，调用者须使用coordinate_observation；不得把新checkpoint交给只支持原绝对坐标的历史冻结campaign脚本。

入口scripts/run_coordinate_screening.py沿用prepare/benchmark/run及--expected-commit/--data-root/--out-root。benchmark是30步GPU闭环（包含其自有选模后测试，不参与主结果）；主实验顺序预定absolute/relative/zero，串行运行，各自验证选模后只测试一次，不据测试结果改方案。共享初始化写到独立输出目录。resume只跳过已验证完整结果，失败尝试保留并从共同初始状态重试，不续接优化器。

预声明：主比较为验证NLL，OA、逐类、AA及测试为辅助报告；单种子仅初筛，不能声称稳定提升。三条件结束前不改配置；不改默认模型。预计30分钟，所有执行通过ssh rmvision；不操作其他项目、依赖或旧实验目录。资源不足保留失败记录，不通过缩短预算制造可比结论。
