# C停机备份与恢复边界

2026-11-26，按主审收到的LXC约8小时后暂时下线通知，临时加入本包。只处理C产物，原文件未删除、移动或覆盖，
未中断任何训练，未新增优化器更新。目标为授权独立rmvision，目录位于checkout外：
`/home/train/hsi-artifact-backups/paper-c-lxc-20261126-01/`，权限700；初查可用空间约965GB。
F盘只保存代码与去身份清单/回执，没有保存模型或原始数据。

## 已实际核验

| 载荷 | 目标位置（相对上述目录） | 源/目标核验 |
|---|---|---|
| 九个DSNet选中模型及独审/失败/恢复记录 | `payload-pilot06/` | 61文件，24305055字节；九模型及每个文件字节/SHA均匹配源清单 |
| 九模型封存归档 | `pilot-memory06.tar.gz` | 21734297字节；SHA256 `2ee8987753bc27862e7e79d68417466a952dc56523ef3ac061a6fda323d04f6e` |
| 本包全部已完成矩阵、稠密概率、独审/配对结果、生成图、三份原始参考PDF和六个DSNet源码文件 | `payload-routing-objects07/restored/` | 1892逻辑文件，538687451字节，全部还原后逐项字节/SHA通过 |
| 精确文件SHA去重对象归档 | `routing-objects-memory07.tar.gz` | 101624360字节；SHA256 `39be56229640270bde2225636c4003c9a8e077eae180d2d65068858f458c6ec9` |
| 独立代码依赖bundle | `paper-c-recovery-dependencies01.bundle` | 6491631字节；SHA256 `ea7ef9c7c0ebf2340230dbd8ccfa7d9f59b763cb4cecbbb7802774f6cfd0ab2c`，两端一致且目标bundle verify通过 |

第二归档按**原文件SHA**去重，1137对象、136765011唯一字节；不是概率舍入、少存重复阶段或改变格式。
1892个原逻辑路径仍全部恢复、保留原SHA，不能把传输体积减少误写为实验结果删减。
原G3九个权重及原统计在rmvision历史目录已有相同SHA，登记去重，不重复制为新训练结果。
独立代码bundle保留WP11 `395caef57fba8ab6a09223538f748179dab552f2`和bank
`49bce8c5fa2d8b79c0e58b2a70693c7b897d4eb7`完整历史。C源码完整bundle与最终审阅包另作小型收尾增量。

源封存采用已完成文件白名单，归档前后重算源文件SHA，未复制仍在写入的阶段。
直接SCP中转、legacy协议和若干交互转发尝试遇连接关闭/超时，失败保留；未删除不完整文件。
最终成功通道是两端独立SSH：源先交互认证，再启动目标接收器，Windows只用1MiB内存缓冲和base64帧转发，
每帧SHA、原子进度、完整归档SHA核完发布，数据不输出到工具/日志，也不落Windows盘。密码仅在SSH提示符交互输入，
没有写入脚本或回执。C预留loopback32294的主机指纹已核为授权LXC指纹；失败的临时转发均已退出。

接收器另在LXC以合成数据通过3项测试：已提交块中断后续传、完成重入不写文件、损坏块/错误帧拒绝覆盖，以及去重对象恢复/损坏拒绝。
两次真实归档都经过整体SHA及逻辑文件复核。备份验证只进行字节/结构校验，未在目标机冒称完成模型推理复现。

## 未备份/尚需确认

`recovery-dependency-index01.json`列1162项冻结依赖身份及487688300字节总大小，其中595项为真实bank/观测数组。
仅记录相对路径、字节和既有SHA，没有为盘点解码数组、打开原GT/query或复制这些数组。
跨机数据权限/所属负责人未明确的，按通知保留缺口；共享WP11权重、原始观测发布也不能因C使用过就擅自移动。
不要在下线后用重新打开原GT、重新选空间划分或重抽支持集来替代原发布字节。

原DSNet one-pass试验保存的是`state`完整参数/BN缓冲及`selected_step`，没有optimizer/RNG续训态。
因此已保全九个选中模型的重载材料，**不能声称具备中途精确续训能力**。所有训练已经完成，不需要继续该one-pass预算。
外部源码备份只含LICENSE/README/model/dataset/demo/utils六文件；未带作者`data/IndianPine.mat`。
作者Git身份校验仍需重建固定SHA的合法源码checkout，不运行作者demo的数据/训练协议。

恢复时先在新隔离目录校验归档和全部文件，再核原C/bank/WP11代码、作者
`e910795073b52373ce0771ac85139a17f250b236`、环境及bank发布身份。
LXC恢复后按原相对路径还原；已有文件同SHA则跳过，异SHA保留并停止，不执行reset/clean或盲目覆盖。
旧合同绑定绝对路径，不能把路径重写后直接称作原合同完成重入；跨机路径迁移需另作显式映射/验收。

## 可复核回执

允许的F盘回执目录：`F:/E2E_VCA/.codex-work/deliveries/paper-c-routing-evidence-20261126/`。
核心文件为`pilot-memory-transfer06.json`、`pilot-verification06.json`、
`routing-objects-transfer07.json`、`routing-objects-verification07.json`与`recovery-dependency-index01.json`。
前三种原始归档/对象/模型文件只在两台授权容器checkout外，未进入Git或Windows交付包。
