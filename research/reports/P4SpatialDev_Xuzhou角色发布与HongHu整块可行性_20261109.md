# P4 Xuzhou 空间开发角色发布与 HongHu 整块可行性

对应主清单 P4 §10.3 与实施计划 C1 §4、§19。前次三景固定角色协议中 Xuzhou 验证缺类 3，HongHu 验证缺类 15、18。保留原回执，并登记独立开发协议。复用已验收 `loadData.spatial_development.make_development`，只在原训练块内贪心整块划分，最大化当前验证缺额的有效补足，等分按最小块 ID，训练每类至少保留 5 个候选。验证每类 5 个，种子 3031。支持种子 512/2048，同种子 K1 嵌套于 K5。原测试块几何保持一致。

开发标签准备进程解码原始 MAT 标签数组并枚举原训练块候选。原验证与测试区标签值不取样、不计数、不发布。该开发分割使用原训练区类别覆盖，报告按 oracle 辅助开发协议解释。影像/运行侧另有访问合同。所有文件在 LXC checkout 外，优化更新 0。

Xuzhou 划出 0、6、79、88、109 五个开发验证块。训练与验证区的最低类别候选数分别为 1,114、7，四个支持/K 组合全部足额。独立核验核对角色坐标、计数、K 嵌套、固定验证集、互斥和测试几何，未解码原始 GT。真实空间训练与恢复尚待完成。

HongHu 贪心失败后，另行冻结同预算整块可行性诊断。SciPy/HiGHS 整数规划返回不可行，并找到直接反证。第 21 类在 270 个原训练块中的 47 个候选全部集中于块 65。该块无论分给开发训练还是验证，另一侧第 21 类都为零，整块条件下无法同时满足每侧至少 5 个。未发布 HongHu 替代角色或支持/验证样本。

授权 LXC `/home/lbx/hsi-pioneer271828/checkouts/p4-g2-family-contract01`，`codex/p4-g2-family-contract`，tested SHA `d656caddf345f2cd2a87adf7393b8b27691d54ce`。从该 checkout 使用 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python` 在单线程 CPU 上运行下列驱动。数据和产物根 `/home/lbx/hsi-pioneer271828/runs/`。

| 驱动及回执 | SHA256 |
| --- | --- |
| `p4-spatial-dev-curator01.py` → `p4-spatial-dev-curator01/receipt.json` | `e012bce1197154f6b6e81978758ae4d8ee3f11fdc374813dd7d317b74b0e789e` |
| `p4-spatial-dev-audit01.py` → `p4-spatial-dev-curator01/independent01.json` | `cbac31a0c55b934dbfd6716841e1f99e8a96066534fe3991aa9bfcb039fdf97e` |
| `p4-honghu-dev-feasibility01.py` → `p4-honghu-dev-feasibility01/receipt.json` | `8bfca29627eac24018fd15d784df972fc6efa9c565602e1c0c9b743fbbe6907a` |

已转入 Xuzhou 四组开发空间运行接入。零更新预检用 73 个开发训练块拟合波段统计，置零其余 74,112 个像素后统计逐位一致。PaviaU/HongHu 源权重有 29 个共享张量形状和 dtype 与 Xuzhou 空间模型匹配，6 个学习图张量需要新初始化。首次预检的输入回执哈希转录有误，在读取影像和构建模型之前被拒绝。改用原文件哈希后通过，保留首次日志。`p4-xuzhou-dev-runtime-preflight02/receipt.json` SHA256 `46268ea5aa32b4b94deab6da9fb6c4308f10294a855ca36d987b8bdf162d68fe`。下一阶段接入角色训练、验证和断点恢复。HongHu 保留原固定协议与本整块开发诊断的缺口，任何新几何或类别范围需另行登记。
