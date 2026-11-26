# C 实验证据包

此目录由 C 唯一写入。基准 `a339ce8d7e5f0a600363b1a8826b96d8a16ae794`，分支 `codex/paper1-c-baselines-20261126`。不写 A/B 文件、主状态表或 main。确认设计见 [冻结草案](confirmation-draft.json)。该草案不签发新标签或长训练权限。

M4-S2 #135 的 1%/0.5% 三景联合实验不是 G4 的 K10/V5；前者是既见场景传导且测试已被查看。#141 原概率复现失败与 #143 训练后、测试前的决策稳定性修订分别保留。开发银行是 support512/K5、validation3031/V5，不能凭三个模型种子改称空间重复。

CEGCN 是首要结构参照：超像素编码/解码和 CNN/GCN 互补与本问题直接相关，但公开固定树没有确认代码许可。较新的 DCG-Net 在论文中直接链接作者仓库；其代码 ZIP 实际存在，不能沿用仓库简介声称“无代码”。本轮在 LXC 只枚举 ZIP 和读取源码条目，未解码其中数据/GT、未运行附带程序；根及 ZIP 成员未见许可文件，文件级许可仍待核查。不能把论文开放许可直接套在代码上。

DSNet 的作者固定提交 `e910795073b52373ce0771ac85139a17f250b236` 有根 GPL-3.0。它融合像素类别特征与解混丰度，不使用 Q/A，也不构造区域内像素残差；作为可执行分解特征比较候选，不冒称最接近的区域图方法。`adapter.py` 只加载外置固定作者模型并处理输入/输出，不复制作者文件、训练器或划分，不改变项目模型。原始模型与此适配的数值配对不等于本方法优于外部基线。

所有代码与测试仅在授权 LXC 执行。原作者 `demo.py` 有随机划分、CUDA 硬编码以及训练期间测试读出；不将它作为项目数据入口。原始 `DSNet.forward` 是本包执行的作者入口，完整 `demo.py` 复现单列为未运行。作者模型丰度归一化和 SAD 未加 epsilon/clamp，遇到奇异输入直接保留失败。原模型构造时的 BN 状态变化也完整继承，不私自重置。

短测先登记真实块清单、内容哈希、点曝光、步数和选择事件。适配器用块内 7×7 边缘复制补齐；与整景随机 patch 的上下文不同。模型原生输出非线性重建、线性解码与分类 logits，仅分类头产生指标。图成本为 N/A；patch 物化、模型构造、重建分支、验证选模和保存重载的时间均需记录。GPU1/4 当前归属未核清前不使用。失败和不合格分组保留。

参考：[CEGCN 作者库](https://github.com/qichaoliu/CNN_Enhanced_GCN/tree/e5bc8a007a7179d2092aa1cc1c0993ac1a171d4c)、[CEGCN 原论文](https://doi.org/10.1109/TGRS.2020.3037361)、[DCG-Net 原论文](https://www.mdpi.com/2079-9292/13/16/3271)、[DCG-Net 固定库](https://github.com/229Kevin/DCG-Net/tree/2eb4e7b9fef602c2899cb2f6d596b95655b5bf3f)、[DSNet 原论文 §II](https://arxiv.org/html/2412.03893v1)、[DSNet 固定模型](https://github.com/hanzhu97702/DSNet/blob/e910795073b52373ce0771ac85139a17f250b236/model.py)、[DSNet 许可](https://github.com/hanzhu97702/DSNet/blob/e910795073b52373ce0771ac85139a17f250b236/LICENSE)。CEGCN IEEE 全文入口本轮被机器人验证阻挡，不声称已完成全文重读。
