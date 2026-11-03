# 基线来源与范围

本模块对应 [Issue #1](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/1)，依据研究主清单 P0 与模块交付顺序 0。来源核对日期：2026-10-29。

当前研究来源是 `F:/E2E_VCA`，其中 Git `master` 的历史 HEAD 为 `dda3a3eba15949d58481357c28965d95e3bfd053`，旧 `origin` 指向 Gitee。该历史 HEAD 不含当前所有研究代码，不能拿它代替工作目录源码。原目录存在已修改/已暂存/未跟踪文件，均保留。

目标 GitHub 的默认分支为 `main`，核对基准为 `018fe70fcbd3e902cd5fe0a05296bdd1d38d5bfb`。逐一比较目标仓库已有 Python 文件与当前工作目录的同路径文件；`smoke_test.py` 的本地来源为 `vca_dual_branch_runnable/smoke_test.py`。忽略 CRLF/LF 后均一致，因此直接从目标 GitHub 基准建立隔离工作树，不再重复导入。

[源码指纹](baseline-source.json) 保存原始 Python 文件的 LF 归一化 SHA-256 和来源路径；本模块扩展后的 `smoke_test.py` 由 Git 提交追踪，单独注明变化，不宣称与原文件相同。新增验收脚本和文档同样由完整 Git SHA 定位。之后修改模型模块时应保留本基线标签，通过 Git 比较，而非更新旧基线指纹以掩盖差异。

本模块保留已有模型、训练入口、静态图默认值、融合比例、损失和选模逻辑。新增内容为远程验收入口、可选 CUDA 冒烟覆盖、导航、协作边界和版本记录。暂不进行 `src/hsi` 迁移；C0 再扩展配置与运行身份，C1 再固化数据协议。

未导入原目录数据、权重、图像、日志、会议材料或无关脚本；研究目录只选入三份主计划，历史状态以最新执行约定为准。旧论文 README 保存在 [原始说明](../SAHGCN.original.md)，未发现可直接沿用的授权文件，暂不替原作者声明许可证。文件来源登记不等于授权已核实。

原仓库 [导入验证](../VALIDATION.md) 中的本机 CPU 检查是历史记录。此后所有新检查只在远程训练容器执行，当前证据见 [模块报告](../../research/reports/baseline_2026-10-29.md)。
