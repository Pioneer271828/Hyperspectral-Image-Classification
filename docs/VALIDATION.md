# 导入验证记录

> 以下为 2026-10-28 的历史导入检查。2026-10-29 起新验收统一在远程训练容器执行；当前基线的实际 SHA、CPU/CUDA 测试和限制见 [基线模块报告](../research/reports/baseline_2026-10-29.md)。

验证日期：2026-10-28。检查在整理后的独立代码副本中执行，未使用原目录的数据、权重或实验结果。

| 检查 | 结果 |
| --- | --- |
| `python -m compileall -q .` | 全部 Python 源文件语法检查通过 |
| `python smoke_test.py` | 12/20 波段共享模型、静态图前向、可学习图前后向通过 |
| `python -m unittest discover -s tests -p "test_interaction_modules.py" -v` | 18 项测试全部通过 |

回归覆盖稀疏/密集对齐、空区域、软分配梯度、门控方向、分块梯度、边权归一化、10 个预设的静态/可学习图、多波段多分类头、检查点保存恢复、数据划分，以及合成数据训练到诊断导出。

本次功能检查使用本机已有 CPU 环境：Python 3.14.2、PyTorch 2.10.0+cpu、NumPy 2.4.1、SciPy 1.17.1、scikit-learn 1.9.0、scikit-image 0.26.0。没有执行从 `requirements.txt` 创建全新环境的安装验证；该文件保留研究依赖中的 NumPy 1.24.3，建议按主 README 使用 Python 3.10/3.11。

未运行真实数据长时间训练、GPU 训练或依赖外部论文图表资产的完整制图流程。通过合成检查不代表已复现精度或验证泛化结论。
