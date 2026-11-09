# WP09：开放集完整已知区域预处理工程验收

Issue359 / PR360；主计划§13、实施§19.10。分支 `codex/wp09-known-context`，实际tested SHA `5d12d81ed77ab6a85d9bbabfdcbfdf0db57c6477`。

## 完成内容和协议边界

新增外部登记CLI及独立输入准备模块，沿用现有E4c评分器和三头观察载荷。源训练与源校准矩形要求完整、合法的已知类标注；未知、未登记或0号未标注即拒绝，包括不在监督索引中的像素。按整个源输入区域拟合逐波段min/max，校准与开发评价只应用该统计；保留全局坐标、原物理波段和像素索引，检查三个完整区域空间不重叠。评价标签只保留路径/hash，不读取；不扫描整景GT、不搜索“更容易”的已知块、不将未知像素置零后冒充排除。

这是一个显式严格子协议，不自动适用所有开放集设置。完整源/校准区域标注是额外前提与成本，不能算少样本K预算内的免费标签。首版要求三角色有相同的有序物理波段；不伪造传感器或重采样。复用已有公式和作者核查记录，未引入新研究机制、复制上游代码或改变模型/训练器。原E4c入口保留，历史训练身份仍标记未验证。

## 实际验收

授权LXC `lbx@10.69.37.243:1222`，Python3.10.20，解释器 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，干净 `checkouts/wp09-known-context01`，CPU单线程、CUDA禁用。

```sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests
python scripts/check_repository.py --expected-commit 5d12d81ed77ab6a85d9bbabfdcbfdf0db57c6477
python -m unittest test_known_context_preparation test_open_set_producer test_registered_open_set test_open_set -v
python /home/lbx/hsi-pioneer271828/runs/wp09-known-context-audit02.py
```

CPU31/31通过，无跳过，32.248秒；checker573 Python/197 JSON通过。包含真实CLI、源统计独立于校准/评价极值、常量波段、全局坐标、未知/0在非监督位置泄漏、缺完整标注、角色/空间/波段/语义/文件身份错误拒绝、评价标签未释放、重复输入输出隔离以及现有评分器/模型生产/冻结评价回归。未执行GPU或全仓回归；检查时8卡均不足12288MiB空闲门槛，未启动训练。

独立核对三个种子64/512/2048的合成场景，1512个归一化值、432个坐标值与独立NumPy逐波段/逐像素计算精确一致；每场景额外完整标注48像素，不能混入已有K预算。三套输出接入三个初始化M8模型，完成生产→冻结→开发标签释放→27份开放集评分报告，2.730071秒；33个原输入文件不变，优化0/query0。这些模型没有训练，不报告其评分为有效分类或未知识别精度。

证据 `runs/wp09-known-context-evidence02/receipt.json` SHA256 `c8f9f1bda5f221a747ade7562aae3100adb7a4507fdf20916fc3a998d9e366141`；同字节仓库摘要 `docs/architecture/wp09-known-context-engineering-results.json`。审计脚本 `runs/wp09-known-context-audit02.py` SHA256 `b6ea0b81f157c2d4f7fec06747d6a996f9c58abe50fe0648720a28c4ffa7a27d`。日志 `runs/wp09-known-context01-tests.log`、`runs/wp09-known-context-audit02.log`。全部运行产物checkout外。

## 失败、结论和后续

首独立集成审计把登记文件放在输出父目录，现有隔离守卫正确拒绝 `Output overlaps input directory`；原 `runs/wp09-known-context-evidence01` 与audit01日志保留。仅在audit02把登记移到独立目录，生产代码、输入门槛不变，重新完整核对通过。单元/回归测试无失败。

本模块CPU工程验收通过；不是WP09真实实验完成。数据语义与标注真实性仍需外部来源证据；现有checkpoint训练历史、未知类是否曾参与优化仍未获认证。真实场景若无足够完整已知区域，保留不可行结果，不能放宽为仅监督点检查。真实精度、遗忘、未知隔离训练及GPU成本继续未完成。下一项按WP12先核查累积工程模块的完整CPU整合回归，随后继续符合进入条件的剩余搭建；不将工程代码合并等同真实研究验收。
