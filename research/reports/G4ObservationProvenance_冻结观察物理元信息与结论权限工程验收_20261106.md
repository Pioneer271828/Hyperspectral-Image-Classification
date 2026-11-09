# G4冻结观察物理元信息与结论权限工程验收

Issue194 / PR195；对应研究主清单§4.2、P4 §10.1–10.3，实施计划C1 §4与§19的空间/传感器/波长质量证据要求。用途是把实际消耗的观察向量、来源等级和允许结论绑定，防止数值元信息存在就被写成物理标定或严格传感器泛化。E0-B1/B2的空间支持和冻结描述性诊断已经交付，本项不重复，也不替代独立空间协议。

## 实现和复用边界

工作树`F:/E2E_VCA_worktrees/issue-194-observation-provenance`，分支`codex/issue-194-observation-provenance`，基于192交付`f2a0512951cd8fe33851ad4543eee3e02aaae383`，唯一写入人为本聊天助手。新增`g4_observation_provenance.py`、`scripts/audit_g4_observation_provenance.py`、`tests/test_g4_observation_provenance.py`及`docs/references/g4-observation-evidence-v1.json`。原331及184全部Python/config、旧registry/波长/图/标签/模型/预算不变，无多智能体或main合并。

复用188的真实27输入manifest及独立接受回执、既有loader和BandMetadata、图/观察身份及版本检查。只解码3个无标签观察`.pt`；27覆盖层和curator family仅哈希，不反序列化。检查实际dtype、形状、波长、缺失FWHM、完整record指纹、图、27角色/标签预算及证据文件身份。输出仅为checkout外的新sidecar/receipt，拒绝覆盖和输入目录交叠；没有模型前向、梯度、优化或query接口。哈希会读取保护文件字节，“不解析标签”不冒充OS安全沙箱。

六项结论权限固定：允许描述“有标签少样本、整景无标签上下文的传导适配”；拒绝严格空间隔离、无目标接触的传感器DG、已核实的完整源家族留出、实测波段中心及物理FWHM。允许描述不等于训练许可、模块效果或模型接受，receipt始终training_executable=false。G4是历史代号，不能借用计划G4/TTA名称把当前K+V监督条件改称纯源TTA或零样本。

## 来源核查与实际发现

先读两计划、原D4设计、现有loader、Xuzhou包说明和HongHu项目metadata。当前可见目录只找到上述说明/metadata，没有在本次冻结输入中认证ENVI header、真实中心波长/SRF/FWHM或原始删带映射。没有全盘搜寻或据此声称这些文件在所有存储中不存在。

[WHU数据提供者页面](https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm)和[原始数据论文v2第1–3页](https://arxiv.org/pdf/2012.13920v2)支持WHU-Hi各景使用Headwall Nano-Hyperspec的名义采集说明；WHU-Hi是集合而非每景一个新传感器类型。该说明不认证当前本地cube的具体仪器校准文件。页面的学术用途/引用要求单列保留，本项不下载或重新发布数据，也不复用外部算法/源码或增加依赖。

Pavia的ROSIS名义说明在已有registry、数据论文表及[EHU提供者缓存描述](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)中可见；EHU两个直接URL均403，未把搜索可见写成当前原页完整获取。Xuzhou本地包说明报告HYSPEX相机和删噪后的436波段，但型号、包出处认证、原始保留索引和逐带中心仍未确认；旧registry的unknown不改成“已核实HYSPEX”。

| 实际冻结观察 | 形状H×W×B | 实际输入向量范围（nm） | 质量与限制 |
|---|---|---|---|
| PaviaU | 610×340×103 | 471.491241–856.228088 | approximate；显式loader数组及所声明删带，不是认证中心表 |
| Xuzhou | 500×260×436 | 415–2508 | approximate线性位置；删噪保留带未有认证原始索引映射 |
| HongHu | 940×475×270 | 400–1000 | approximate线性位置；项目metadata明确没有真实中心波长文件 |

Pavia这一范围是**当前代码实际消耗值**，不是对真实ROSIS覆盖范围的重新测定；不能把资料中的一般范围代写成当前输入，也不在运行中“纠正”冻结向量。三组均无FWHM/SRF；波长间距不能替代物理带宽。BandMetadata的0…B−1是本地保留数组位置，不自动成为原始仪器波段编号。向量逐值等于现有loader只证明绑定正确，不证明物理真实。

## 实际版本、环境与测试

实际tested/工程生产完整SHA：`4f454b217f79576de92afc769fb6cb8452e71874`。

| 环境 | 独立checkout/分支 | 结果 |
|---|---|---|
| rmvision Linux，Python3.12.3/Torch2.7.0a0+ecf3bae40a.nv25.02/CUDA12.8 | `/home/rmvision/issue194-provenance-engineering01` / `codex/issue-194-observation-provenance` | CPU6/6，9.528s；完整588项587通过+1既有GPU专用CLI跳过，369.611s；checker372Python88JSON/help通过 |
| LXC Linux，Python3.10.20/Torch2.7.1+cu128，CPU且CUDA_VISIBLE_DEVICES为空 | `checkouts/g4-observation-provenance-issue194-engineering01` / 同分支同SHA | 真实3观察/27角色，52输入/证据文件不变，6.672247s；只解码3观察、不解析标签、更新0/forward0/query0，无GPU上下文 |
| LXC独立checkout，同SHA | `checkouts/g4-observation-provenance-issue194-independent01` / `codex/issue194-provenance-independent01` | 独立809个波长值、27预算/图绑定、6权限，54文件不变，4.457053s；checker372Python88JSON通过 |

Windows仅编辑/读文件/Git/SSH/SCP。新模块是CPU元信息与权限工程，新增GPU专项0；不虚列GPU数学或速度验收。完整回归保留GPU0可见以执行未改的既有CUDA hooks，启动前free14403MiB/util0%，满足free≥12288MiB/util≤60%，单队列。既有显式GPU-only CLI仍跳过，192等历史GPU记录保留，不冒充本轮重跑。CI runner仍未注册。

```text
# rmvision，/home/rmvision/venvs/hsi-baseline/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -p test_g4_observation_provenance.py -v
CUDA_VISIBLE_DEVICES=0 MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit 4f454b217f79576de92afc769fb6cb8452e71874
python scripts/audit_g4_observation_provenance.py --help
# LXC，CUDA_VISIBLE_DEVICES为空；外部驱动调用同一CLI并禁止模型/optimizer与其他.pt解码
python /home/lbx/hsi-pioneer271828/transfer-records/issue194-evidence01/issue194-real-audit01.py
# 另一个干净checkout，独立NumPy向量与手工权限/预算oracle，不调用生产审计函数
python /home/lbx/hsi-pioneer271828/transfer-records/issue194-evidence01/issue194-independent01.py
python scripts/check_repository.py --expected-commit 4f454b217f79576de92afc769fb6cb8452e71874
```

6项专项包含实际CLI/错误SHA提前拒绝、27预算/严格三次观察反序列化、向量NumPy参考、改dtype/伪FWHM/额外标签拒绝、质量和权限伪升级拒绝、完整重哈希仍错误的角色/图/路径/空间标记拒绝、输入与输出隔离、禁止覆盖及末端变更守卫。真实生产驱动禁止Adam和所有Module forward，并记录加载文件；独立核验以原Pavia常量、另写的NumPy线性构造和字面权限/类数公式为oracle，重新核每条图/向量/预算和文件不变，不调用新生产函数。它不独立认证仪器或包来源真实性。

## 证据、未测项与交付

真实producer `runs/g4-observation-provenance-issue194-audit01/receipt.json` SHA256 `a311a4a08f3a6bfa3d639d4cde65444bc6f9deff2bd86727b884d526efaebdff`，同目录access.json保留3次观察加载及331生产身份。独立`runs/g4-observation-provenance-issue194-independent01/receipt.json` SHA256 `b86a4f843ec5d81fc209cbb429f9fd46230daac010d982db9964275c99215eb1`；真实证据包SHA256 `f7df7257707357815fcdc1692d93008d924a32145a635a5c1d544ff3903cd34a`。证据目录只增加说明的只读副本，用户原文件原位保留。

汇总`/home/rmvision/issue194-provenance-engineering01-receipt.json` SHA256 `7452dbf4c25435e9c6cf68b66fd3f1c88b291eb7a9bad15f732f4b60f7b3acb6`，同字节副本`docs/architecture/g4-observation-provenance-engineering-results.json`。来源复核目录为`docs/references/g4-observation-evidence-v1.json`，SHA256 `c839736f0d18650795c89cce97ed0fc9fd0b3bf99a3fe378ee4ec1a5315820e6`。原始输入、只读证据副本和日志在checkout外，不提交真实影像、标签张量、权重或运行日志。主入口只同步报告/目录/摘要/状态与远程投影，不合未评审Python源码。

本tested版本CPU/完整回归及真实审计未发生代码/数值失败；EHU403访问失败、传感器型号/映射及中心/FWHM/SRF缺项均保留。**本只读绑定与结论权限工程验收通过**；不据此完成物理标定、严格空间/传感器泛化或真实效果实验。tag `module-g4-observation-provenance-v1`指实际4f454b2；Issue194关闭、PR195独立待评审，main不合并。

184交付检查点：45/54、67500/81000接受，新增HongHu512残差scratch选931/LR1442，1417.550s，三头验证无零召回；控制器按GPU利用率门槛等资源，属于合法运行批次，不重启。原HongHu64残差scratch验证CNN第11类0召回等限制继续保留。本次不审计184全批次、不读原GT/query、不改预算。下一阶段仍是全54接受→331独立审计→186真实拟合/逐类/成本复盘→统一冻结query/预登记报告，之后才登记支持/K/参数范围及独立空间/传感器对照；元信息缺口与其余计划必测机制保持未完成。

本聊天报告位置：2026-11-06 Issue194冻结观察物理元信息与结论权限工程模块完成报告。该报告是模块交付的一部分，后续不重复。
