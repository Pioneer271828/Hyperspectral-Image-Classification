# P4 Xuzhou 四组开发空间训练与恢复验收

对应研究主清单 P4 §10.3、实施计划 C1 §4 和 §19。复用已发布的 Xuzhou 开发角色及四个支持/K覆盖层。开发训练和验证从原训练区分离，分别使用 73 块和 5 块。原验证区标记为 unused，原测试几何沿用。既有 PaviaU 固定角色入口保留，四组发布与两种初始化在新代码上通过零更新兼容核对。

运行入口 `python -m transfer.p4_spatial` 对开发角色核验原几何、角色、哈希与标签预算。训练统计只拟合开发训练块，优化和验证分别限定相应块 ID。测试把开发验证或测试坐标放入支持集并重新计算覆盖层哈希，两种输入均拒绝。MAT 影像按变量解码，模型前向只接收允许的块。真实训练与审计原始 GT/query 标签访问为 0。

迁移沿用 PaviaU 的共享初始化合同。源登记仅含 PaviaU、HongHu 更新。29 个共享张量逐位复制，6 个学习图参数、25 个 Xuzhou 域归一化状态和三头与从头组保持同初态，模型种子固定 64。两方法配对用同一块采样序列、三头监督和 Adam 预算。

LXC `/home/lbx/hsi-pioneer271828/checkouts/p4-xuzhou-dev-spatial01`，分支 `codex/p4-xuzhou-dev-spatial`，tested SHA `1218070d302d7c23b9134f42275b987488c5ff9d`。Python `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，两条单线程 CPU 队列。每个支持/K组合各运行从头与共享迁移，比较原图连续 12 步和原 unused/测试块置零后的 2+10 步。登记至最后回执 253.438 秒。

| 核验项 | 实际结果 |
| --- | --- |
| 真实训练与恢复 | 8 对、192 次更新，所有完成态恢复 0 更新，错登记恢复被拒绝并保留原状态 |
| 空间隔离与选中态 | 两份影像哈希不同，连续与恢复路径的模型、Adam、调度、RNG、训练/验证轨迹及选中态 8/8 对逐位一致 |
| 图与指标 | 每步图梯度为正且参数实际变化。独立复算 24 个选中态验证头、216 个逐类召回值 |
| 代码与数据 | 15 个冻结输入不变，CPU 29+2 项回归通过，checker 321 Python/128 JSON 通过 |

八例均按首次最小融合验证 NLL 选中第 1 步。融合验证 OA 均为 11.11%，每例 8 类零召回。完整预算效果与独立测试仍待验证。

从上述 checkout 在 `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` 环境执行 `p4-xuzhou-dev-spatial-register01.py`、`p4-xuzhou-dev-spatial-worker01.py 0` 与 `1`、`p4-xuzhou-dev-spatial-audit01.py`。另执行 `python -m unittest discover -s tests -p 'test_spatial*.py' -v`、`python -m unittest discover -s tests -p test_development_training.py -v` 和 `python scripts/check_repository.py --expected-commit 1218070d302d7c23b9134f42275b987488c5ff9d`。驱动均在 `/home/lbx/hsi-pioneer271828/runs/`，主证据在 `p4-xuzhou-dev-spatial-real01/`。

| 文件 | SHA256 |
| --- | --- |
| `registration.json` | `b96a7304ed4499a5d98f4d88697d8e02c95038031727aa2c0032f7cf4e948f05` |
| `registration-split.json` | `138665bd62cabf83fe9aea84edb6c158b8fa8f4e67edd1ae64085f38b27b9008` |
| `initialization.json` | `7efcc1ee48018eea966b7b99834da8a93835fdb413cdc7be67c8d100347e12405` |
| `acceptance.json` | `0bcc860206106b9cdcb155416f56022e3bd9712fc9d95a44f69bd392841e8ccb` |
| `pavia-compatibility.json` | `58595a9b175fa54fd085c6ee386a1c5dfcdceea091a56986a049760a5ec7de64` |
| `regression.log` | `68ba787c9630d76d7048e6573da23f8dba8021ed90d32227a11d1a031146da5a` |

PaviaU 兼容脚本首次把字符串当作 Path，修正后通过。原失败日志和正式训练产物均保留。Xuzhou 正式短测与独立审计完成。

## 后续依赖核查

HongHu 另行登记一次带缓冲细分可行性诊断。原 270 个训练块按固定长边二分、并列取高轴，中间留 2 像素。只枚举原训练块标签，原验证和测试几何不变。540 个候选子块中第 21 类的 47 个候选仍全部在子块 58（原块 65），同预算 MILP 仍不可行。诊断没有发布角色或样本。`p4-honghu-guarded-subtile-feasibility01/receipt.json` SHA256 `fc905dc11431b7a814b969fdc18e21f80ba6c3573b557112e16b0e00660e0c22`。

G2 发布绑定核查已接续。公开发布目录列 `data.mat` 与 `all_x/all_y/x/y`。本地快照是 `xuzhou.mat` 与 `xuzhou_gt.mat`，只核 MAT 头和文件哈希，授权数据树中未找到发布档案或对应同名文件。仍需发布原档或有输入输出哈希的格式转换记录。`p4-g2-xuzhou-layout-provenance01/receipt.json` SHA256 `7140d94fecb8a185041b4b99813eed537b30e251654db66d802344e5ab2b8049`。 [发布团队资源目录](https://www.3sobs.cn/share)

另按发布方的公开类别表核对 PaviaU → HongHu 的二类闭集候选。固定候选顺序为裸土（PaviaU 6 → HongHu 3）、树木（PaviaU 4 → HongHu 22）。这里只完成类别表与顺序核对，跨场景标注范围仍需登记。后续仍须核本地发布绑定，使用全新源端二类共享头，记录所有源训练祖先和仅源端的验证访问。其余类别排除在该候选闭集之外。本次没有访问目标标签、启动训练或评分。 [PaviaU 类别表](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes) [WHU-Hi 类别表](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm)
