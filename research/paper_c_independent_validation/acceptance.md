C9 / Issue529 完成的是 WP02/WP05 独立验证条件的元数据核查与论文证据表。
没有新增可执行的独立确认设备对；真实拟合、query及新影像/GT下载保持0。
结果不改变C7两类开发准入，也不恢复C8研究训练。

分支 `codex/paper-c-independent-validation-20261127`，基于C8
`c6526763b874df55fab4991e3076a235adcac174`；唯一写入本目录，原C6–C8、D的
交付文件、主工作树和共享状态不变。元数据核验使用已发布干净源码
`7dc59390e38269b3ff36735932836e29ea78b7dc`，最终表格与8项测试使用
`9436235b360b242b77f2889e09eccbd385e70a7e`；两者均已同步GitHub再送LXC。

实际LXC为 `lbx`，Python3.10.20，使用原 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，
无依赖安装，所有执行在 `/home/lbx/hsi-pioneer271828/checkouts/paper-c-independent-validation-20261127-01`。
8项专项测试通过，0失败、0错误；最终专项测试自身0.020秒，不代表整包工时。
`scripts/check_repository.py --expected-commit 9436235b360b242b77f2889e09eccbd385e70a7e`
实际exit1，仍是继承的状态Markdown过期；未宣称全仓验收或CI通过。

| 交付 | 实际结果 |
|---|---|
| 既有候选证据表 | 17条产品记录，版本、设备、权限、类别/预处理/曝光、公开与已准入数量分别记录；未知独立地块数为null |
| 论文实验矩阵 | 16条比较，区分C7开发、同家族参照、不同家族候选和排除项；没有新放行实验 |
| 原30方向扩展 | 每个原单元格不变，追加当前限定；6同家族、14已报告不同家族、10本地身份未绑定；共同两类开发方向仍仅2个 |
| 真实授权元数据 | 已驻留C6包只提取274个JSON/399275字节；77个其他成员跳过，不提取或解释NPY |
| 洪湖空间单位 | 裸地/树冠训练各5点，parent数2/3；验证各5点，parent数1/1。复用C7和独立标签ID到原box/parent连接均一致 |
| 不确定性结构 | 两验证parent分属两类，4个有序重采样中仅2个同时有两类，单类召回各有3个有效重采样；仅结构计数，无真实模型分数或置信覆盖结论 |

执行命令为本目录的 `build_tables.py --expected-commit <上述SHA> --out <新外部目录>`、
`python -m unittest discover -s research/paper_c_independent_validation -p 'test_*.py' -v`，
以及 `audit_honghu_metadata.py` 对指纹固定的原C6迁移包的元数据审计。
实际命令参数与完整来源见脚本、registration、机器回执；原始点ID与JSON发布元数据留LXC。
另一段不导入生成器的标准库审计逐格比较旧/新30方向、17/16行、40点、逐类parent
和空值，结果见 `independent-audit-receipt.json`。它没有重新读取影像或标签数组。

复用优先于重复搭建：C7原ZIP回执和D的产品/类别/30方向表按字节固定，
SpectralEarth原始CDL交集9/11再与固定作者配置核对。HyRANK只把公开Dioni/Loukia
纳入候选，隐藏验证保持排除；森林、岩沙、稀疏植被不被强行套入C7类别。
WHU同家族、Houston同CASI系列、FLAIR多光谱补充、徐州已合并双流和Hyperview
航空代理分别保持其比较边界。MUUFL采集PDF未解析、原权限/版本及材料映射缺口
明确保留，不把MIT软件文本或数据可访问性当成新项目角色许可。

外部读取失败包括Pavia/DLR403、当前HyRANK API不可读、MUUFL PDF的错误初始路径
及正确路径的MIME限制、HyRANK截图超时。分别复用已有原证据或保持未知，没有
用镜像替换版本或下载影像。一次宽范围metadata文件查找耗时过长被中断，未修改
远端设置、其他项目进程或旧产物。专项测试、独立元数据连接与恢复核验没有失败项。

私有备份根 `/home/lbx/hsi-pioneer271828/backups/paper-c-independent-validation-20261127-01/`。
`preservation03.tar.gz`为5325388字节；574文件恢复无差异，完整源码03干净，7份
表格/回执重建逐字节一致，洪湖C7及独立连接再次复现。机器哈希见
`recovery-receipt.json`。这是同机恢复备份，不是异地灾备；包内只有源码与元数据，
不包含影像、GT数组或权重。去身份 `review03.zip`为17675字节、13项，LXC CRC及
Windows传输哈希、全部7份表哈希均通过；私有逐点/空间JSON不在审阅包中。

独立确认所缺材料和优先次序见findings与比较表每行minimum_input：已有对先补
真实地块/采集及曝光角色；HyRANK先补新粗任务定义、原多边形/覆盖与获准的有限
输入；SpectralEarth继续按D已有账户/产品/年度CDL/地理与预训练足迹缺口推进。
所需新标注或地块数量尚不能由当前计数给出统计充分下限，不填造功效或独立性。
