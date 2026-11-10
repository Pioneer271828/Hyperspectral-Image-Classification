# WP03：Issue249失败批次证据清点与恢复边界

Issue339对应研究主清单§22/WP03及实施计划§19.4。完成的是只读证据清点工程与原批次材料复核；显存修复、数值兼容证明、恢复训练和完整实验仍未完成。不得据本文续跑失败目录或释放query。

## 版本、实现与验收

- 分支`codex/wp03-failed-campaign-inventory`，实际tested SHA `a66081740361c8bdcccafc73eafb71311bac69d6`；基准main `f9ac9bede461f114ebcd18953be31195bee311ac`。
- 新增独立只读入口`g4_light_failed_inventory.py`及负例测试。复用原JSON格式与哈希证据，不导入训练器、不反序列化模型或标签、不改变模型/损失/预算/选模/旧生产代码。此工具没有恢复或全批接受功能。
- LXC `lbx@10.69.37.243:1222`，Python `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`；干净测试目录`checkouts/wp03-failed-inventory02`。
- CPU组合25/25通过，88.397秒，无跳过；checker 552 Python/177 JSON及CLI帮助通过。组合含新增13项，以及原队列、全54冻结门禁、复盘入口12项。原队列测试仍有实际合成1500步案例，与本次真实材料零新增更新区分。
- 负例覆盖权重/受保护输入篡改、重复和混登记回执、未固定外部回执、结构别名冲突、遗漏更新/错误首次最优、缺失权威状态、额外产物、GPU角色/query声明、运行中批次、已有全局接受与符号链接。仅有result不能算接受。

```sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests
python -m unittest tests.test_g4_light_failed_inventory tests.test_g4_light_full_campaign tests.test_g4_light_frozen_boundary tests.test_g4_light_trainval_review -v
python scripts/check_repository.py --expected-commit a66081740361c8bdcccafc73eafb71311bac69d6
python -m g4_light_failed_inventory --help
python -m g4_light_failed_inventory \
  --expected-commit a66081740361c8bdcccafc73eafb71311bac69d6 \
  --registration /home/lbx/hsi-pioneer271828/runs/g4-light-full-issue249-registration01.json \
  --registration-sha256 b3cca8f1028004da30177704b107e959cd2fa47aa7892390522e4589fe39e735 \
  --producer-commit 2cfca12ae52308189c81c6d32dcea10f0ac1b50c \
  --extra-receipt /home/lbx/hsi-pioneer271828/runs/issue249-post-drain-audit01/receipt.json 22c593478f4badda8b8f54e61d7bc148dd261d666f73283b63ed71e8e11af8aa \
  --out /home/lbx/hsi-pioneer271828/runs/wp03-issue249-inventory03.json
```

输出必须是新文件；上述输出已存在，不覆盖重跑。日志根为`runs/wp03-failed-inventory-tests01/`。

## 原批次实际处置

| 材料级别 | 案例数 | 本次结论 |
|---|---:|---|
| 控制器原接受回执 | 35 | 输入、产物、JSON训练/选步记录重新核验通过 |
| 排空后外部独立回执 | 1 | 原固定回执与产物重新核验通过，未改写控制器35计数 |
| 首步失败 | 1 | HongHu64/none/VCA加三头；仅identity与adaptation-cost，没有持久恢复游标 |
| 未运行 | 17 | 无案例目录，不以计划预算当已完成更新 |

36例按场景为PaviaU18、Xuzhou17、HongHu1；历史接受更新54,000/81,000。27,000是尚无接受材料的预算，不是授权的新训练预算，也不保证可直接补跑。现有36例中19例选择1500末步，不能据此宣称充分收敛。

4,432个文件在读取前后哈希一致，耗时21.320823秒；复核包括固定登记、三个登记依赖、逐例保护输入、全部产物、1500步trace、51事件及first-min选择。原回执累计216头指标、2022逐类召回、108次精确重载，本次仅重新认证这些历史声明，没有再次重算概率/优化器/调度器或执行GPU重载。额外直接读取回执和result的计数交叉核对也是36例/54,000步。

生产checkout仍干净且HEAD为`2cfca12ae52308189c81c6d32dcea10f0ac1b50c`。原协调器PID658729已不存在，status为failed，缺`training-acceptance.json`。本机`vca`配置实际为PAUSED；未重启训练、未增加真实更新、未读query标签。证据清点工具本身不判断PID存活，运行前由操作者核对停止状态。

## 原因与恢复条件

原失败日志SHA256 `a73da54c51a430085c36a236e9bb1c5be6de467e9c6075bd123a60b9406e3349`未变。日志报告值投影申请540 MiB时OOM，PyTorch已分配约21.38 GiB、仅余70.31 MiB，另有约1.53 GiB共存占用。代码路径中骨干eval但适配器反传，而VCA重计算条件要求training，因此分块投影仍保留反传激活；这是与日志一致的显存压力解释，不是本次新做的峰值实验。

PR251仍为Draft，提供的CPU saved-tensor短测候选曾让另一LP-FT HongHu案例完成12步，但该分支使用不同两阶段/优化器/BN合同，且全网络GPU位级等价未建立，不能当作Issue249的修复认证。保留其结果与限制，不把它移入旧登记。

下一步若修复本批次：先独立版本验证eval反传显存路径、输出/梯度/实际更新、BN与随机状态保护，以及优化器/调度器/选模恢复一致性；再按原数据与判据判断是否兼容。旧协调器明确拒绝failed恢复，失败例也没有cursor；不得删除目录、伪造游标、把status改回running。改变执行合同须新版本新登记，并说明36例能否合法复用。仍须全54接受、独立审计、复盘后才能统一冻结评价。

本次GPU0—7空闲显存均低于12,288 MiB（最高7,100 MiB），没有启动GPU工作。GPU/显存修复实验、全批效果与全仓回归未做；这不阻塞其他已授权工程。

## 失败记录、证据与交付

1. 初次bundle依赖尚未同步的main合并提交，fetch拒绝；随后测试在旧目录报缺模块。保留`cpu.log`，改以远端已有ff75为bundle基准并验证新目录后运行。
2. 首版`2e68e05bbd96c5a3b4d9f72051076de8835481ae`的11合成测试通过，但真实NONE没有额外m4_mode字段，`real01.log`拒绝。新版从所有原模型都有的config字段读取结构，并拒绝额外别名冲突；不修改原训练材料。
3. `real02.log`拒绝交接摘要误抄的65位外部回执SHA；从原文件直接读取正确64位SHA后`real03.log`通过。固定登记、回执和数值门槛均未改。

完整清点回执`runs/wp03-issue249-inventory03.json`，SHA256 `37b398e982f4e9069147ccada2bf55da71b544810c03ecd24620e03ed304ce51`；本机原字节副本`F:/E2E_VCA_worktrees/wp03-issue249-inventory03.json`。仓库摘要[light-failed-campaign-inventory-results.json](../../docs/architecture/light-failed-campaign-inventory-results.json)保留54例分类和完整回执定位，不存训练产物。

验收结论：**失败批次证据清点模块通过；WP03整体恢复与完整实验未完成。** 本聊天2026-11-10交付本模块报告后，继续WP04缺失的I7/I9组合工程接入；已有消息/联合边权算子复用，旧单机制接口和冻结实验保持可追溯。
