# G4 M8位置门控有界训练与恢复工程验收

Issue234 / PR238。对应研究主清单§2.3/P4和实施计划§12、§18.3 F、§18.4 残差×门控四组。本阶段验收末端融合适配的有界运行时、恢复、门诊断与真实配置组合。完整实际tested SHA为 `c78696ac494030871ac68676a956459afb88fad3`，开发分支 `codex/g4-position-fusion-runtime`，实际远端分支 `codex/issue234-runtime01`。

## 实现

`g4_position_fusion_target_runtime.py` 和 `scripts/train_g4_position_fusion_target_short.py` 将已验232 fixed/position_gate组件接入scratch和完整none源FT四角色。显式engineering profile上限12更新，分别验证完整12步、跨实例2+10步及完成态恢复0步。沿用支持/K身份、三头概率损失、Adam/调度、首次最小验证NLL选步和ordered FP32总体BN。回执保留完整214基座、232门配置、原BN回执及源/观察/状态身份。恢复先按登记设备重建，再移到请求设备。GPU登记态恢复至CPU的专项也通过。

启用门新增129参数与8字节版本buffer。训练trace分别保存门权重与偏置梯度范数、实际变化元素及参数量。Adam更新前拒绝缺失或非有限门/头梯度，合法零梯度通过，异常后清理检查钩子。测试覆盖源、标签、配置、模式、原始文件哈希、嵌套BN回执、版本、输出目录及CLI权限。原232工程回执认证通过，与实际232版本的439个Python/config文件一致；只读复用216的三份Python文件和已验参考，不引入新外部依赖。

## 环境、命令与验收结果

授权LXC `/home/lbx/hsi-pioneer271828/checkouts/issue234-position-runtime01`，解释器 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`，Python3.10.20 / Torch2.7.1+cu128 / CUDA12.8 / RTX4090 GPU0。各执行使用单线程、无bytecode和 `PYTHONPATH=.:tests`。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
CUDA_VISIBLE_DEVICES='' POSITION_RUNTIME_DEVICE=cpu python -m unittest discover -s tests -p test_g4_position_fusion_target_runtime.py -v
CUDA_VISIBLE_DEVICES=0 POSITION_RUNTIME_DEVICE=cuda python -m unittest discover -s tests -p test_g4_position_fusion_target_runtime.py -v
CUDA_VISIBLE_DEVICES=0 POSITION_RUNTIME_DEVICE=cpu POSITION_FUSION_DEVICE=cpu POSITION_TARGET_DEVICE=cpu MATCHED_TARGET_DEVICE=cpu LIGHT_TARGET_DEVICE=cpu SUPPORT_TARGET_DEVICE=cpu OBS_TARGET_DEVICE=cpu COVERAGE_TARGET_DEVICE=cpu COVERAGE_RUNTIME_DEVICE=cpu INTERACTION_TARGET_DEVICE=cpu INTERACTION_RUNTIME_DEVICE=cpu EDGE_TARGET_DEVICE=cpu EDGE_RUNTIME_DEVICE=cpu FUSION_TARGET_DEVICE=cpu FUSION_RUNTIME_DEVICE=cpu python -m unittest discover -s tests -v
python scripts/check_repository.py --expected-commit c78696ac494030871ac68676a956459afb88fad3
python scripts/train_g4_position_fusion_target_short.py --help
```

| 验收 | 通过 | 跳过 | unittest耗时 |
|---|---:|---:|---:|
| CPU专项 | 7/7 | 0 | 76.149s |
| GPU专项 | 7/7 | 0 | 89.549s |
| 完整回归 | 669/670 | 1项既有GPU专用CLI | 1295.278s |

checker为410 Python / 109 JSON，CLI help通过。GPU专项/全量启动前GPU0均空闲18130MiB，利用率分别0%和14%，满足工程资源门槛。全量保留GPU可见供既有用例，专项CUDA新路径另测。CPU恢复对model/optimizer/scheduler/history/RNG/概率/trace逐位一致。两个fixed角色在CPU及声明的固定图GPU参考上与216默认数值路径一致；四角色每套共12次同路径精确重载。原生GPU验证包括启用门执行与严格恢复，独立优化轨迹等价保留为未测。

## 真实配置与独立复核

在原232工程回执、真实初始化与独立状态回执基础上，按三景×两条件×两角色实际组合12个配置并构造12个初始化。model64/head1064、support512/K1和每类另5个验证标签。新runtime的架构、初态、门/基座绑定和支持/验证上下文逐一匹配232记录。独立进程仅使用已验旧支持设置与232配置/状态记录，手工重建12份完整配置并核哈希。

```bash
CUDA_VISIBLE_DEVICES='' python /home/lbx/hsi-pioneer271828/issue234-runtime-real01.py compose
CUDA_VISIBLE_DEVICES='' python /home/lbx/hsi-pioneer271828/issue234-runtime-real01.py verify --producer-sha256 124da03f82f0d06317b410b6335bd8614aa08b33c32433a2b28e574cbbea1e74
```

| 证据 | 项数 | 保护文件 | 耗时 | SHA256 |
|---|---:|---:|---:|---|
| 真实组合 `g4-position-fusion-runtime-issue234-preflight01/receipt.json` | 12 | 99 | 128.561s | 124da03f82f0d06317b410b6335bd8614aa08b33c32433a2b28e574cbbea1e74 |
| 独立复核 `g4-position-fusion-runtime-issue234-independent01/receipt.json` | 12 | 100 | 3.928s | 633acd8fc4c3d03b61d5714a278eb4455160f5d20405207510e81d91e2e775f0 |

路径以LXC `runs/` 为根。两阶段真实NN前向、BN校准、优化更新、原GT/query访问均为0。组合阶段读取释放的支持/验证覆盖，独立复核使用其已验身份记录。真实四组B0/R/F/R+F、配对收益、逐类与受控成本属于后续实验。

## 保留事项与衔接

初次SCP连接关闭，成功传输后才执行。源码初版的文件末尾空行在测试前修正。GPU/全量执行器初次保存的验证元数据含字面换行分隔符，另存 `issue234-runtime-validation02.json` 规范化并保留原日志、计时与校验。合成数值验收失败0。CI专用runner待注册，测试证据来自实际LXC。

工程回执 `runs/g4-position-fusion-runtime-issue234-engineering01/receipt.json`，SHA256 `45bf77248cec505d070d2f1adf603f720d3c92c80d9e1273b78281f02fa235f4`，仓库证据为 [工程结果](../../docs/architecture/g4-position-fusion-runtime-engineering-results.json)。验收结论为**有界训练、严格恢复及真实配置组合工程通过**。后续衔接E0-R冻结表征入口与184实际分析身份，保留原220边界、只读保护和完整真实诊断待办。
