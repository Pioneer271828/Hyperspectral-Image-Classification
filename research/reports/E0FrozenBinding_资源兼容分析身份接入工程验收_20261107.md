# E0-R资源兼容分析身份接入工程验收

Issue239 / PR240。对应实施计划§16.1 E0-R冻结表征诊断和§19冻结评价衔接。实际tested SHA `a81d9ca88c472ce4939bd4132d337fca44f8d5fa`，分支 `codex/e0-frozen-resource-analysis-binding`。

本构建接纳184实际使用的分析代码 `bdb4b3f14cc108e632b986ad7bffec7cd1a70bfc`，完整源码身份 `790a5e39400419ec9d51e06f56875a62520b348d4604c4a34f3902c2286e98ee`。合入该版本的两个分析文件，冻结表征入口同时核对commit和source hash。已有模型、原54冻结合同、计划hash、权限、观察身份和逐文件保护沿用原实现。早期220构建及回执保留。

## 实际验收

授权LXC `/home/lbx/hsi-pioneer271828/checkouts/issue239-binding01`，实际分支 `codex/issue239-binding01`，Python为 `envs/hsi/bin/python`，单线程及无bytecode。

```bash
CUDA_VISIBLE_DEVICES='' REPRESENTATION_CAPTURE_DEVICE=cpu python -m unittest discover -s tests -p test_frozen_representation_capture.py -v
python scripts/check_repository.py --expected-commit a81d9ca88c472ce4939bd4132d337fca44f8d5fa
```

CPU **4/4通过，20.130s**。checker **405Python/107JSON**。既有用例覆盖六种合成角色的原阶段采集、状态/RNG保护、异常清理和冻结合同。身份拒绝测试增加旧分析身份组合。此次修改集中于分析身份门控，验收采用完整相关CPU专项及真实元数据链路。

实际184认证使用原freeze SHA `7dd7e87334766d74005d14345e994fc2e700e109f6d8141b462004831496b48c` 和分析receipt SHA `9b9d2e1390fa54915cc34bac87e6826cc5cb138cc746ae1c5012bead8fec441b`。调用 `authenticate` 核全54结构及保护边界，并认证 `paviaU-s64-none-target_scratch_full` 的具体case/无标签观察身份。执行器将Tensor反序列化、模型执行和query释放替换为拒绝函数，确认新增训练更新0。

**14205个保护文件核验通过，耗时355.068836s。** 接受184主要收益为false的实际分析回执。工程范围为全批次边界及一个具体case的身份核验，完整54例表征提取、统计与真实诊断待后续阶段。

## 证据与衔接

实际回执位于LXC `runs/e0-frozen-resource-analysis-issue239-boundary01/receipt.json`，SHA256 `ce143e36f1afbe9673824ad93205ad0f79d40de058b12baeb9c71cbe5e4486c3`。仓库副本见[工程结果](../../docs/architecture/e0-frozen-resource-analysis-binding-results.json)。CPU及checker日志为 `runs/issue239-cpu01.log`、`runs/issue239-check01.log`，真实入口核验日志为 `runs/issue239-actual-boundary01.log`。

验收结论为资源兼容分析身份接入工程通过。表征采集、冗余统计、容量及逐类效果仍按计划分别验收。下一项完善全批次表征采集的工程调度与完整证据汇总。
