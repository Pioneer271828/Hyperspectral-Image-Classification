# WP05 HongHu CUDA 生产准入与逐参数更新审计

## 范围和版本

从主对话审查后的交付 `7a1554838209ac7f9de831b56fab537902410369` 建立独立分支 `codex/wp05-cuda-production-admission`。本包最终工程测试提交为 `8bde423bed831d0f247c4947018b4e56df18504c`。对照主计划 WP05/P4 §10 与 WP06 §16、实施计划 §19.6–19.7。关联 Issue 178 为固定源权重，Issue 188/190 为独立支持集及有界目标运行的协议背景。GitHub 中未查到本次 WP05 准入的同名 Issue。[Issue 178](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/178) · [Issue 188](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/188) · [Issue 190](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/190)

上一包已获主对话认可的范围为 CPU 多种子登记、1,500 步合成恢复、120 例合成接受、单卡合成恢复与真实只读前向。其数据、代码身份、数值协议和证据原位保留。本包使用新证据 `p4_adam_named_step_evidence_v1`、新的 CUDA 协议 `cuda_deterministic_fp32_named_adam_gemm_v3`、独立的有界用途 `real_admission` 与独立的完整预算登记用途 `audited_full_budget`。

## 逐参数 Adam 验收

- 登记从实际模型和 Adam 参数组重建参数名称、组内顺序、序列化索引、形状、类型、可训练性和参与规则。关闭的覆盖模块没有参数，冻结参数显式记原因，启用覆盖模块根据实际可用锚点核对梯度是否存在。
- 每次 Adam 更新由实际 step 前后钩子记录一字节标记，分别保留梯度存在、梯度非零和参数值改变。零梯度的 Adam 更新计入步数，动量或权重衰减仍可导致数值变化。按图、覆盖、共享骨干、目标域归一化和目标头分别复盘。
- 接受器逐步重算累积计数，核对每个应有 moment 的键和精确 step，检查类型、形状、有限性、非负二阶矩与参数名绑定。参数名绑定从实际优化器的活参数对象生成，可识别同形状 moment 错换。恢复也执行相同计数检查。Adam 浮点 moment 的整个历史轨迹未独立重算。
- 测时与可复现轨迹分离。GPU 对阶段边界同步。更新计时覆盖前向、反向、Adam 和记录钩子，另外保存准备、轨迹证据、验证、原子 checkpoint 写盘和端到端单步时间。训练峰值记录 PyTorch allocator 的已分配与已保留字节。完成态重载、每个审计、进程时间和资源等待有各自记录。

## FP32 算子门槛及失败定位

真实 HongHu 几何为 940×475，原格 32×32，包含截短边和守卫裁切。保留实际授权 train/devval 尺寸与场景边缘形状共 10 种。按尺度 100 的 10 像素 token 间距和登记半径生成真实结构的 Q、转置 Q 和 A。检查 64/128 通道、非整除池化、重复 COO、空行、转置和非连续输入。种子 20261111，52 个算例共 438 项比对，各项沿用 `rtol=1e-5`、`atol=1e-6`，保留独立 CPU FP64 数学参考、原生 CPU FP32 参考和直接 FP32 对照。

初版在 8×23 的转置 Q 权重梯度，第 349 个存储元素（从 0 计，COO 坐标 [2,59]）发生一处直接比对超限。独立 FP64 值为 `-0.012861190728432392`，初版逐元素乘后归约为 `-0.012861132621765137`，原生 CPU FP32 为 `-0.012862300500273705`。绝对误差 `1.16787850856781e-6`，容差界 `1.128623005002737e-6`，比值 `1.0347817680404077`。该点 64 项乘积绝对值和为 18.5049161，符号抵消后约为 -0.01286。独立复算复现了逐元素归约值，原生梯度精确复现为 `dY @ X.T` 的对应元素，两者各自通过与 FP64 的检查。该结果定位到 FP32 点积归约顺序。

新版保留原自适应窗口均值和 COO 前向，使用严格 FP32 矩阵乘计算稀疏权重梯度并在 COO 坐标取值，密集特征梯度仍按索引求和。相同全部夹具、种子、三种参考和原门槛重新检查。工作进程内显式隔离替换，只拦截两个位置参数形式的 CUDA 池化和 COO×密集特征乘法。实体资格入口记录真实卡 UUID、驱动、设备类型、拦截计数、非目标调用、作用域恢复、原生 CUDA 严格反向可用性诊断和超限元素。

最终提交对真实几何的 52 个算例重新在 CPU 执行，438 项全通过。相对误差统计以非零参考为分母，接近零的分量由同一绝对误差门槛控制。

| CPU 比对 | 最大绝对误差 | 最大相对误差 | 最大误差/容差界 |
|---|---:|---:|---:|
| 新实现 FP32 对独立 FP64 | 1.00036e-4 | 7.00844e-3 | 0.983307 |
| 原生 FP32 对独立 FP64 | 7.45672e-5 | 7.30134e-3 | 0.983307 |
| 新实现 FP32 对原生 FP32 | 1.06812e-4 | 6.32765e-3 | 0.173475 |

授权 LXC 的 Python 3.10.20、PyTorch 2.7.1+cu128、CUDA 构建版本 12.8、NumPy 2.0.2。整合 checker 与两个 CLI 帮助已通过，未运行自动 CI。最终提交的执行命令和原始日志见证据目录。

| 检查 | 结果 | 进程墙钟 |
|---|---|---:|
| 新准入专项 | CPU 7 项通过 | 20.02 秒 |
| 完整预算与恢复专项 | CPU 7 项通过，实体 CUDA 1 项跳过 | 782.48 秒 |
| 相关回归 | CPU 37 项通过 | 152.49 秒 |
| 空间回归 | CPU 29 项通过 | 8.29 秒 |
| 合成 20 例成对批次 | 2,400 更新、20 例与 4 组全接受，重调 0 | 219.61 秒 |

新证据下，连续 1,500 步与 2+28+630+840 步共 3,000 次合成更新，参数参与、Adam、调度、RNG、轨迹、验证历史和选中状态逐位一致，每路径 50 次验证，末态重载各新增 0 步。53 个参数均记录 1,500 个 Adam step。共享骨干有 4 个参数步为零梯度，覆盖模块有 2 个零梯度参数步和 1 个数值未变化的参数步。旧合同的合成 120×6 路径另完成 720 次更新，新有界单例合成路径另完成 120 次更新，旧 CLI 另有 2 步。真实登记缺少设备资格回执时在创建产物前拒绝，实际源权重、标签和授权范围均经固定父身份读取。首次源码测试的设备变量命名冲突和测试语句错误已修正。随后的测试断言需随新协议名调整，旧日志和复测结果均保留。

**实体 CUDA 待测，真实接入受资源阻塞。** 最终版本资格窗口 40 次硬件快照中，39 次零卡达标，1 次仅 GPU 4 达标。GPU 4 两次启动时分别为 17,281 MiB / 48% 和 21,217 MiB / 49%，子进程复检分别变为 9,587 MiB / 99% 和 9,769 MiB / 98%。两次均在创建 CUDA 上下文前停住。预加载重试也未获合格资源。上海时间 21:31:55 最后一轮仍为 0 张卡达标，GPU 4 为 9,255 MiB / 98%，GPU 7 为 10,772 MiB / 100%。设备、驱动和过程日志已保留。最终版本没有实体算子通过或失败的数值结论。

## 真实接入预算与权限

真实有界用途固定 support seed 2048、K1/K5、scratch/shared_transfer、五个覆盖组、目标种子 512、sampler 64、原 source64、135 个原带与 nominal 元数据。20 例为 4 个五组配对。每例连续 60 与分段 2+28+30 两条路径，总预算 `20×60×2=2400`。开发验证在 30/60 步，选第一次最低融合 NLL。两卡各承担一组 K1、一组 K5，一种 scratch、一种 shared_transfer。每组五臂及两条路径固定同卡。完整 120 例长训练是独立用途。

已按真实释放角色零更新复算 60 步曝光，两个 K 的前 24 步与旧序列逐项相同。K1 第 22 步覆盖全部 22 点，第 21 类在 30/60 步累计 2/5。K5 第 18 步覆盖 22 类，在 30/60 步只覆盖 49/71 个支持点（共 110 点），第 21 类累计 5/10。K5 保留未覆盖点，属于短预算限制。

接受要求每例两条路径的模型、Adam、调度、RNG、轨迹、验证历史、选中状态和命名证据指纹精确一致。固定 train/devval 的初态、选中态及末态复用独立三头逐类读数合同。仅重放原观测；七观测身份按原规则固定。本包真实表现、覆盖收益与小类召回以实际完整配对接受为前提。

## 运行入口

以下命令在干净且 SHA 固定的授权 LXC checkout 执行。资源门槛为每卡空闲显存至少 12,288 MiB、利用率至多 60%。每卡先独立生成同 SHA 的算子资格文件。正式登记固定两个 UUID、资格回执、各角色及预算，首派发前必须在同一次硬件清单中两卡同时合格，且可持有两卡租约。之后每个协调器调用与子进程重复检查该卡。原预算、数据权限和失败停止规则保持。

```bash
base=/home/lbx/hsi-pioneer271828
python="$base/envs/hsi/bin/python"
runs="$base/runs"
sha=8bde423bed831d0f247c4947018b4e56df18504c
parent="$runs/wp05-honghu-paired-exposure-short01/registration.json"
prior="$runs/wp05-honghu-paired-exposure-independent01/receipt.json"
protocol=cuda_deterministic_fp32_named_adam_gemm_v3
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test "$(git rev-parse HEAD)" = "$sha" && test -z "$(git status --porcelain)"
# gpu1/gpu2 是通过资源检查的不同物理 UUID，qa1/qa2 是新的 checkout 外目录。
mkdir -p "$runs/.p4_gpu_leases"
flock -n "$runs/.p4_gpu_leases/$gpu1.lock" env CUDA_VISIBLE_DEVICES="$gpu1" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$python" -m transfer.p4_cuda_qualification --expected-commit "$sha" --parent "$parent" --gpu "$gpu1" --out "$qa1"
flock -n "$runs/.p4_gpu_leases/$gpu2.lock" env CUDA_VISIBLE_DEVICES="$gpu2" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$python" -m transfer.p4_cuda_qualification --expected-commit "$sha" --parent "$parent" --gpu "$gpu2" --out "$qa2"
"$python" -m transfer.p4_full_budget_campaign prepare --expected-commit "$sha" --purpose real_admission \
  --parent "$parent" --prior-receipt "$prior" --protocol "$protocol" --gpus "$gpu1" "$gpu2" \
  --qualification "$qa1/qualification.json" --qualification "$qa2/qualification.json" --out "$registration"
scope="$registration/scope.json"
scope_sha=$(sha256sum "$scope" | cut -d' ' -f1)
"$python" -m transfer.p4_full_budget_campaign run --expected-commit "$sha" \
  --scope "$scope" --scope-sha256 "$scope_sha" --out "$execution"
# 资源等待返回后，以相同 SHA、scope 和目录续接。
"$python" -m transfer.p4_full_budget_campaign run --resume --expected-commit "$sha" \
  --scope "$scope" --scope-sha256 "$scope_sha" --out "$execution"
"$python" -m transfer.p4_full_budget_campaign accept --expected-commit "$sha" \
  --scope "$scope" --scope-sha256 "$scope_sha" --out "$execution"
```

未来完整批次使用 `--purpose audited_full_budget` 与新的独立 scope 和产物目录。矩阵为支持 512/2048、K1/K5、两初始化、五臂、三个目标种子，固定 120×1,500=180,000 更新，每 30 步验证。CUDA 登记绑定每张卡资格，两/三卡每卡 60/40 例。正式训练步耗以真实短接入的稳态单步（含写盘）、验证、重载及完整性检查测时为依据。短接入权重和轨迹独立保留。

## 证据

| 文件（相对 LXC 项目根） | SHA256 |
|---|---|
| `runs/wp05-cuda-admission-engineering02/fp32-reduction-detail.json` | `67c5b0a554dfd5dfb1168f467d4f9d730e27ce9bf89196d19b262bcdf3743ba0` |
| `runs/wp05-cuda-admission-engineering04/real-geometry-reference.json` | `745fd21d131358c9d5d47fd538db1bd00591d2f9f508528f0c5ba16c25843e78` |
| `runs/wp05-cuda-admission-engineering04/synthetic/long-split-audit.json` | `872691fd9520c61c77b8d1f1cc605437eb050c7ea986280c9edcabdc02eaa96d` |
| `runs/wp05-cuda-admission-engineering04/synthetic-paired01/execution/acceptance.json` | `c16674ca0d78f815b4db93ddbbe33812dc59be765ed66946b441af52a06df13a` |
| `runs/wp05-cuda-admission-audited-full-preflight04/scope.json` | `358b50aee1bf51feeb80fe7ef1e603b42662da3c6bf6886fa4b636155ca391fd` |

证据副本位于 `F:/E2E_VCA_worktrees/wp05-cuda-admission-evidence01`。其中保留实际工程序列、资格窗口的原始硬件快照、失败数值与三项参考全量统计。真实完整预算 v2 的 CPU 零更新预检完成 120 个登记初态并绑定命名参数身份，旧计划父回执的 107 个文件已复核。

本轮真实短接入实际为 **0/2,400 次更新**。原始 GT 解码和 query 访问均为 0。实体 CUDA 三路数值对照、同卡精确恢复、双卡真实 2,400 步和训练峰值显存仍待完成。真实 180,000 步未启动，真实 2/3 卡耗时估计待含写盘步耗的实测数据。
