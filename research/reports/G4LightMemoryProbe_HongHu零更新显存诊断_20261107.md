# G4 HongHu 零更新显存诊断与交接

对应研究主清单 P4 §10.3、实施计划 §19、Issue249 HongHu 首步 OOM。新增独立探针 `g4_light_memory_probe.py`，核对原登记、失败角色输入身份及冻结代码，用 PyTorch saved-tensor CPU 暂存执行原前向和反传。保留训练态三头和 eval 骨干，检查模型参数、缓冲区与梯度权限，输出峰值显存、内存、时耗及哈希。诊断不调用优化器，产物在生产目录外。[PyTorch autograd 文档](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu)

## 验证结果

- 分支 `codex/g4-light-memory-probe`，实测 SHA `7d09fd14821964fb398b8ff1669537b0fcbbdce8`。授权 LXC 独立检出 `/home/lbx/hsi-pioneer271828/checkouts/issue249-memory-probe01`，Python 3.10.20、PyTorch 2.7.1+cu128。仓库检查423 Python、131 JSON与 CLI 帮助通过。合成 CPU 1/1（2.930秒），两路径输出、损失、梯度和模型状态哈希相同。
- 物理 GPU5 合成逐字节对照失败（4.222秒），原路径和暂存路径的梯度哈希不同。追加复测发现原路径重复本身也有差异，测得梯度最大绝对差约3.05e-5，原路径对暂存约4.58e-5。带确定性开关和 cuBLAS 工作空间配置仍有前向输出哈希不一致。保留失败，不降低判据。
- 另行做一次真实显存测量，物理 GPU5 的原失败角色 `whu-honghu-s64-none-source_pretrained_adapter_heads` 完成零更新前向与反传，19项许可梯度均有限。峰值 GPU 已分配2.35 GiB，前向加反传32.47秒，进程峰值 RSS约34.05 GiB。启动前可用 GPU 显存约21.60 GiB。训练更新0，原始 query 读取0。单步容量证据范围限此探针，真实梯度等价、后续更新和成本未验收。

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_repository.py --expected-commit 7d09fd14821964fb398b8ff1669537b0fcbbdce8
python -m g4_light_memory_probe --help
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_g4_light_memory_probe.py -v
CUDA_VISIBLE_DEVICES=5 LIGHT_MEMORY_TEST_DEVICE=cuda:0 python -m unittest discover -s tests -p test_g4_light_memory_probe.py -v
```

证据根为 `/home/lbx/hsi-pioneer271828/runs`。常规专项 `issue249-memory-probe01.log` SHA256 `824a282d4ba57ae8bea2256956061138d3fc23e55bfa2a5bb2fdf4148b5ea207`，复测 `issue249-memory-gradient-diagnosis01.log` SHA256 `1547b42a76d99fbefb88c69a99862f6db90b94777ab081044faac91cfba3965c`，确定性复测 `issue249-memory-parity-deterministic01.log` SHA256 `decaabdd134e6bf6f95306d77c6a8a05d9080d242f04ef24614c31cb341048ac`。真实探针 `issue249-honghu-memory-probe01/receipt.json` SHA256 `6d64637cdd5cafc7b06bd407cbb358c722f63814a373033128f55b1e0e498da2`。

## 交接

后续模块搭建和数值复验交由侧聊，使用各自独立工作树。主聊天负责实际启动后的定时值守。原 Issue249 仍 failed，35/54已独立接受，GPU1另有一个1500步完成待审，`vca` 暂停。原生产检出、产物和当前状态主文件保留。

下一依赖是定位 GPU 输出与梯度差异，决定可审查的数值和成本合同，再验证恢复执行器与旧证据接纳。单步32秒的暂存开销也需纳入预算。任何新真实批次均需先登记并在实际启动后转交定时值守。本聊天2026-11-07交付诊断和交接边界。
