# WP11 徐州无标签留出候选观测准备

Issue #375，接续 #373 的真实几何预检。复用此前通过的影像空间预检、原始影像哈希、固定空间计划与开发角色记录。入口只读取原 MATLAB 文件的 `xuzhou` 影像变量，并按未使用验证角色裁出候选。每个 FP32 光谱块保留原边界、形状及文件哈希。角色尚未分配。预检和导出均无需原始标签。

```sh
python scripts/prepare_wp11_held_aside.py --expected-commit <commit> \
  --registration <image-only-registration.json> --registration-sha256 <sha256> \
  --out <new-external-directory>
```

授权 LXC 实测源码 `8c95cf57892be8470687044a5687419194d5f1f8`，分支 `codex/wp11-confirmation`。CPU 相关回归 41/41，5.167 秒，仓库检查 598 Python / 198 JSON。专项验证切块逐值一致、MATLAB 同文件内标签变量未请求、影像身份不符拒绝。实际导出 25 个徐州候选，独立回读重切核对 8,399,104 个光谱值完全一致，五份输入哈希保持不变。真实更新及 query 均为 0。

LXC 清单 `runs/wp11-xuzhou-held-aside01/candidate-observations.json`，SHA256 `f73ce2d5152ef4d18f08ec1359c2c7ecce9edc4df849ea2fd84c4e51d7e0c370`。可将通过后续准入的条目绑定到确认 CLI。实际角色、类别覆盖和标签授权仍须单独验收。最新 GPU 检查最高空闲 6000 MiB，低于 12288 MiB 门槛，既定 54 例常驻成本继续等待资源。WP04 真实短测后补。
