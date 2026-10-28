# 远程基线验收

Windows 仅执行编辑、文件检查、Git 和 SSH/SCP；下面的 Python、pip 和测试命令均在 `ssh rmvision` 登录后的训练容器执行。首次确认 `hostname`、`pwd`、`/.dockerenv`、GPU 和运行环境，复用代码前检查 `git status --short --branch`。

## 代码同步

容器有仓库权限时，直接从目标 GitHub 克隆或 fetch 到独立分支；运行前核对完整 SHA 与干净工作树。不要覆盖别人的未提交工作，不在运行中更新 checkout。

目前本机已有目标私有仓库权限，容器尚无 GitHub 凭据。可先在本机提交并推送模块分支，核对 `git ls-remote github <分支>`，随后用 `git bundle create <文件> <分支>` 打包完整 Git 历史，通过 SCP 传入容器。容器用 `git bundle verify` 验证并 clone；将 origin 设置为目标 GitHub URL，保留 bundle 来源记录。已有 checkout 用 bundle fetch 到新分支，不使用强制覆盖。此方式传输的仍是已推送提交，没有复制凭据，也不能称为容器直接 GitHub 拉取成功。

后续有未推送修改时，先提交并同步，或在独立工作树记录补丁/源码指纹后做开发检查。正式基线验收要求提交且工作树干净，不能测试旧代码替代当前修改。

## 本次远程环境

容器的 Python 3.12.3、预装 NVIDIA PyTorch/CUDA 可用。保持预装框架，新建 `/home/rmvision/venvs/hsi-baseline`（`--system-site-packages`），仅补充缺失依赖。以下是本次执行过的安装方式：

```bash
python -m venv --system-site-packages /home/rmvision/venvs/hsi-baseline
/home/rmvision/venvs/hsi-baseline/bin/python -m pip install scikit-image==0.24.0 spectral==0.24 h5py==3.12.1
```

历史 `requirements.txt` 中的 `numpy==1.24.3` 对应旧环境。本次验证沿用容器已有 NumPy 1.26.4，不声称完成该依赖文件的全新安装复现；实际版本随 manifest 保存。本模块不重建通用环境或升级预装 PyTorch。

## 验收入口

在代码根目录执行，将参数替换为本次预期完整 SHA 和未使用过的产物目录：

```bash
/home/rmvision/venvs/hsi-baseline/bin/python scripts/validate_baseline.py \
  --expected-commit <完整40位SHA> \
  --out-dir /workspace/hsi-artifacts/<run_id>
```

入口检查 Linux 容器、HEAD、干净工作树、冻结源码指纹、输出目录隔离和断言开启状态，然后运行：

```bash
python -m unittest discover -s tests -p test_interaction_modules.py -v
python smoke_test.py --device cpu
python smoke_test.py --device cuda
```

CUDA 不可用会报错；`--skip-cuda` 仅用于明确标注的 CPU 验收。每条检查有独立日志和超时，manifest 保存实际命令、时长、包版本及失败状态；相同输出目录不能复用。运行后再次核对代码未变。这个入口专用于冻结基线；后续模型模块应从已命名基线建立相关回归，而非改写其原始指纹。

合成测试确认当前链路、梯度与序列化能工作，不产生真实数据 OA/AA/Kappa 或跨传感器结论。新研究模块仍需对应真实协议、预算控制和消融。
