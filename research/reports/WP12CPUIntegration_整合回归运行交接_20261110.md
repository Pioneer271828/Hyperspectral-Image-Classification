# WP12 CPU整合回归：运行中断连交接（不是验收报告）

Issue361，对应实施计划§19.13。用户要求连续推进；当前遇到授权LXC连接关闭，不能伪称回归结束。未创建自动化，未重启测试或真实训练。

- 实际冻结tested源码：`5d12d81ed77ab6a85d9bbabfdcbfdf0db57c6477`。
- 容器：`lbx@10.69.37.243:1222`，已核lxc；Python3.10.20，解释器 `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`。
- 干净独立checkout：`/home/lbx/hsi-pioneer271828/checkouts/wp12-cpu-integration01`。运行期间不可修改。
- 命令：在该checkout中以CPU单线程、`CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1`、`PYTHONPATH=.:tests`运行 `python /home/lbx/hsi-pioneer271828/runs/wp12-cpu-integration01.py`。该外部脚本执行全部unittest discovery，不过滤测试。
- 启动记录UTC：2026-11-10T07:21:38Z；PID `1019307`。发现1125项，清单位于 `runs/wp12-cpu-integration01/inventory.json`，启动记录 `started.json`，完整结束才写 `receipt.json`。
- 输出日志：`/home/lbx/hsi-pioneer271828/runs/wp12-cpu-integration01.log`。
- 最后可读记录：课程CLI三类别/三臂的连续与恢复比较仍在运行，子进程执行合成24步恢复；尚未观察到FAIL/ERROR，**不表示剩余测试通过**。当前父进程是否存活、测试是否结束均无法确认。
- 原SSH会话随后被远端关闭；三次独立重连（包含直接只读命令）均在认证前被 `Connection closed by 10.69.37.243 port 1222` 拒绝。认证信息没有写入此文件或日志。

恢复连接后，先只读检查原PID、日志和receipt；若仍运行继续等待，若完成按完整回执核对1125项及失败/跳过，若异常结束保留所有产物诊断。不要凭断连认定测试失败或重新启动一份。最终源树和输入必须保持不变。

本地交接工作树 `F:/E2E_VCA_worktrees/wp12-cpu-integration`、分支 `codex/wp12-cpu-integration`，基准 `87636efbc92065d85195f8129f9219a05f398acf`；该基准相对实测源码只增加Issue359已接受文档。Issue357/359已分别完成CPU工程验收并合入PR358/360。主入口92个已有tracked修改均保留。本交接没有修改当前状态JSON或生成投影，避免在无法运行远程投影时提交不一致状态。

真实长训练未启动；8张卡最后观察均不足12288MiB空闲门槛。GPU回归、真实研究效果与后续协议缺口继续未完成。不得把本文件作为WP12完成、全仓通过、CI通过或新增分类收益的证据。
