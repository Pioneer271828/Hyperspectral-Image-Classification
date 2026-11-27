C9 / Issue529：WP02、WP05独立验证条件核查。C独占写入本目录；基于C8
`c6526763b874df55fab4991e3076a235adcac174`，不改C6–C8冻结实现或全仓状态。

本目录是现有清单的证据补充和论文表投影，不是新读取器或训练框架。
`reuse/`逐字节复用C7去身份准入回执、D的30方向表以及SpectralEarth产品/类别
核查。`reuse-pins.json`固定其字节，不把旧表中未更新的类别/准入字段当成当前结论。
`original30-current-qualification.csv`保持全部旧单元格并添加C9限定列：只有
徐州↔洪湖两方向已有C7共同两类开发输入，独立确认方向仍为零。

`candidates.json`覆盖17个既有产品/候选记录，`comparisons.json`登记16条论文比较。
公开影像/patch总数、已选parent数、独立地块和独立采集不互相替代。未知写null；
已释放独立确认单元0仅表示项目未准入，不断言候选数据实际没有可用地块。
比较表的`project_comparison_task_points_admitted`只计该共同任务已冻结的监督点，
不抹掉C5已合法准入的3幅图像/mask和1条序列，也不把公开patch总数当成监督预算。

执行仅限授权LXC，使用干净、已发布完整SHA：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lbx/hsi-pioneer271828/envs/hsi/bin/python \
  -m unittest discover -s research/paper_c_independent_validation -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 /home/lbx/hsi-pioneer271828/envs/hsi/bin/python \
  research/paper_c_independent_validation/build_tables.py \
  --expected-commit <published-full-SHA> --out <fresh-external-directory>
```

`audit_honghu_metadata.py`可另从本机已驻留、指纹固定的C6迁移包中只提取JSON，
复用C6 ReleasedBank及C7原选择核对洪湖parent/类别计数；不提取/解释任何NPY。
该结果只补充地理单位数量，不重新分组、放行新角色或证明地块独立。

所有原始点ID及私有空间元数据只留LXC；公开输出只含计数、来源及缺口。
新影像/GT下载预算0，真实拟合0，query0，研究训练持续暂停。
