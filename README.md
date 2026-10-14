# 五库 HAPPO：真实来水年度训练

基于 PKU-MARL HARL 的 HAPPO 实现，五库独立 actor，集中式 critic，日尺度，零河道传播时滞。

## 数据与划分

数据直接读取 data/cascade_reservoir_new 下的五个 CSV，覆盖 1965—2024 年。

- 验证：2020、2021、2022、2023、2024；以五年的平均年发电量选择模型。
- 最终测试：1998（特丰）、1999（丰）、1970（平）、1997（枯）、2006（特枯）。类别由用户指定。
- 训练：其余 50 年；来水标准化只使用这些训练年份。
- 一个 episode 是一个完整自然年，等概率随机抽年，允许重复。平年 365 天、闰年 366 天；每年按该年初始库容表重置，当前各年均为统一上限库容。

二次下泄映射为 Q_exec = Q_low + (Q_high - Q_low) × u²，0≤u≤1。可行下泄区间由五库水量平衡及硬约束计算；HAPPO 仍按归一化动作 u 计算概率。保留当前 18 维全系统观测、恒定水头、效率 0.90、期末库容自由、gamma=1 和 ValueNorm。溪洛渡按数据表使用 12600 MW；固定奖励归一化常数为 1624800 MWh。

## 运行

```bash
cd /Users/hujingxin/Documents/Code_python/HARL-main
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python examples/train_cascade.py \
  --episodes 400 \
  --eval-interval 10 \
  --seeds 1 2 3 4 5 \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

每个种子各训练 400 个年度 episode。仅运行一次 400 episodes 时使用 --seeds 1。每 10 个 episode 评估五个验证年，包含 episode=0 初始点；所有种子模型选择结束后才进行典型年最终测试。

训练结束后弹出一个窗口：左侧显示五个验证年的平均年发电量学习曲线，右侧显示五个典型年的最终测试发电量。多种子时展示标准差。图只在窗口显示，程序不导出 PNG 或 PDF；模型、训练记录及结果只在内存中保留，关闭窗口后退出。没有断点恢复或自动保存功能。可用 --no-plot 禁用弹窗。

只检查输入和划分：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python examples/train_cascade.py --check-data-only
```

## 代码

examples/train_cascade.py 为训练入口；examples/plot_cascade.py 为弹窗绘图入口。harl/envs/cascade_reservoir 负责真实数据读取、年度环境、硬约束与发电、冻结策略评估；harl/runners 组织年度采样和原有顺序 HAPPO 更新；harl/algorithms、common、models、utils 保留所需算法与公共组件。配置位于 harl/configs，回归检查位于 tests。

合成数据生成入口、旧版文件读取格式、单个验证／测试年配置、文件导出和可选 LP 输出入口已移除。真实 CSV 保持原样。运行时期表仅用于读取时核对日历，不据此自动添加分期运行约束。

## 检查

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests /opt/anaconda3/envs/harl/bin/python -m unittest test_cascade_calendar test_cascade_reservoir -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests MPLCONFIGDIR=/private/tmp/happo-cascade-matplotlib /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m unittest test_cascade_plot -v
```

这些检查只执行少量小网络更新；不会运行完整 400 episodes，也不会打开真实窗口。保留原 HARL 来源与安装信息于 setup.py。

## 诊断模式

`--diagnostics` 只增加前向诊断和终端输出，不修改网络、学习率、奖励、年度抽样或优化步骤；检查保证观察诊断不改变训练的随机状态。`--skip-final-test` 可在排查时跳过典型年测试。

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python examples/train_cascade.py \
  --episodes 160 --eval-interval 10 --seeds 1 \
  --diagnostics --skip-final-test \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

终端的 DIAG_UPDATE 包含最近一个验证间隔内的 critic 对完整采样回报与 GAE 目标的拟合、动作分布 KL、概率比、裁剪比例和二次映射下的动作统计。critic 的 MC 误差是对这条随机采样轨迹的实际累计回报计算，不是准确期望价值；GAE 目标来自 critic，不能用拟合 GAE 良好来替代完整回报检验。梯度范数仍是裁剪前数值。

DIAG_VALIDATION 默认在第 0、20、100、150、400 轮及本次最后一轮输出。它列出逐年发电量、各库弃水体积、期末库容占可用库容比例、平均动作，以及三次随机策略评估的平均结果。随机评估使用隔离并恢复的 RNG 状态，不影响后续训练采样；三次评估的波动是动作采样不确定性，不是跨训练种子的标准差。

二次映射使 E[u²]=E[u]²+Var(u)，因此随机训练的平均映射位置与评估均值动作的映射位置不同。诊断用于判断该差异是否与退化相关，本身没有更改评估方式。

诊断仍不保存模型、日志、图或新增报告文件；需要保留控制台内容时可自行复制。

## Beta 探索集中度对照

`--beta-concentration-offset C` 在原 Beta 参数 α、β 上增加总集中度 C，同时保持同一网络权重下的均值函数：m=α/(α+β)，α_new=α+C×m，β_new=β+C×(1−m)。因此均值仍为 m，而方差由 m(1−m)/(α+β+1) 变为 m(1−m)/(α+β+C+1)。采样和 HAPPO 概率比都使用新分布，不对动作样本另行截断。

未指定时 C=0，精确保留原分布。C=98 是探索方差对照参数，总集中度至少为 100，标准差始终小于 0.05；它不是已证明最优的超参数，也不改变二次下泄映射。

对照运行时，在原命令增加 `--beta-concentration-offset 98`。先在验证集比较，再确定是否进行典型年最终测试；每次从头训练，不能把新运行直接拼接到旧曲线。

`--gae-lambda 1` 是另一项独立对照设置。gamma=1 且完整年度终止时，λ=1 使价值目标等于该条年度轨迹的完整实际回报，减少对短期 bootstrap 的依赖，但采样方差可能增大。未指定时继续使用原配置 0.95。两个选项可分别或组合比较，不改变验证／测试划分。
