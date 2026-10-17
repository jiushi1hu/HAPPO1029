# 五库 HAPPO：真实来水年度训练

基于 PKU-MARL HARL 的 HAPPO 实现，五库独立 actor，集中式 critic，日尺度，零河道传播时滞。

## 数据与划分

数据直接读取 data/cascade_reservoir_new 下的五个原始 CSV，覆盖 1965—2024 年；默认另读取同目录的 reservoir_head_sensitivity.csv 水头敏感性假设表。

- 验证：2020、2021、2022、2023、2024；默认以五年的平均年度净奖励选择模型，发电量单独报告。关闭软约束惩罚后以平均年发电量选择。
- 最终测试：1998（特丰）、1999（丰）、1970（平）、1997（枯）、2006（特枯）。类别由用户指定。
- 训练：其余 50 年；来水标准化只使用这些训练年份。
- 一个 episode 是一个完整自然年，等概率随机抽年，允许重复。平年 365 天、闰年 366 天；每年按该年初始库容表重置，当前各年均为统一上限库容。

各库使用 categorical 离散策略，默认从 200 个动作索引 k∈{0,…,199} 中选择，归一化为 u=k/199。二次下泄映射为 Q_exec = Q_low + (Q_high - Q_low) × u²。可行下泄区间由五库水量平衡、硬约束及年末库容目标可达性共同计算；同一动作索引对应的实际流量随状态改变，并非 200 个固定流量值。HAPPO 按所选离散索引计算概率；训练时采样，验证和最终测试时选取概率最大的索引。可用 --action-bins 调整离散动作数量。

默认启用分期约束，使用 43 维全系统观测；显式关闭分期约束时使用原 18 维观测。效率为 0.90，并使用 ValueNorm；训练折扣 gamma 可通过配置或 `--gamma` 调整。默认根据日初、日末库容分别插值得到水位，取两者平均水位计算当日净水头；每步按当前净水头重新计算装机容量对应的过机流量上限，超出部分计为弃水。日发电量仍为平均功率乘 24 小时，各库年末库容须回到该年初始库容；环境根据剩余来水约束当日可行下泄区间，保证目标可达。溪洛渡按数据表使用 12600 MW；固定奖励归一化常数为 1624800 MWh。

该变水头模式 `storage_sensitivity` 是显式标记的敏感性假设：典型水头锚定在运行水位区间中点，尾水加损失的合并偏置固定，不代表实测或校准。它引入了库容—水头—发电反馈，但效率仍固定，尚未引入随流量变化的尾水或损失曲线；日均功率限制也不等同于日内瞬时功率限制。假设、缺失资料与来源见同目录 reservoir_head_sensitivity.md。启动信息及评估记录标识模式与假设状态，诊断评估提供各库日水头的年内最小值、均值与最大值。默认参考控制器与 HAPPO 使用相同水头模式和分期硬约束。

## 奖励与软约束成本

默认奖励为 `r_t = E_t / 1624800 - C_target,t - C_rise,t - C_fall,t`。`E_t` 为五库当日总发电量（MWh）；三个成本分别对应日末水位超出软目标带、当日净水位上升超过建议值、当日净水位下降超过建议值。每项先按库计算超出量（米），除以该库全局运行水位范围，再取五库均值并乘权重；目标带内和建议速率以内的成本为零。升降速率按日末与日初水位差计算，不能代表日内瞬时变化。

`harl/configs/envs_cfgs/cascade_reservoir.yaml` 的 `reward_penalties` 控制开关与权重。`target_band_weight`、`daily_rise_weight`、`daily_fall_weight` 初值均为 0.10，是可调整的敏感性情景参数，尚未校准，也不是货币成本。软目标带来自日参考表，升降建议值按水库和统一时期标签读取分期参数表。CSV 的 `weight_flood_safety`、`weight_generation`、`weight_storage`、`weight_downstream_support` 仅作情景元数据；缺少相应损失函数定义，不直接套用为这三个惩罚系数。

硬约束继续由可行动作区间执行，包括水量平衡、库容边界、下泄能力、功率／过机能力、汛限、蓄水截止和年末闭合；不可行时直接报错，不通过付出惩罚来放宽。软目标偏差和建议速率超限可发生，并扣减实际奖励，仍与硬约束违规计数区分。即使硬约束迫使软目标或速率偏离，也如实记录并计费；枯水年不保证软成本为零。

训练 buffer、GAE 和 critic 均使用扣除成本后的日奖励。默认模型选择使用五个验证年未折扣年度净奖励的均值；策略与参考采用完全相同的成本口径。训练历史及评估提供净奖励、发电奖励、总成本、三项成本，评估另提供逐库成本、软超限天数与累计超出米数。各图中的 TWh 仍是实际发电量；`--plot-details` 的奖励图直接绘制净奖励，不再由发电量反算。奖励或权重改变后需要重新训练。

## 运行

```bash
cd /Users/hujingxin/Documents/Code_python/HARL-main
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python examples/train_cascade.py \
  --episodes 400 \
  --eval-interval 10 \
  --seeds 1 2 3 4 5 \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

使用恒定典型水头时添加 `--head-model fixed`，它只关闭变水头；分期约束仍由独立开关控制。添加 `--no-reward-penalties` 可保留分期硬约束而恢复纯发电奖励及按发电量选模，`--reward-penalties` 可显式开启。使用原全局边界和恒库容参考时添加 `--no-seasonal-constraints`，同时恢复 18 维观测并关闭软约束惩罚；此模式不能显式开启依赖分期数据的惩罚。`--seasonal-constraints` 可显式开启分期硬约束。默认模式的补充表缺失、与原始参数／配置不一致时直接报错，不自动退回旧模式。

每个种子各训练 400 个年度 episode。仅运行一次 400 episodes 时使用 --seeds 1。每 10 个 episode 评估五个验证年，包含 episode=0 初始点；所有种子模型选择结束后才进行典型年最终测试。

启动时输出分期开关、观测维数、数据来源状态和参考控制器类型，并检查全部 60 年的物理／截止约束是否可行。随后逐年计算五个验证年的参考发电量并打印其均值；典型年参考只在全部种子模型选择结束后计算，`--skip-final-test` 同时跳过这些参考。

训练结束后弹出一个窗口：左侧显示 2020—2024 五个验证年的平均年发电量学习曲线及多种子标准差，不绘制参考虚线；右侧每组柱子分别比较该典型年选定策略与参考的发电量，多种子策略展示标准差。默认选定策略按净奖励选取，因此其发电量不一定是训练过程中最高的发电量。右侧参考图例默认为 `Seasonal rule reference`；关闭分期约束后才使用 `Constant-storage reference`。绘图检查运行模式、水头模式、分期动作网格和惩罚配置是否一致，拒绝混合对照。图只在窗口显示，程序不导出 PNG 或 PDF；模型、训练记录及结果只在内存中保留，关闭窗口后退出。没有断点恢复或自动保存功能。可用 --no-plot 禁用弹窗。

只检查输入、划分和逐年硬约束可行性，不执行参考轨迹或训练：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python examples/train_cascade.py --check-data-only
```

## 代码

examples/train_cascade.py 为训练入口；examples/plot_cascade.py 为弹窗绘图入口。harl/envs/cascade_reservoir 负责真实数据读取、年度环境、硬约束与发电、冻结策略评估；harl/runners 组织年度采样和原有顺序 HAPPO 更新；harl/algorithms、common、models、utils 保留所需算法与公共组件。配置位于 harl/configs，回归检查位于 tests。

合成数据生成入口、旧版文件读取格式、单个验证／测试年配置、文件导出和可选 LP 输出入口已移除。真实 CSV 保持原样。统一运行时期表用于核对日历，逐库分期边界与截止日期由补充分期表给出。

分期运行已接入数据读取、剩余时域可达性、物理动作执行、环境观测、评估参考和绘图。默认物理配置为 `seasonal_constraints: true`，环境读取四个补充分期 CSV，并在每一步同时执行当日日末边界和剩余时域约束。也可通过 `data.load_seasonal_operation()` 或 `load_seasonal_operation: true` 只读取数据，这两种方式不会单独启用动作约束。输入按日期和水库 ID 对齐，并检查缺失、重复、库容—水位关系、参考节点、闰年及跨表一致性；来源状态原样保留。当前关键参数仍标记为 `scenario_reference_not_official_operating_rule`，日参考为软情景目标；启用约束不代表已核实为官方运行规程。

分期参考控制器每天尝试接近输入表的日末目标库容。它按上游至下游顺序，使用实际已选上游下泄计算期望流量 `Qin + (Vstart - Vtarget)/dt`，限制在环境给出的可行区间内，再反解二次映射并量化到与策略相同的归一化动作网格，最后调用同一物理转移。软目标在枯水期或动作量化后可能无法精确达到；控制器不裁剪库容、不补水、不放宽硬约束。它是可行的规则对照，不是最优年发电量或上界，HAPPO 是否超过它需要实际训练验证。关闭分期约束时保留原恒库容解析参考，不对它施加分期图例。

分期评估记录对策略和参考统一报告 `violation_count`、逐库汛限超量、蓄水截止缺额、年末库容误差、软参考水位平均绝对偏差，以及各库各运行阶段的天数和发电量；软目标偏差不算违规。库容比较沿用物理模块的舍入容差，原始最大超量仍单独报告。参考逐年通过 `REFERENCE_VALIDATION`／`REFERENCE_TEST` 输出，策略记录保留于内存评估历史，诊断模式逐年输出。

`seasonal.storage_bounds(start, stop)` 返回该半开日期索引区间的逐日日末库容下限、上限副本，形状均为 `(days, 5)`。汛限上限自消落截止日的日末起生效，包含汛限结束日；蓄水目标只在截止日日末作为下限，未自动要求之后持续保持该水位。全局 `vmin/vmax` 不变；日参考轨迹、软目标带、建议升降速率与奖励权重均不作为硬约束，也不改变训练来水标准化。

```python
seasonal = data.load_seasonal_operation()
start, stop = data.indices(year, year)
lower, upper = seasonal.storage_bounds(start, stop)
target = data.initial_storage(year)
limits = TerminalReleaseBounds(
    data, 86400., target, data.external[start:stop], target,
    storage_lower=lower, storage_upper=upper,
)
```

`TerminalReleaseBounds` 从 `harl.envs.cascade_reservoir.terminal_storage` 导入。逐日约束可包含中途消落和蓄水截止日；`target=None` 表示不增加年末等式。只有全局恒定边界且满足原条件时才使用原快速可达性公式；分期边界采用 LP，年末舍入容差不会放宽分期边界。分期路径逐日求解剩余时域 LP，运行明显慢于旧快速路径；当前机器单个完整年度转移约需数十秒，训练及多次验证均有此开销，首次运行宜先用少量 episodes 验证耗时。

开启分期约束后，物理模块使用当日上下界计算向下游兼容的下泄区间，再与完整剩余时域的可行区间相交。消落截止日允许日初水位高于当日日末汛限，通过真实下泄满足截止要求，不裁剪库容。环境还检查前一日日末约束；无年末等式的 `terminal_storage_mode: free` 也会保留中途分期约束。转移记录和逐库 info 给出当日边界、汛限／蓄水截止标识和逐库运行阶段；阶段按各库截止日区分消落、汛限控制、蓄水和蓄水完成后运行，不直接套用统一的四季标签。

43 维观测的前 18 项与原模式相同，后 25 项依次为：当日库容下限比例（5）、上限比例（5）、下一硬截止点的剩余天数／366（5）、该截止点的库容下限比例（5）、上限比例（5）。库容比例仍按全局 `vmin/vmax` 归一化；无剩余截止点时天数为 -1。年末回到初始库容也作为截止点；不添加未来来水。软参考表与建议速率按已知日期查表，不增加观测维数。actor、critic 和 buffer 根据环境空间自动使用新维数，开启此模式应重新训练策略。

## 检查

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests /opt/anaconda3/envs/harl/bin/python -m unittest test_cascade_calendar test_cascade_reservoir test_seasonal_operation test_terminal_reachability test_cascade_reward_penalties -v
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

终端的 DIAG_UPDATE 包含最近一个验证间隔内的 critic 对完整采样回报与 GAE 目标的拟合、动作分布 KL、概率比、裁剪比例和二次映射下的动作统计。critic 的 MC 误差使用当前 gamma 计算完整轨迹的折扣回报，不含 critic bootstrap，也不是准确期望价值；GAE 目标来自 critic，不能用拟合 GAE 良好来替代完整回报检验。启用惩罚时，`predicted_start_return_reward_units` 和 `realized_sampled_return_reward_units` 保持奖励单位，不转换为 TWh；`last_episode_penalty` 和 `last_episode_penalty_components` 给出总成本及分项。仅关闭惩罚时保留 `predicted_start_return_TWh` 和 `realized_sampled_return_TWh`；它们是按当前 gamma 加权的回报换算值，gamma<1 时也不是未折扣年发电量。`last_episode_reward` 为实际日奖励直接求和，`last_discounted_episode_reward` 为折扣求和，`last_episode_energy_TWh` 为实际年发电量。梯度范数仍是裁剪前数值。

DIAG_VALIDATION 默认在第 0、20、100、150、400 轮及本次最后一轮输出。它列出逐年发电量、各库弃水体积、期末库容占可用库容比例、平均动作，以及三次随机策略评估的平均结果；启用惩罚时另报告确定性／随机评估的年度净奖励和逐年成本明细。随机评估使用隔离并恢复的 RNG 状态，不影响后续训练采样；三次评估的波动是动作采样不确定性，不是跨训练种子的标准差。

离散策略下，随机训练的平均映射位置为 Σ p(k)×(k/199)²，确定性评估的映射位置为 (argmax p(k)/199)²，两者可能不同。诊断分别报告这两项统计，本身不更改评估方式。

诊断仍不保存模型、日志、图或新增报告文件；需要保留控制台内容时可自行复制。

## 折扣与 GAE 回报对照

`--gamma 0.99` 或 `--gamma 1` 可以覆盖 `harl/configs/algos_cfgs/happo_cascade.yaml` 的 `algo.gamma`；未指定时读取配置，当前配置为 0.99。支持有限数值 0≤gamma≤1，非法值在训练前报错。启动输出显示实际 gamma 和 gae_lambda。gamma 只影响训练回报及对应 critic 诊断，不折扣评估中的发电量或软成本总量；默认验证选模使用未折扣年度净奖励，关闭惩罚后使用未折扣年发电量。gamma<1 会减弱对远期收益和成本的权重；gamma=1 对应未折扣年度目标。

`--gae-lambda 1` 是可选对照设置。完整年度终止时，λ=1 使每步价值目标等于从该步到年末、按所选 gamma 计算的完整实际回报；gamma=1 时即为未折扣累计回报。该设置消除未来价值 bootstrap 的依赖，但采样方差可能增大。未指定时继续使用原配置 0.95。

GAE λ 无论来自 YAML、命令行还是直接构造 runner，均须为 [0, 1] 内的有限数值。种子须为 [0, 4294967295] 内的整数，布尔值和浮点种子会被拒绝；种子列表仍须非空且不重复。命令行先覆盖 YAML，再校验最终生效值；非法参数在读取数据、可行性检查和参考计算前报错。
