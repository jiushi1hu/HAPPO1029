# 五库 HAPPO：真实来水年度训练

基于 PKU-MARL HARL 的 HAPPO 实现，五库独立 actor，集中式 critic，日尺度，零河道传播时滞。

## 数据与划分

数据直接读取 data/cascade_reservoir_new 下的五个原始 CSV，覆盖 1965—2024 年；默认另读取同目录的 reservoir_head_sensitivity.csv 水头敏感性假设表。

两项服务约束的数据按 Xie 等（2026，DOI: 10.1016/j.jhydrol.2026.136160）Table A.1 接入 `reservoir_service_requirements.csv`：按 WDD/BHT/XLD/XJB/THR 顺序，保证出力为 `[2290.0, 4567.5, 5345.7, 2750.0, 4990.0]` MW，生态流量为 `[900, 1160, 1160, 1200, 6000]` m³/s。论文 TGR 显式映射为本仓库 THR，数值固定用于各日和各年。默认 `load_service_requirements: true` 会加载、校验并在启动信息中报告阈值、来源及 SHA-256；开启服务惩罚时也必须加载这张表。单位、语义和来源见同目录 `reservoir_service_requirements.md`。

- 验证：2020、2021、2022、2023、2024；以五年的平均折扣净奖励选择模型，gamma 与训练一致，发电量单独报告。关闭软约束惩罚后仍以同一 gamma 加权的发电奖励选择模型。
- 最终测试：1998（特丰）、1999（丰）、1970（平）、1997（枯）、2006（特枯）。类别由用户指定。
- 训练：其余 50 年；来水标准化只使用这些训练年份。
- 一个 episode 是一个完整自然年，等概率随机抽年，允许重复。平年 365 天、闰年 366 天；每年按该年初始库容表重置，当前各年均为统一上限库容。

默认每次 HAPPO 更新先冻结 actor、critic 和 ValueNorm，顺序采集 `rollout_episodes: 6` 个独立年度轨迹，再合并有效数据更新一次。每个年度仍独立重置环境、计算 GAE 和从 `gamma**0` 开始折扣；闰日不填充、不丢弃，跨年 terminal mask 阻断 bootstrap 与回报串联。年度均匀抽样允许同一批重复年份，且只从训练划分抽取。这是批量采集，不是新增六个并行进程。

`--episodes` 仍表示每个种子的年度 episode 总数；`--eval-interval` 现在明确以 **HAPPO 更新轮数** 为单位。默认 `300 / 6 = 50` 轮、每轮验证一次；不足六个年度的末批也会更新。`--rollout-episodes 1` 可恢复旧的一年一更新基线。相同采样步数下，六年一批会减少更新轮数；这是一项待比较的稳定性实验，不保证收敛。其他网络、奖励、阈值、优化器、学习率、gamma 和 GAE lambda 没有随本次批量采集改变。

各库使用 categorical 离散策略，默认从 200 个动作索引 k∈{0,…,199} 中选择，归一化为 u=k/199。二次下泄映射为 Q_exec = Q_low + (Q_high - Q_low) × u²。可行下泄区间由五库水量平衡、硬约束及年末库容目标可达性共同计算；同一动作索引对应的实际流量随状态改变，并非 200 个固定流量值。HAPPO 按所选离散索引计算概率；训练时采样，验证和最终测试时选取概率最大的索引。可用 --action-bins 调整离散动作数量。

默认启用分期约束，各 actor 使用自己的 16 维局部及直接邻库观测，集中式 critic 单独使用 43 维系统状态；关闭分期约束时分别为 11 维与 18 维。五个 actor 不再接收同一全系统向量，训练、验证、最终测试及诊断评估使用相同的局部输入接口。效率为 0.90，并使用 ValueNorm；训练折扣 gamma 可通过配置或 `--gamma` 调整。默认根据日初、日末库容分别插值得到水位，取两者平均水位计算当日净水头；每步按当前净水头重新计算装机容量对应的过机流量上限，超出部分计为弃水。日发电量仍为平均功率乘 24 小时，各库年末库容须回到该年初始库容；环境根据剩余来水约束当日可行下泄区间，保证目标可达。溪洛渡按数据表使用 12600 MW；固定奖励归一化常数为 1624800 MWh。

该变水头模式 `storage_sensitivity` 是显式标记的敏感性假设：典型水头锚定在运行水位区间中点，尾水加损失的合并偏置固定，不代表实测或校准。它引入了库容—水头—发电反馈，但效率仍固定，尚未引入随流量变化的尾水或损失曲线；日均功率限制也不等同于日内瞬时功率限制。假设、缺失资料与来源见同目录 reservoir_head_sensitivity.md。启动信息及评估记录标识模式与假设状态，诊断评估提供各库日水头的年内最小值、均值与最大值。默认参考控制器与 HAPPO 使用相同水头模式和分期硬约束。

## 奖励与软约束成本

默认团队奖励为 `r_t = E_t / 1624800 - C_guaranteed_output,t - C_ecological_flow,t`。`E_t` 为五库当日总发电量（MWh）；保证出力缺额为 `max(P_G - P_actual, 0) / P_G`，生态缺流为 `max(Q_E - Q_release, 0) / Q_E`。每项在固定的有效水库集合上取均值并乘其权重；本表两项均为五库有效。零阈值显式关闭相应水库的该项要求，分母改为该项有效水库数量，全部关闭时该项为零。达标时惩罚为零；总奖励允许负值。只忽略相对阈值 `1e-10` 以下的浮点舍入缺额，计费与违规天数使用同一容差。

`harl/configs/envs_cfgs/cascade_reservoir.yaml` 的 `reward_penalties` 控制开关与权重。`guaranteed_output_weight`、`ecological_flow_weight` 默认均为 `1.0`，是预设权衡参数，不是货币成本。保证出力按实际执行的日均功率计算，生态流量按实际总下泄（过机加弃水）计算；日尺度达标不等于全天瞬时达标。旧水位目标带、日升速和日降速不再计入奖励，旧权重键会报错以防误用。分期 CSV 中的目标曲线仍供规则参考控制器使用，其余软建议和权重仅保留为数据元信息。本奖励沿用论文的相对缺额和线性扣罚形式，采用本仓库统一团队奖励；没有额外叠加论文的局部／全局发电混合项、航运或动作修正成本。

硬约束继续由可行动作区间执行，包括水量平衡、库容边界、下泄能力、功率／过机能力、汛限、蓄水截止和年末闭合；不可行时直接报错，不通过付出惩罚来放宽。保证出力和生态流量在本实现中为软要求，即使受硬约束或枯水影响无法满足，也如实记录缺额并扣罚，不添加出力／流量硬下限。关闭分期硬约束不会关闭这两项服务惩罚。

训练 buffer、GAE 和 critic 均使用扣除成本后的同一日奖励，五个 actor 共用该团队目标。默认模型选择使用五个验证年按训练 gamma 加权的年度净奖励均值；策略、分期规则参考和恒库容参考采用相同阈值、折扣与成本口径。逐库及全系统另报告未折扣的达标率、缺额天数、最大相对缺额、出力缺额电量（MWh）与生态缺水体积（m³）。各图中的 TWh 仍是实际发电量。本次保留 `gamma=0.99`；如研究目标为等权年度净奖励，可显式设 `--gamma 1`。服务惩罚并不保证零缺额，奖励或权重改变后需要重新训练。

## 运行

```bash
cd /Users/hujingxin/Documents/Code_python/HARL-main
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python examples/train_cascade.py \
  --episodes 300 \
  --rollout-episodes 6 \
  --eval-interval 1 \
  --seeds 1 2 3 4 5 \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

Custom subsets are supported with `--training-years`, `--validation-years`, and `--test-years`. For example, use `--training-years 2007 2008 2009 2010 2011 --validation-years 2012 2013 --test-years 2014 --seeds 1 2 3` with the command above, and omit `--fixed-training-year`. All three lists must be nonempty, unique, within 1965--2024, and mutually disjoint. Explicit training years leave unassigned years unused; inflow normalization is fitted only on the selected training years. Episodes sample from that list, checkpoint selection uses the mean reward over the selected validation years, and the selected actors are tested only after every seed finishes training. Custom CLI test years receive neutral labels such as `test_2014`. With no year-list overrides, the original 50/5/5 partition and wet/dry labels remain unchanged. A YAML `experiment.training_years` list has the same meaning; `null` keeps the complement-of-holdouts default. YAML test labels, when provided, must match the test-year list.

使用恒定典型水头时添加 `--head-model fixed`，它只关闭变水头；分期约束仍由独立开关控制。添加 `--no-reward-penalties` 可恢复纯发电奖励及按同一 gamma 加权的发电奖励选模，`--reward-penalties` 可显式开启两项服务惩罚。使用原全局边界和恒库容参考时添加 `--no-seasonal-constraints`，此时 actor 使用 11 维局部输入、critic 使用 18 维系统状态，服务惩罚仍由独立开关控制。`--seasonal-constraints` 可显式开启分期硬约束。默认模式的补充表缺失、与原始参数／配置不一致时直接报错，不自动退回旧模式。

每个种子各训练 300 个年度 episode，对应 50 次 HAPPO 更新。仅运行一个种子时使用 `--seeds 1`。每轮更新后评估五个验证年，包含更新前的 episode=0 初始点；所有种子模型选择结束后才进行典型年最终测试。旧命令若显式写了 `--eval-interval 10`，会变成每十轮更新验证一次；本实验请改为 `--eval-interval 1`。新实验须使用新的日志文件名，已经启动的进程不会切换到新采集方式。

启动时输出分期开关、观测维数、数据来源状态和参考控制器类型，并检查选定训练、验证、测试年的物理／截止约束是否可行（默认共 60 年；显式子集仅检查选中的年份）。随后逐年计算验证年的参考发电量并打印其均值；典型年参考只在全部种子模型选择结束后计算，`--skip-final-test` 同时跳过这些参考。

训练结束后弹出结果窗口：左侧显示 2020—2024 五个验证年的平均年发电量学习曲线及多种子标准差，不绘制参考虚线；右侧每组柱子分别比较该典型年选定策略与参考的发电量，多种子策略展示标准差。默认选定策略按净奖励选取，因此其发电量不一定是训练过程中最高的发电量。右侧参考图例默认为 `Seasonal rule reference`；关闭分期约束后才使用 `Constant-storage reference`。绘图检查运行模式、水头模式、分期动作网格和惩罚配置是否一致，拒绝混合对照。图只在窗口显示，程序不导出 PNG 或 PDF；模型不保存，训练记录默认保留在内存和控制台，也可显式指定 `--episode-log` 保存逐回合 JSONL。没有断点恢复功能。可用 --no-plot 禁用弹窗。

无论是否启用软约束惩罚，都会自动打开 `HAPPO validation net reward` 窗口：横轴为训练 episode，纵轴为五个验证年按训练 gamma 加权的平均年度净奖励。细线表示各训练种子，粗线表示种子均值，阴影表示种子间 ±1 标准差（不是置信区间）；单种子不绘制阴影。该图包含 episode=0，直接使用各验证点记录的 `mean_reward`，保留真实波动，不绘制历史最优包络线，也不从发电量反算。此图不依赖 `--diagnostics` 或 `--plot-details`，无需额外评估。另外自动打开奖励分项和惩罚占发电奖励比例两个窗口，详见下文。`--plot-details` 另开训练奖励、训练发电量和优化诊断三个窗口；关闭惩罚时以折扣发电奖励选模，分项图显示实际为零的惩罚。

只检查输入、划分和逐年硬约束可行性，不执行参考轨迹或训练：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python examples/train_cascade.py --check-data-only
```

## 代码

examples/train_cascade.py 为训练入口；examples/plot_cascade.py 为弹窗绘图入口。harl/envs/cascade_reservoir 负责真实数据读取、年度环境、硬约束与发电、冻结策略评估；harl/runners 组织年度采样和原有顺序 HAPPO 更新；harl/algorithms、common、models、utils 保留所需算法与公共组件。配置位于 harl/configs，回归检查位于 tests。

合成数据生成入口、旧版文件读取格式、单个验证／测试年配置、模型／图片导出和可选 LP 输出入口已移除；逐回合日志支持按需导出 JSONL。自然入流、物理参数及硬运行边界保持原样。统一运行时期表用于核对日历，逐库分期边界与截止日期由补充分期表给出。

分期运行已接入数据读取、剩余时域可达性、物理动作执行、环境观测、评估参考和绘图。默认物理配置为 `seasonal_constraints: true`，环境读取四个补充分期 CSV，并在每一步同时执行当日日末边界和剩余时域约束。也可通过 `data.load_seasonal_operation()` 或 `load_seasonal_operation: true` 只读取数据，这两种方式不会单独启用动作约束。输入按日期和水库 ID 对齐，并检查缺失、重复、库容—水位关系、参考节点、闰年及跨表一致性；来源状态原样保留。当前关键参数仍标记为 `scenario_reference_not_official_operating_rule`，日参考为软情景目标；启用约束不代表已核实为官方运行规程。

分期参考控制器每天尝试接近输入表的日末目标库容。它按上游至下游顺序，使用实际已选上游下泄计算期望流量 `Qin + (Vstart - Vtarget)/dt`，限制在环境给出的可行区间内，再反解二次映射并量化到与策略相同的归一化动作网格，最后调用同一物理转移。软目标在枯水期或动作量化后可能无法精确达到；控制器不裁剪库容、不补水、不放宽硬约束。它是可行的规则对照，不是最优年发电量或上界，HAPPO 是否超过它需要实际训练验证。关闭分期约束时保留原恒库容解析参考，不对它施加分期图例。

日参考已保留 THR 的 2006、2011、2022 年和 XJB 的 2022 年编辑轨迹。
这些年份通过可选的 `reservoir_reference_year_nodes.csv` 明确定义逐日边界节点；
其他库／年继续使用循环参考节点。默认分期汇总表描述循环路径，覆盖年份的
分期起止目标则取自年度节点。日参考必须与对应节点、水位—库容曲线及派生字段
一致，否则仍然拒绝启动。软带的上下端点均限制在运行最低水位和参考汛期上限
之间，防止编辑目标超出汛限时出现倒置区间；硬汛限与蓄水截止约束仍独立执行。
年度覆盖文件参与数据哈希，并在启动元数据的 `reference_year_overrides` 中列出。
格式、来源状态和本次 2 m³ 数值修正在
[年度参考说明](data/cascade_reservoir_new/reservoir_reference_year_nodes.md) 中记录。

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

局部 actor 观测布局标记为 `local_neighbors_v1`，按如下顺序构造：自身库容比例、当日外部来水标准化值、前一日已执行下泄／本库下泄上限（3）；直接上游的库容比例、当日外部来水标准化值、前一日已执行下泄比例（3）；直接下游的库容比例、距当日日末库容上限的剩余空间比例（2）；自身当日上下限、下一硬截止点剩余天数／366、该截止点上下限（5，仅分期模式）；年内相位 sin/cos 和本 episode 剩余天数／366（3）。合计分期模式 16 维，非分期模式 11 维。下游剩余空间按其全局库容范围归一化，在必须消落的日期允许为负；无上游时三项为 `[0,0,0]`，末库无下游时两项为 `[0,1]`。无剩余截止点时天数为 -1；年末回到初始库容也作为截止点。

各 actor 只使用本库、直接邻库和共同日历信息，不输入远端水库状态、系统汇总值、未来实际来水或软参考轨迹。本接口沿用“决策时已知当日外部日均来水”的既有模拟假设；五库动作在同一天开始时共同选择，因此上游下泄只能用前一天的已执行值，不能提前使用当天尚未选择的上游动作。邻库信息需要在决策前共享。本设计借鉴论文的局部／邻库结构，但不是论文 22 维观测的逐项复现，也没有虚构来水预报。

集中式 critic 保留原系统状态：前 18 项为五库库容、外部来水、前一日下泄与三项时间信息，分期模式后加五库的当日上下限、下一截止点时间与上下限（25）。环境以 `actor_observations()` 返回 `(5,16)` 或 `(5,11)`，以 `state()` 返回 `(43,)` 或 `(18,)`；`reset/step` 的 actor observation 与共享 critic state 分别返回各自空间。启动信息、逐回合 JSONL 和策略评估记录标注布局、actor 维数及 critic 维数，以区分旧全局观测实验。旧 actor 输入形状不兼容，需要重新训练并使用新的日志文件名。本次观测调整不改变奖励、年度抽样、HAPPO 更新、物理约束或仍使用实际未来来水的可行性计算。

## 检查

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests /opt/anaconda3/envs/harl/bin/python -m unittest test_cascade_calendar test_cascade_reservoir test_seasonal_operation test_terminal_reachability test_cascade_reward_penalties -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests MPLCONFIGDIR=/private/tmp/happo-cascade-matplotlib /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m unittest test_cascade_plot -v
```

这些检查只执行少量小网络更新；不会运行完整 300 episodes，也不会打开真实窗口。保留原 HARL 来源与安装信息于 setup.py。

## 诊断模式

`--diagnostics` 只增加前向诊断和终端输出，不修改网络、学习率、奖励、年度抽样或优化步骤；检查保证观察诊断不改变训练的随机状态。`--skip-final-test` 可在排查时跳过典型年测试。

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python examples/train_cascade.py \
  --episodes 160 --eval-interval 10 --seeds 1 \
  --diagnostics --skip-final-test \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

终端的 DIAG_UPDATE 包含最近一个验证间隔内的 critic 对完整采样回报与 GAE 目标的拟合、动作分布 KL、概率比、裁剪比例和二次映射下的动作统计。critic 的 MC 误差使用当前 gamma 计算完整轨迹的折扣回报，不含 critic bootstrap，也不是准确期望价值；GAE 目标来自 critic，不能用拟合 GAE 良好来替代完整回报检验。启用惩罚时，`predicted_start_return_reward_units` 和 `realized_sampled_return_reward_units` 保持奖励单位，不转换为 TWh；`last_episode_penalty` 和 `last_episode_penalty_components` 给出按同一 gamma 加权的总成本及分项。仅关闭惩罚时保留 `predicted_start_return_TWh` 和 `realized_sampled_return_TWh`；它们是按当前 gamma 加权的回报换算值，gamma<1 时也不是实际年发电量。`last_episode_reward` 与兼容字段 `last_discounted_episode_reward` 均为同一折扣求和，`last_episode_energy_TWh` 为实际年发电量。梯度范数仍是裁剪前数值。

DIAG_VALIDATION 默认在第 0、20、100、150、400 轮及本次最后一轮输出。它列出逐年发电量、各库弃水体积、期末库容占可用库容比例、平均动作，以及三次随机策略评估的平均结果；启用惩罚时另报告确定性／随机评估的年度净奖励和逐年成本明细。随机评估使用隔离并恢复的 RNG 状态，不影响后续训练采样；三次评估的波动是动作采样不确定性，不是跨训练种子的标准差。

离散策略下，随机训练的平均映射位置为 Σ p(k)×(k/199)²，确定性评估的映射位置为 (argmax p(k)/199)²，两者可能不同。诊断分别报告这两项统计，本身不更改评估方式。

诊断不保存模型和图片；默认日志仅在内存和控制台，指定 `--episode-log` 才保存逐回合指标。

## Fixed-training-year learning check

Use `--fixed-training-year YEAR` to repeat one year from the existing training split. The initial
check below uses **2013**. Every rollout resets to that year's original initial storage and covers
the complete 365/366-day calendar. Training still samples actions; the five actors, 200 action
bins, physical constraints, reward weights, gamma **0.99**, GAE lambda **0.95**, and optimizer
settings are unchanged. Inflow normalization continues to use the original 50-year training split.

This mode evaluates the current policy **deterministically on the same training year**, before
the first update (episode 0), at every `--eval-interval` HAPPO updates, and after the last update even when it
does not fall on an interval. Evaluation uses a fresh environment, does not insert transitions
into training buffers, and preserves the Torch RNG state. Validation years and typical-year
tests cannot be chosen as the fixed training year. Held-out evaluations and reference simulations
are skipped automatically in this mode; the standard multi-year run is available by omitting
`--fixed-training-year` (the YAML default is `null`).

Run this command from the repository root in the VS Code terminal:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python examples/train_cascade.py \
  --fixed-training-year 2013 --episodes 60 --rollout-episodes 1 --eval-interval 5 --seeds 1 \
  --gamma 0.99 --gae-lambda 0.95 --diagnostics --plot-details \
  --episode-log runs/fixed_2013_seed1_run1.jsonl \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

The primary figure has five **unsmoothed deterministic** curves: actual annual generation,
Power Reward, Guaranteed-output penalty, Ecological-flow penalty, and Total Reward. Dashed lines mark the
untrained checkpoint. The sampled training curves remain separate and are labelled with the
fixed year. `--diagnostics` still reports critic errors against realized discounted returns and
GAE targets, along with policy update statistics; it does not run extra stochastic evaluations
on the held-out years in fixed-year mode.

The JSONL file contains `record_type="training_episode"` for each sampled rollout and
`record_type="fixed_year_evaluation"` for deterministic checkpoints, including episode 0.
In multi-year batch mode it also contains `training_update` summaries and, in
held-out mode, `validation_checkpoint` records with the full per-year results.
Annual episode, HAPPO update, and cumulative training-step counters are separate.
Update-level returns are means of the independently discounted annual returns;
the six years are never discounted as one continuous episode. Diagnostic start
values are likewise averaged over the six true annual starts. Learning curves
use cumulative environment steps; across seeds, mean curves use the mean step
position of matched annual checkpoints, without interpolating the returns.
Checkpoint records contain generation, all reward components, penalty percentages, settings,
and the detailed evaluation trajectory summary. With `--diagnostics`, checkpoints after an
update also retain `last_training_update_diagnostics`, including critic MC/GAE errors from
the latest sampled training rollout (not from the deterministic evaluation). Only training records contribute to training
episode counts and failure-rate denominators. Always choose a new log filename when repeating
the command; existing logs are never overwritten.

Improvement above episode 0 on this fixed year demonstrates in-sample learning. Sustained
improvement and stabilization are useful diagnostics, but do not establish optimality or
generalization to other inflow years. Inspect Total Reward together with generation, penalties
and critic diagnostics. The original information assumptions of the feasibility layer are
retained; this option changes the training-year selection, not the inflow model.

Regression coverage for this mode is in `tests/test_fixed_year_training.py` and
`tests/test_cascade_plot.py`.

## Policy optimization budget diagnostic

The six-year batch baseline still uses five full-batch actor epochs. A larger
batch does not increase the number of optimizer steps. An opt-in comparison
allows more actor epochs with exact categorical KL early stopping:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python examples/train_cascade.py \
  --episodes 300 --rollout-episodes 6 --eval-interval 1 --seeds 1 2 3 \
  --action-bins 50 --ppo-epochs 20 --target-kl 0.01 --diagnostics \
  --diagnostic-episodes 0 24 60 120 180 240 300 --no-plot \
  --episode-log runs/service_batch6_kl001_20epochs_300ep_trial1.jsonl
```

This is an experimental optimization setting, not an established convergence
fix. The source requirements, penalty weights, gamma, GAE, critic epochs,
physical action grid and HAPPO sequential likelihood factors are unchanged.
`--ppo-epochs 5 --no-kl-limit` reproduces the preceding optimization budget.

For each actor, KL(old || new) is measured over all controllable states in
the collected batch, using the complete categorical distributions and original
available-action masks. The old distribution is fixed for that actor's entire
update. If mean KL exceeds the target after an epoch, subsequent actor epochs
are skipped. The last step can exceed the target; this is early stopping,
not a strict projection or a guarantee of monotonic return improvement.
The critic completes its own configured updates. The approach of stopping
further PPO optimization based on KL is described in
[OpenAI's PPO documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

Startup metadata records the requested budget and KL target. Update JSONL
records include each actor's actual optimizer steps, completed epochs, final
KL and early-stop indicator. Diagnostics also show maximum action probability,
the gap between the two largest probabilities, normalized entropy, and the
fraction of old-rollout states whose preferred action changed. Stochastic
validation at the requested diagnostic milestones remains isolated from the
training RNG; compare it with deterministic validation and service compliance.

`examples/compare_cascade_policy_budget.py` provides a paired training-year
check with a common seed, initial policy, annual episode budget and sampled
evaluation seeds. It compares five unguarded epochs with twenty guarded
epochs without tuning on test years. For a bounded smoke check:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python examples/compare_cascade_policy_budget.py \
  --year 2013 --updates 1 --rollout-episodes 1 --seeds 1 \
  --evaluation-repeats 1 --output runs/policy_budget_smoke_trial1.jsonl
```

The default comparison uses six updates of six annual trajectories and three
sampled evaluation repeats per arm; it is substantially more expensive. A
single-year check establishes neither held-out improvement nor convergence.
Full runs must assess several seeds, sampled and deterministic held-out
returns, and physical shortfalls. Keep the test years reserved for final
evaluation, and preserve the preceding run as the baseline.

## Reward scaling and episode logging

Every sampled training episode prints an `EPISODE_METRICS` JSON record, independently of `--diagnostics` and `--plot-details`. Two new plot windows open automatically unless `--no-plot` is set:

- **Reward components:** Power Reward, Guaranteed-output penalty, Ecological-flow penalty, and Total Reward. Every component is weighted by the same configured `gamma**t`, with `t=0` on the first executed day of each episode. Penalties are positive applied costs: `Total = Power - Guaranteed output - Ecological flow`. Power Reward is a dimensionless generation return, not MW or actual annual energy. With `gamma=1`, all these sums become undiscounted. The plots can still read separate historical three-penalty logs, but reject mixed old/new component schemas.
- **Penalty percentages:** each cost and their sum divided by Power Reward, multiplied by 100. Thin raw curves use each episode's ratio. The 20-episode curves use `100 * sum(window cost) / sum(window Power Reward)` for each seed, followed by the seed mean and sample standard deviation. They do not average episode percentages. A zero denominator is undefined (`null` in logs, a gap in plots).

Episode records include:

- Actual generated MWh/TWh and generation by reservoir, with sampled year and simulated/calendar days. `annual_generation_TWh` is present only for completed episodes; failed records retain the actually executed prefix and set annual generation to `null`.
- Discounted reward components and their exact percentages of discounted Power Reward, plus gamma, GAE lambda, reward weights, the normalization constant, `reward_aggregation`, and `reward_definition= generation_minus_mean_service_shortfalls_v1`. Thresholds and paper provenance are included in `service_requirements`. Physical service counts and deficits remain undiscounted.
- `hard_constraint_failed` and `hard_constraint_failure_rate` (a per-attempt indicator, 0 or 1), and `hard_constraint_failure_rate_so_far` (failed training attempts divided by attempted training episodes, separately for each seed). A physical infeasibility/assertion during a transition is logged and then re-raised: no failure penalty, relaxed constraint, or automatic continuation is introduced. A zero failure rate describes the environment-protected executed trajectories; it does not establish that the raw policies independently enforce the constraints. The current future-inflow feasibility layer is explicitly identified in the record.
- `service_shortfalls` for `guaranteed_output` and `ecological_flow`: shortfall reservoir-day count/rate, compliance rate, days with any shortfall, mean/maximum relative shortfall, maximum physical shortfall (MW or m³/s), and per-reservoir arrays in `reservoir_ids` order. Integrated deficits are `shortfall_energy_MWh` and `deficit_volume_m3`, using the time step in hours and seconds respectively. Rates use active reservoir-days, including leap days and excluding rejected transitions; inactive requirements have null compliance rates. Cascade ecological deficit volume is a sum over outlet requirements, not unique missing basin water. Deficits are still measured with penalties disabled when the table is loaded, and do not depend on the seasonal-constraint switch.

To retain logs, specify a **new** JSONL path; existing files are never overwritten. Parent directories are created, each episode is flushed immediately, and the log remains available if a later episode fails. This adds accounting only and does not change reward weights, action selection, inflow information, or HAPPO updates.

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python examples/train_cascade.py \
  --episodes 100 --eval-interval 10 --seeds 1 \
  --plot-details --skip-final-test \
  --episode-log runs/reward_scaling_seed1.jsonl \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

Focused accounting checks: `PYTHONPATH=tests /opt/anaconda3/envs/harl/bin/python -B -m unittest test_reward_scale_metrics test_cascade_reward_penalties`. Plot checks use the Matplotlib interpreter listed above and `test_cascade_plot`.

## 折扣与 GAE 回报对照

`--gamma 0.99` 或 `--gamma 1` 可以覆盖 `harl/configs/algos_cfgs/happo_cascade.yaml` 的 `algo.gamma`；未指定时读取配置，当前配置为 0.99。支持有限数值 0≤gamma≤1，非法值在训练前报错。启动输出显示实际 gamma 和 gae_lambda。训练 GAE、采样回合日志、固定年评估、五年验证、最终测试、参考控制器、随机策略诊断和奖励图统一使用同一 gamma；每个年度回合的第一天权重为 1，随后为 gamma、gamma² 等。选模始终使用平均 `episode_reward = sum(gamma**t * r_t)`，关闭软约束惩罚后也遵循此公式。年度实际 MWh/TWh、弃水量和约束超出次数仍是物理量，不能与折扣奖励混用；gamma<1 时不能用实际年发电量反算 Power Reward。gamma=1 对应全流程未折扣年度目标。

Daily rewards are not pre-discounted before insertion into the critic buffer: GAE applies gamma once. Reward accounting independently applies the same gamma to generation reward and every penalty, so decomposition and penalty ratios use one convention. Policy/reference evaluation requires an explicit `gamma` argument. New JSONL records include `reward_aggregation`; older logs used undiscounted reward sums even when their gamma field was below one, so their reward values must not be compared directly with new runs. Actual energy totals remain comparable under identical physical assumptions.

The HAPPO actor now also weights each calendar row by `gamma**t`, where `t=0` at the start of that episode. This is the episode-start state-visitation weight, distinct from GAE's discount of future rewards relative to the current decision. The buffer carries it through minibatch shuffling separately from the sequential HAPPO likelihood factor. Both the clipped policy surrogate and the entropy bonus use these weights; advantage normalization happens before weighting.

Losses average over all sampled rows, with forced-action contributions set to zero, rather than dividing by the minibatch's active count or discount mass. This preserves time weighting even with one sample per minibatch. With `gamma=1`, every day has unit time weight; when some actions are forced, the full-row average scales both policy and entropy losses by the active fraction compared with the previous active-only average. All-forced or zero-discount minibatches skip Adam entirely. The optimizer's `actorN_dist_entropy` is now a discounted full-row mean; `actors[].entropy` in the distribution diagnostics remains an unweighted entropy on active states.

GAE, critic targets, daily rewards, physical constraints, reward coefficients, and evaluation scores are unchanged by this actor correction. PPO clipping, approximate advantages, and entropy regularization still make this a surrogate update, not a guarantee that each update improves the unregularized evaluation return. Start a new training run to assess the correction; an already-running process keeps its loaded implementation.

`--gae-lambda 1` 是可选对照设置。完整年度终止时，λ=1 使每步价值目标等于从该步到年末、按所选 gamma 计算的完整实际回报；gamma=1 时即为未折扣累计回报。该设置消除未来价值 bootstrap 的依赖，但采样方差可能增大。未指定时继续使用原配置 0.95。

GAE λ 无论来自 YAML、命令行还是直接构造 runner，均须为 [0, 1] 内的有限数值。种子须为 [0, 4294967295] 内的整数，布尔值和浮点种子会被拒绝；种子列表仍须非空且不重复。命令行先覆盖 YAML，再校验最终生效值；非法参数在读取数据、可行性检查和参考计算前报错。
