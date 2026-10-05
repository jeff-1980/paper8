# BM3 CWRU 窗口泄漏修复 — STEP 1 预注册

写入时间早于本卡任何新 run（目前尚无任何新 run 落盘）。

## P1（无条件替换）
修复后结果无论方向如何，全量替换受影响的表与图。不与旧结果混用、不做挑选。

## P2（主张随数据走）
若修复后某些主张不再成立，照实修改稿件。BM3 尚未投出，此时发现问题是运气好，不是损失。

## P3（禁止调参）
超参与现行配置逐字段一致（`d_model`/`d_state`/`n_layers`/`lr`/`weight_decay`/
`epochs`/`batch_size`/`lambda_kin`/`kin_variant`/`seeds`/`conv_stride`/
noise 协议等，参见各 `config*.yaml`）。修的是数据划分，不是模型。

## P4（划分公开）
划分索引（每个 split 的 file_id + 段边界 + guard gap 长度）落盘并随论文公开，
路径形如 `results/<exp>_leakfree_<timestamp>/split_index.json`。

## P5（不得回退）
不得因修复后指标下降而恢复旧划分，或在两套结果间挑选。

## P6（p=0.031 诚实条款，2026-09-12 补）
`exp_b2_dual_*`（DE+FE 双传感器 SNR 曲线）在旧管线下的 CWRU B2 −4dB 是
"Sufficient-data regime" 这条论点在正文里被明确标注为**统计主证据**的唯一
显著性来源（p=0.031），XJTU 结果仅为"机制层分证据"。**若 leak-free 重跑后
该 p 值升到 0.05 以上**：

- Sufficient-data regime 这条论点必须照实修改或撤下；
- **不得**改用 XJTU 的分证据顶替、补位或"仍可支持该论点"式改写；
- **不得**调整统计检验口径（换检验方法、换显著性水平、改用单侧检验等）；
- **不得**挑选 SNR 档位（例如换成另一个恰好显著的 SNR 点重新定义"代表性档"）。

本条在重新出结果之前写入，就是为了让"诚实"不依赖于看到结果之后的自觉。

## P7（文档同步条款，2026-09-12 补）
**背景**：本文件下方原"STEP 3 门槛触发声明"写着"人工对范围做出决定前不得
继续"，但范围决定实际已在 2026-09-11 做出（分档C：exp01-05 主表 + B2，见
`gen_configs.py` 开头声明"per user decision 2026-09-11"），本文件当时未
回写更新，导致后续核实时一度被误读为"决策前流程被绕过"。

**规则**：此后任何一次范围/口径/协议决定做出后，**当次即回写本 prereg 的
状态描述**（不得留待下一次任务卡或下一个 [redacted] 去重新解读文件间接证据）。
这是记录在案的一次真实教训，不是预防性条款。

---

## 范围声明（依据 `bm3_exposure_inventory.md`，2026-09-11 定稿）

- CWRU：LEAKY，受影响实验族列于敞口盘点 §5，真实 run 数下界 551，
  实测 GPU 时长下界 44.76 GPU-h。
- PU / XJTU（窗口级）：已判定 CLEAN，不在本次重跑范围内。
- XJTU 逐 epoch 选点问题：已知、范围外，由 `exp_e3_lobo_leakfree` 单独处理。
- **范围决定：分档C**（exp01/02/03/04/05 主表 + `exp_b2_dual_*` C3 统计主证据），
  2026-09-11 由用户拍板，`run_manifest.json` 共 42 条，`gen_configs.py`/
  `run_all.py` 已按此范围实现。

## STEP 3 门槛触发声明（历史记录，见下方 2026-09-12 更新）

`bm3_rerun_cost.md` 中枚举的成本已确认超过任务卡规定的 8 GPU 小时硬停止阈值
（实测下界 ≈44.76h，约为阈值的 5.6 倍）。这一超阈值判定本身不受下面的更新
影响——分档C的 42 条已经是在此判定基础上、经用户在分档 A/B/C/D 中选定的
子集，不是绕开阈值。

## 2026-09-12 更新：因缺逐 epoch checkpoint 中止已跑的 23/42 条，补三件套后重启

`run_all.py` 于 2026-09-12 10:22 启动分档C的满量执行，跑到第 23/42 条
（`exp04_mamba2_snr-6`）时，核实发现 `train_leakfree.py` 全程零次
`torch.save`——不仅没有逐 epoch 权重，连 best checkpoint 都未落盘，仅在内存
中 `copy.deepcopy` 后随进程退出丢弃。这与 KG-cSSM 只存 `best.pt` 导致后续
无法换选点准则重评是同一类问题，且更严重。

**处理**：
1. 待当前正在跑的 `exp04_mamba2_snr-6` 自然完成（避免中断产出半截 run），
   随后停止 `run_all.py`（tmux 会话 `bm3_leakfree_c`），不 kill 当前子进程。
2. 已完成的 23 条（含 `exp01_cwru_baseline`/`exp01_cwru_kin` 两个 2026-09-11
   夜间开发期产物）整体移入 `results/_superseded_no_checkpoints/`，保留不删。
3. `train_leakfree.py` 打补丁：新增逐 epoch checkpoint 落盘（`model_state`+
   `optimizer_state`+`epoch`+`val_acc`+loss，估算 42×5×50=10,500 个文件、
   实测约 1.14MB/个、共约 11.7GB，disk 762GB 可用，**全存不做准则子集裁剪**）
   与源码快照（`train_leakfree.py`/`data_cwru_leakfree.py`/`data_cwru.py`/
   实际用的 config，四者 md5 一起落盘到每条 run 的 `results_dir`）。
4. 划分零交集断言（`disjointness_check.json`）本已存在且逐 run 执行，本次
   补丁同时把其 pass/fail 状态与源码快照路径写入 `summary.json`，便于事后
   核对而不必逐个打开 `disjointness_check.json`。
5. 补丁就位后，先跑 1 条×2 epoch 烟雾测试确认权重、快照、断言三者均正确
   落盘，**用户已明确要求：补丁完成后先暂停，不自动进入烟雾测试或满量重跑，
   等待用户下一步指令**。
6. 42 条全部按新版 `train_leakfree.py` 重新执行（不是断点续跑那 19 条——
   已完成的 23 条同样缺三件套，一并作废重跑）。

## 2026-09-15 更新：42/42 全部完成，P6 判定结果

烟雾测试通过（checkpoint/source_snapshot/disjointness 三件套核验正常）后，
42 条已于 2026-09-15 19:55 全部跑完，27.52 GPU-h，0 报错。期间因另一 [redacted]
会话（`~/论文4/github_repo/macp_rul`）竞用同一 GPU，中止过一次（未触碰对方任务），
2026-09-14 10:18 GPU 空闲后干净重启，同配置前后实测吞吐持平（9.74s/epoch vs 原
10.22s/epoch），排除电源模式导致降频的假设。

**P6 判定（最高优先级）**：C3 主统计证据（CWRU B2 -4dB nokin，Dual vs Single）
复核结果 **p=0.0312**（与原始 0.031 几乎一致，未升过 0.05）。**P6 触发条件未满足，
Sufficient-data regime 论点无需修改或撤下**——按 P6 本条规则，此判定不依赖看到
结果后的自由裁量，检验方法（one-sided greater，与 `analyze_b2.py` 逐字段一致）
在跑之前已由预注册锁定。

完整证据见 `EVIDENCE_REPORT_bm3_fix.md`（STEP 6 逐格对照、RUBRIC 自查）与
`comparison_bm3_old_vs_leakfree.md`（39 项数字对照表）。

## P8（B2 旗舰点 n=8 leak-free 口径，2026-09-15 补，跑之前写）

B2 旗舰三点（−4/−6/−8dB）的 leak-free n=8 重跑（BATCH 1，seed 5–7 新增）结果
**无论 p 值落在哪个区间**，论文口径统一维持：

> "方向一致的描述性趋势（dual > single，随 SNR 降低单调增大），不主张统计显著。"

**−4dB 不再作为"唯一显著点"引用**——这与 sensors 版 2026-07-20 text-surgery 已经确立
的口径一致（`section4_experiments_sensors.tex:551-553`："the originally reported
'unique statistically significant point' was a false positive of small-$n$ exact
Wilcoxon testing, not a real effect"）。

本条在跑 BATCH 1 之前写入，目的与 P6 相同：不让"诚实"依赖于看到结果之后的自觉。
即使 leak-free n=8 意外地把 −4dB（或任一旗舰点）的 p 值重新压到 <0.05，也**不得**
据此恢复"唯一显著点"式表述或升级为主统计证据——leak-free 划分改变的是数据完整性，
不改变 P8 锁定的叙事框架决定（该决定基于统计功效分析 + sensors 版已有共识，不是
基于单次检验结果）。

## P9（BATCH2 逐族口径锁定，2026-09-16 补，跑之前写）

覆盖 BATCH2 的 9 项（`BATCH2_cost_table.md` 里的 8 个源实验族 + single_kin 补跑；
原表"源实验族数：8"一行系笔误漏计 exp_e5_dual_pink，实际为 9 族，此处一并更正）。
每项列出 sensors 版依赖的具体主张（原文+行号）、检验方法（与原脚本/原表一致，
不更换）、以及无论 leak-free 结果如何都要遵守的口径规则：

| 族 | Sensors 版依赖的具体主张（原文+行号） | 检验方法（与原脚本同） | 口径规则 |
|---|---|---|---|
| exp07_cnn1d | `section4_experiments_sensors.tex:161-166`（`tab:cwru_baselines`）："1D-CNN 100.00/99.97/95.96%"；line 178"1D-CNN (BN) reaches 95.96% at −8dB, outperforming BM3+L_kin (85.76%, Δ=+10.2pp)" | 无显著性检验，纯描述性 mean±std 对照（n=5 seeds） | 无条件全量替换三点数字；若相对排名因 leak-free 改变（如 CNN 不再是最强基线），照实改写"1D-CNN reaches 95.96%..."这类具体数字引用，不得为维持"BN是CNN低SNR优势主因"的既定结论而选择性只替换部分方法 |
| exp07_transformer1d | `tab:cwru_baselines` 同上表 Transformer-1D 行（100.00/95.85/82.41%） | 同上，无显著性检验 | 同上，无条件替换 |
| exp07_sksvm | `tab:cwru_baselines` 同上表 SK-SVM 行（99.64/84.21/65.86%）+ line148-149"SK-SVM relies on a manually designed frequency feature set and does not generalise..." | 无显著性检验；另需注意 `train_sksvm.py` 本身不经过 `train_leakfree.py` 训练循环，是独立特征提取+SVM管线，需要专用脚本（见下方任务2） | 无条件替换三点数字；"does not generalise to variable shaft speed"这句定性论点与leak-free划分无关，不受本次影响，不用改 |
| exp_mext_e21_bm3bn_cwru (nokin) | `section4_experiments_sensors.tex:1161-1176`（`tab:bm3bn_cwru`）："BM3+BN (ablation): 100.00/95.79/82.46%"，"Δ(BN added)=0.00/−1.58/−2.99"，"Adding BN hurts BM3...confirms BM3 cannot benefit from the statistical regularisation" | 无显著性检验，纯描述性 Δ 值（mean±std，n=5 seeds） | 无条件替换 Δ 值；"adding BN hurts BM3"这一方向性论点只有在方向真的反转（BN不再让BM3变差）时才能撤下或改写，不得只改数字保留结论 |
| exp_mext_e22_bm3bn_kin_cwru (kin) | 同上 `tab:bm3bn_cwru`，BM3+L_kin 对应行（表中隐含，nokin行为BM3 no-BN原始对照，kin行需查完整表） | 同上 | 同上 |
| exp_mext_e21b_1dcnn_nobn_cwru | `section4_experiments_sensors.tex:1105-1108`+`1139-1151`（`tab:bn_ablation` 表注）："Removing BatchNorm reduces 1D-CNN accuracy by 12.69pp at −8dB (two-sided Wilcoxon: 5/5 seeds concordant, p=0.063)...Without BatchNorm, 1D-CNN (83.26%) is no longer significantly different from BM3 (85.45%, Wilcoxon p=0.125)" | **两个两侧 Wilcoxon**（与原文一致）：① CNN-BN vs CNN-noBN 配对（n=5, two-sided）② CNN-noBN vs BM3-CE-only 配对（n=5, two-sided）——注意 B2 用单侧、这里用**两侧**，检验方法不能混用 | 这是 D33"Win2"叙事的关键论据（BN是CNN低SNR优势主因）。按 P6 同款规则：无论两个新 p 值落在哪个区间，"BatchNorm是主要机制"这一定性结论只有在方向真的反转时才能改写，不得因 p 值数值变化就重新措辞成"更强"或"更弱"的显著性声明；`% ⚠️ 论文禁用：写"BM3 significantly outperforms CNN-noBN" / 写"BN is the sole cause"`（line 1184-1185注释）这条既有禁令continue有效 |
| e2_cnn_ln (cwru_cnn1d_ln_*) | `section4_experiments_sensors.tex:1195-1230`（`tab:cnn_ln_within` + 预注册 recovery 公式）："recovery = (CNN-LN_OOD − CNN-BN_OOD)/(BM3_OOD − CNN-BN_OOD})...recovery≥0.5 would demote the paradigm claim...recovery<0.5 would reinforce" | 无显著性检验；核心是预注册的 recovery 公式阈值判定（≥0.5 vs <0.5），公式依赖 OOD(XJTU)数据，本次只重跑 within-condition(CWRU)部分 | 预注册阈值判定必须原样执行，不得因 leak-free 后 within-condition 数字变化就重新选阈值或加例外条款；由于 recovery 公式还需要 OOD 部分（不在本次范围），本次只能标注"within-condition部分已更新为leak-free，recovery最终判定待OOD leak-free后确认"，不得单凭 within-condition 数字改写最终 recovery 结论 |
| exp_e5_single_pink | `section4_experiments_sensors.tex:640-671`（`tab:e5_pink` + 正文）："the dual−single gain is negative at every SNR and grows more negative as SNR decreases (−0.20→−0.32→−1.16pp), the mirror image of the AWGN trend" | 无显著性检验；核心主张是 **Δ 符号在 pink noise 下相对 AWGN 反转**（3个SNR点全部为负） | 无论 leak-free 后各点具体数值如何，如实报告符号方向；不得为维持"变负"的叙事而挑选SNR点或颠倒符号描述——若 leak-free 后某点符号变回正，必须如实指出"符号反转在leak-free下不再全SNR点成立" |
| exp_e5_dual_pink | 同上 | 同上 | 同上 |
| single_kin 补跑（补全B2 kin的n=8配对） | `section4_experiments_sensors.tex:592-616`（`tab:b2` kin列 + 表注）——与 P8 同源 | one-sided Wilcoxon(dual, single, alternative="greater")，与 `analyze_b2_leakfree.py`/`analyze_b2.py:132` 一致 | 补全后若 kin 在 n=8 下也呈现"−8dB最强、−4dB不显著"的模式（与 BATCH1 nokin 结果一致），按 **P8** 同款原则处理（不据此升级为新的显著点宣称）；若 kin 模式与 nokin 不同（例如 kin 在 −4dB 反而显著），**照实报告这个不对称**，不得强行统一叙事 |

写完本条才开始执行 BATCH2 的任务2（新脚本）和后续训练队列。

## 2026-09-17 更新：BATCH2 全部完成

GPU队列45 config/219 run（18.35 GPU-h，0报错，中途因用户临时任务暂停一次，
幂等续跑未丢进度）+ SK-SVM CPU队列30 run（0报错）全部完成，三件套逐config核验通过
（45/45 GPU config PASS）。9族"旧vs leak-free"对照 + 按P9逐条判定见
`BATCH2_RESULTS.md`。

关键发现（均按P9承诺如实报告，未做选择性宣称）：
- 8/9族数字变化在1.5pp以内、无方向反转
- **E5 −8dB符号反转**：原始"dual−single gain negative at every SNR"在leak-free下
  不再成立（−8dB从−1.16pp变为+0.40pp），需要改稿时重新措辞
- **B2 kin/nokin不对称**：single_kin补跑后完成n=8完整配对检验，kin条件统计证据
  显著弱于nokin（−4dB几乎无效应p=0.523，−8dB虽显著但p=0.043弱于nokin的p=0.0039）
- 1D-CNN −8dB是本轮最大单点变化（−1.32pp，95.96%→94.64%），方向未反转

## 2026-09-17 更新：BATCH3（coherence用n=8 leak-free增益重算）完成

不涉及新训练（coherence是原始信号谱特性，与窗口划分无关）。CWRU 5个gain点替换为
leak-free值（0dB/−2dB用n=5主campaign值，−4/−6/−8dB用BATCH1 n=8值），XJTU 2点不变
（已由XJTU自己的leak-free工作产出，不在本次CWRU任务范围）。

关键发现：CWRU-only 5点Spearman从sensors版r=−0.900,p=0.037变为**r=−1.0000，精确
p=0.0167**（原因：leak-free数据恰好消除了sensors版−2dB那个非单调点）；⚠️
**scipy渐近公式对此会给出p≈1.4e-24这种无意义极端值，已用n=5精确置换检验
（120种排列，2种给出|r|=1）验证正确值为1/60≈0.0167**。Pooled 7点结论方向不变
（r=−0.25,p=0.589 vs 原−0.214,p=0.645）。按既定纪律（呼应P6/P8/P9），不得把
r=−1.0据此包装成更强的机制证据——sensors版原文"mechanism demonstration，非独立
统计发现"的限定性表述对leak-free版本同样、甚至更需要成立。详见
`BATCH3_RESULTS.md`（含需要改稿的具体位置清单）。

## P10（XJTU Cross/PU "测试集选点"问题的降级路径，2026-09-17补，BATCH5跑之前写）

**背景**：`BATCH5_cost_table.md` 确认 XJTU Cross-condition（含 E1b 三backbone的
"+38.4pp" OOD头条数字所在数据）和 PU 全部实验，都存在"测试域/测试折每epoch直接用于
选点"的问题（`experiments/exp_xjtu/train.py::train_one_run()` 逐epoch在test_ds上
取max；`experiments/exp06_pu/train.py::build_loaders()` 的 cross-condition
"val_ds"就是COND_TEST）——这与本任务已修复的CWRU窗口重叠泄漏是**不同类别**的问题，
但同属"报告数字来自被选点污染的测试集"。XJTU LOBO 已有局部修复
（`train_lobo_leakfree.py`），Cross-condition 和 PU **从未修复**。

**判定方法（跑之前锁定，不得事后更换）**：用"训练域内部划分独立val、目标域(测试折/
跨工况条件)只在训练结束后评估一次"的协议重跑，与现有（选点污染）结果做
Wilcoxon配对检验（沿用本任务一贯的one-sided/two-sided惯例，按各自原表已用的检验
方式，不新引入检验方法）。

**"+38.4pp" OOD头条数字的降级路径（预先定义，按干净选点重跑后的实际结果对号入座，
不得事后选择性适用）**：

| 干净选点重判后的实际情形 | 判定阈值（预先定义） | 必须采取的行动 |
|---|---|---|
| 基本不变 | 仍 ≥20pp 且方向/显著性不变 | 维持现有"+38.4pp"框架式表述，仅替换具体数字 |
| 明显缩水但仍成立 | 10–20pp，方向不变，仍显著或至少5/5(8/8)方向一致 | 弱化措辞（删除"dramatic"/"substantial margin"类强调词），数字如实替换，不撤主张 |
| 大幅缩水或显著性消失 | <10pp，或配对检验不再显著 | **必须降级**：不得继续作为paper的\*primary\*OOD证据（当前 `subsubsec:paradigm`/
D31"Win1"定位）；比照本任务对C2"方差稳定"主张的处理方式（`EDITS_C2_withdraw.md`），
改写为"conditional/exploratory finding"或诚实负结果，并重新评估整个paradigm章节
（Win1若倒、Win2已经在D33降级，两个OOD支柱都不在，需要用户参与决定paradigm section
是否整体重写或降级为次要发现） |
| **方向反转**（BM3不再优于1D-CNN OOD） | Δ<0 或方向不一致 | **停止自动处理，立即升报用户**——不得由[redacted]单方面决定如何改写这个级别的
方向性反转，比照本任务GPU资源冲突/更重要任务打断时"先问后动"的处理惯例 |

**PU的对应主张**（`tab:c1_pu`"BM3 accuracy non-inferior to BM2"、abstract finding
(1)）同样需要按干净选点重判；若"non-inferior"论断在干净选点下不再成立（BM2显著
优于BM3），比照上表第3/4档处理，不单独另定阈值。

**本条锁定的原则**：与P6/P8/P9一致——阈值和行动路径必须在看到干净选点重跑结果**之前**
锁定，跑完后按对号入座执行，不得因为结果好看/难看而重新协商阈值本身。

## P10 补充三条硬规则（2026-09-17，BATCH5跑之前写）

**(a) 选点协议——二选一，写死为"固定最终epoch"**：Cross-condition 和 LOBO 统一采用
**固定最终epoch**（训练满50个epoch，取最后一个epoch的checkpoint，不做任何逐epoch
选点），**不采用"源域val选点"**。理由：
  - LOBO 已有验证过的先例（`exp_e3_lobo_leakfree/train_lobo_leakfree.py`的
    `selection_mode: fixed_epoch`，已产出4臂×4折×5seed=80个run，无需按新规则重跑）；
    延续同一规则，LOBO现有leak-free结果保持有效，不需要因为本条新规则而作废重来。
  - 固定最终epoch是"零选点"，天然满足"目标域数据从训练到选点全程不可见"——不存在
    任何选点过程去污染，比设计一套新的源域val切分方案风险更低、更快能验证正确性。
  - Cross-condition 此前从未有任何修复，直接采用与LOBO相同的规则，代码改动量最小
    （去掉`run_cross()`里逐epoch用`test_ds`算`best_macro_f1`的`max(...)`累积，
    改成训练满50epoch后只评估一次）。

**(b) 同一选点规则对 1D-CNN / BM2 / BM3 一视同仁**：三个backbone在Cross-condition
和LOBO上都套用(a)的固定最终epoch规则，不因backbone不同而有选点方式的差异
（当前1D-CNN/BM2走的是与BM3相同的`train_one_run()`共用函数，天然满足这条）。

**(c) 若"零选点"结果不稳定，如实报告，禁止回退到test选点**：固定最终epoch本身没有
"源域val选点不稳定"的问题（因为压根没有选点），但**训练本身**可能不稳定（例如最后
一个epoch恰好赶上一次loss尖峰，导致该seed的数字明显偏离其余seeds）。若观察到这类
不稳定（如某seed显著偏离，或std异常大），**照实报告该不稳定性**（不隐藏、不剔除
异常种子），**绝对不得**以"选一个更稳定的epoch"为由重新引入逐epoch用测试域/测试折
选点的旧模式——固定最终epoch的"确定性代价"（可能不是训练过程中最好的checkpoint）
是这条规则本身接受的权衡，不能事后用测试集选点来"优化"掉这个代价。

**队列顺序**：Cross-condition 全部族 → LOBO n=8 三组（1D-CNN/BM2/BM3 backbone-
agnostic表所需的seed 5-7扩展）→ PU。Cross-condition 完成后立即通知，不等LOBO/PU
一起汇报。

## 2026-09-17 更新：四项任务（EDITS_C2_withdraw补算 / BATCH5成本表+P10 / BATCH3再扩展 / BATCH4脚本改造）全部完成

1. `EVIDENCE_REPORT_bm3_fix_addendum2.md`：`[p_kin_min]=0.0625`（−8dB）、
   `[p_10class]=0.6875`、`[p_bm2_-4/-6/-8]=0.1875/0.0625/0.0625`，均支持
   `EDITS_C2_withdraw.md`的撤回方向。
2. `BATCH5_cost_table.md`：XJTU LOBO/Cross+PU全部族盘点，合计≈28-39 GPU-h；
   确认XJTU Cross-condition和PU从未修复过"测试域选点"问题（比LOBO更彻底未处理）；
   P10写入，"+38.4pp"降级路径已预注册（4档阈值+方向反转时的"停止自动处理"条款）。
3. `BATCH3_RESULTS.md`§6：纳入kin增益，发现kin相关性不显著（r=−0.667,p=0.233，
   对比nokin的完美r=−1.0），XJTU LOBO行标记`[BATCH5]`占位，同时主动标注Cross-
   condition行同样未修复（用户未要求但如实披露）。
4. `BATCH4_summary.md` + `batch4_diffs/*.diff`：6个出图脚本全部完成代码改造
   （路径/字段/删硬编码），**未运行**。顺手修复了`step5_b2_snr_curve_n8.py`里
   与`snr_curve`同类的双负号格式bug。t-SNE两个脚本因为自带训练逻辑，做了更大的
   管线重写（CWRUDataset→CWRULeakfreeSplitBuilder，特征提取从val改为test）。

## 2026-09-17 更新：Cross-condition完成 + P11（LOBO n=8扩展，跑前写）

Cross-condition 全部16族完成（含smoke污染事故自查自修，详见BATCH5_RESULTS.md），
原"+38.4pp"OOD claim（BM3+kin single vs 1D-CNN single）在leak-free下缩水到+21.7pp
（p=0.0625不变，5/5一致；nokin n=8口径+22.9pp,p=0.0078,8/8一致）——按P10四档表
判定为"基本不变"档（Δ≥20pp且方向/显著性不变），维持框架仅换数字。

**LOBO n=8 范围决策（用户2026-09-17拍板）**：不新增backbone，仅现有
`exp_e3_lobo_leakfree`4臂（single/dual×nokin/kin）加seed 5-7扩展，协议/选点规则/
脚本与现有leak-free LOBO完全一致（`train_lobo_leakfree.py`，`selection_mode:
fixed_epoch`，沿用P10(a)(b)(c)）。

**P11（LOBO n=8扩展，跑前写）**：

- **范围**：`exp_e3_lobo_leakfree`现有4臂（single/dual×nokin/kin）加seed 5-7，
  不新增backbone。协议、选点规则、脚本与现有leak-free LOBO完全一致。
- **主张不变**：负迁移方向（dual<single，即B4-5发现的LOBO双通道OR类退化）是既定
  观察，本次仅提升统计精度（n=5→n=8），不重新开放这一结论是否成立的讨论。
- **结果处理（四种情形，跑前预写，跑后机械套用）**：
  a) 8/8同向 → p=0.0078，"do not fuse under per-bearing scarcity"升级为有统计
     支撑的建议；
  b) 7/8或6/8同向 → 如实报p，建议保留但措辞维持descriptive（不升级为"显著支撑"）；
  c) 方向不一致（≤5/8）→ 负迁移主张降级为"在本设置下未观察到融合收益"，C1的
     regime-dependent表述相应弱化，**不得**改用n=5结果替代（即不能挑对自己有利
     的样本量口径回避新证据）；
  d) dual臂std仍在10pp量级导致检验无意义 → 照实报std，以描述性呈现，并在表注
     说明结构性方差来源（呼应[internal project notes, not released]已记录的"每折仅1个OR训练轴承"归因）。
- **无论哪一档，n=5结果不再单独引用，表内统一改标n=8。**

**队列进度**：Cross-condition ✅完成 → **LOBO n=8（进行中）** → PU（未开始）。
LOBO n=8完成后按P11机械判定并通知，不等PU一起汇报。

## 2026-09-18 更新：LOBO n=8完成 + P12（PU leak-free改造，跑前写）

LOBO n=8扩展4/4 configs全部完成，provenance核验通过（12/12 fold-json+12/12
checkpoint+seeds匹配，BAD: NONE）。P11判定：nokin/kin两个变体均8/8同向为负
（dual<single），Wilcoxon p=0.0078（n=8理论下限）→ 落入情形(a)，"do not fuse
under per-bearing scarcity"升级为有统计支撑的结论。详见BATCH5_RESULTS.md。

**P12（PU leak-free改造，跑前写，用户2026-09-18原文）**：

- **现状**：`exp06_pu`的`COND_TEST`同时充当选点依据与最终测试条件，属P10禁止的
  test选点（`experiments/exp06_pu/train.py:56-62`构建`val_ds=PUDataset(conditions=
  COND_TEST,...)`；训练循环内`val_acc=val_epoch(model,val_loader,...)`逐epoch算，
  `best_val=max(best_val,val_acc)`——`best_val_acc`即为对COND_TEST逐epoch选出的最大值）。
- **改造规则**：
  (a) 从训练域内部切出独立val，测试条件仅在训练结束后评估一次；
  (b) 训练域val的切分粒度必须是文件级或轴承级，禁止窗口级随机切分——PU为离散
      两转速（900/1500rpm），窗口级切分会引入与CWRU同类的窗级泄漏；切分方案、
      每个split的文件/轴承清单、窗口数写入split记录并落盘；
  (c) 选点规则对BM3 nokin/kin与BM2一视同仁；
  (d) 若训练域val因样本量小导致选点不稳定，如实报告，不得回退test选点。
- **主张处理**：PU现有结论为负结果（L_kin在离散转速下无效，均值与方差均无改善）。
  若leak-free后该负结论不变→直接换数字；若转为正向效应→停止并升报，不得自行
  改写L_kin定位。
- 7个族（nokin/kin × clean/snr0/snrm4 + bm2_nokin）全部适用。

**执行流程**：①先出改造方案（改哪些行/新脚本命名/split构建方式）连同本条P12
交用户确认，②方案确认后smoke（独立临时路径）→三件套核验→全量，③写入
BATCH5_RESULTS.md，附每族"旧vs leak-free"对照与P12判定。

**方案已批准（用户2026-09-18）**：val_frac=0.2、文件级切分（COND_TRAIN内每个
(轴承,工况)组20文件留4个val）、split-seed与训练seed解耦（同一config的5个训练
seed共用同一份split）、不加guard gap（stride=win_len=4096无重叠，PUDataset对每
文件独立切窗从不跨文件取窗，文件级split天然窗口安全）。新文件
`data_pu_leakfree.py`（PUFileSplitBuilder+PUDatasetFromFiles，只读import
`bearmamba3/data_pu.py`的_parse_fname等，不改原文件）+
`train_pu_leakfree.py`（源域val逐epoch选点，test仅评估一次）。

**P12补充条款(e)(f)(g)（用户2026-09-18，跑前写）**：

(e) **分层**：每个(轴承,工况)组抽val时保证类别覆盖；若组内标签单一（PU每个轴承
    ID对应固定损伤类型，天然组内单一标签——K001/K002=Normal, KA04/KA15=Outer,
    KI01/KI03/KI05=Inner），则在14组汇总层面确认val集覆盖全部类别。
    `split_record.json`增列每个split的类别分布（类别→文件数、窗口数）。
(f) **选点稳定性判据（跑前定义，事后机械套用）**：每个run记录best epoch的位置、
    best epoch源域val_acc与最后5个epoch均值之差。若某族多数run的best epoch落在
    前10个epoch，或best点附近val曲线抖动>2pp，该族判为"选点不稳定"，在结果中
    单列，**不因此回退test选点**。
(g) **last-epoch对照**：训练结束后除best-val权重外，另用最后一个epoch权重对test
    评估一次（不额外训练），两套test指标同时落盘。若两者结论一致，在报告中注明；
    不一致则如实列出差异，不做取舍。

**目前处于代码实现阶段，尚未动GPU。**

## 2026-09-19 更新：PU完成，BATCH5（Cross-condition+LOBO n=8+PU）全部完成

PU 7/7 configs全部完成，provenance核验通过（BAD: NONE）。中途经历一次宿主机整体
重启（WSL掉线，无数据损失，幂等设计仅需重跑1个未落盘的seed）和两次为[论文2]
延迟基准测试的计划内暂停，详见BATCH5_RESULTS.md"事故记录2"。

**P12主张判定**：三个噪声档（clean/0dB/−4dB）的kin-vs-nokin方向在leak-free后
不稳定（部分翻转），但**全部不显著**（p≥0.125）——与原负结论（L_kin在PU离散
转速下无统计显著效果）完全一致，符号不稳定+始终不显著是"无真实效应"的典型
指纹。**按P12预注册规则：负结论未变→直接换数字，不改叙事，未触发"转正向→
停止升报"条款。**

**BATCH5全景**：Cross-condition（16族，P10判定"基本不变"档，+38.4pp→leak-free
+21.7pp/+22.9pp）→ LOBO n=8（4族，P11判定情形(a)，负迁移p=0.0078有统计支撑）
→ PU（7族，P12判定负结论不变）——三个子任务全部完成，均全量provenance核验
通过，BATCH5至此收尾。

## 2026-09-20 更新：BATCH6（收尾）P13——CWRU dual-BM2 leak-free补跑，跑前写

`tab:regime`的CWRU行此前从未以leak-free协议跑过dual-BM2（`EDITS_2_applied.md`
"偏离清单"一节已标记为数据缺口）。这张表承载的是`section4_experiments_sensors.tex`
`subsubsec:regime_table`（label P1）里已经写在正文的一条**跑前预注册规则**——
P13原样复制该规则，补上"两个regime各自的比较对象"和"检验方法"的具体细节，
作为这次补跑前按本项目"跑前先写prereg"纪律补记的正式记录（不是新造规则）。

**判定条件（原文照抄，`subsubsec:regime_table`）**：若dual-BM2在CWRU低SNR和
XJTU-SY跨工况**两个regime都**追平或超过dual-BM3，则论文backbone主张改写为
regime-agnostic（不分场景统一推荐）；若BM3在**至少一个**regime保持明显优势，
则维持现有"regime-dependent"叙事框架，该优势作为论证重点保留。

**两个regime各自的比较对象**：
- CWRU低SNR regime：dual-BM2 vs dual-BM3，**accuracy**（within-condition AWGN，
  −6/−8dB两点是原表已有的；本次新增−4dB点补齐网格）。
- XJTU-SY跨工况regime：dual-BM2 vs dual-BM3，**macro-F1**（Cond2→Cond3，已在
  `tab:e1b_backbone_agnostic`/`tab:xjtu`口径下完成，本条不重跑，直接复用：
  70.63±9.68 vs 89.72±9.98，n=8，p=0.0078，8/8）。

**检验方法**：exact Wilcoxon signed-rank，按seed配对，两侧（不预设方向）——
CWRU侧n=5（5个seed，理论下限p=0.063），XJTU侧沿用已有n=8结果（理论下限
p=0.0078）。"Range across per-seed paired differences"（而非均值差的单点range）
延续`tab:regime`既有†footnote的报告惯例。

**范围**：CWRU dual-BM2，{−4,−6,−8}dB × seeds 0-4，15个run。config从既有
dual-BM2（leaky协议）config直接派生，只改`results_dir`，其余超参数逐字复制
（P3纪律）。三件套核验（checkpoint+source_snapshot+disjointness_check）。

**结果处理**：不预设哪个backbone会赢——按上面"判定条件"机械套用，写入
`BATCH6_RESULTS.md`。若CWRU侧leak-free后方向反转（BM2不再明显领先，甚至BM3
反超），直接按新方向如实报告并相应改写`tab:regime`后的判定段落，不因为
"论文已经这么写了"而回避新结果。**目前处于此prereg记录阶段，尚未动GPU。**

## 2026-09-20 更新：P14——训练轴承数×传感器数覆盖度扫描（评审第3条，EDITS_3§四）

**目的**：把XJTU LOBO负迁移从"单轴承训练下观察到"变为"随训练轴承覆盖度变化
的函数"，隔离"训练轴承数"这一变量，固定测试轴承、固定工况、固定数据总量口径。
原始清单（`EDITS_3_protocol_BN_relwork.md`§四）给出的设计基于"Cond3共5个轴承"
的假设；下面先核对并修正这个前提，再据此重算run数。

**⚠️ 前提修正（与清单不同，需明确记录）**：Cond3实际有5个物理轴承
（Bearing3\_1~3\_5），但`Bearing3_2`是Mixed(OR+IR+ball+cage)故障，按本项目
D26既定协议（[internal project notes, not released] `B4-1.5`）**排除在分类任务之外**——它既不能作训练
数据（标签不干净）也不能作测试轴承（现有4折LOBO主表`tab:xjtu_lobo_strat`
本身就只有4折，没有第5折）。因此可用轴承实际是**4个**，不是清单假设的5个：
Bearing3\_1(OR,fold0)/Bearing3\_3(IR,fold1)/Bearing3\_4(IR,fold2)/
Bearing3\_5(OR,fold3)。固定1个作测试轴承后，训练组合池只剩**3个**轴承，
$k$的上限相应从4降到**3**，不是清单写的4。这不是新发现的bug，只是清单
撰写时没有对齐本项目自己已经在用的D26轴承排除规则，此处按项目既有事实
更正。

**固定测试轴承的选择（按清单委托"[redacted]按LOBO分层表选类别覆盖最完整
的一折"，判断依据见下）**：选 **Bearing3\_1（OR，fold 0）**。理由：
1. 本实验要解释的现象——LOBO负迁移——按`tab:xjtu_lobo_strat`表注，
   **几乎全部集中在OR折**（fold0 −48.0pp，fold3 −23.0pp），IR折
   （fold1/fold2）退化可忽略（+0.04pp / −4.6pp）。若固定测试轴承选IR折，
   覆盖度扫描很可能测不到任何随$k$变化的效应（IR折本来就没有负迁移可
   解释），实验会失去意义——因此测试轴承必须是两个OR轴承之一。
2. Bearing3\_1单传感器基线**99.54±0.82%**（fold0），接近天花板且方差极小，
   使得双传感器−48pp的退化信号干净、不与基线本身的噪声混淆；
   Bearing3\_5单传感器基线本身已经很弱且高方差（52.88±9.52%），拿它做
   固定测试轴承会让$k$变化时"精度变化"和"该轴承本身难分类"两个因素纠缠，
   不利于把"训练覆盖度"这一个变量干净地隔离出来。
   故选Bearing3\_1而非Bearing3\_5。

**修正后设计**：
- 数据：XJTU-SY Cond3，固定测试轴承=Bearing3\_1（写死，不再改）。
  训练组合池=剩余3个轴承：{Bearing3\_3(IR), Bearing3\_4(IR), Bearing3\_5(OR)}。
- 自变量：训练轴承数 $k \in \{1,2,3\}$。$k=1$：3个单轴承组合
  {3\_3}/{3\_4}/{3\_5}；$k=2$：3个双轴承组合
  {3\_3,3\_4}/{3\_3,3\_5}/{3\_4,3\_5}；$k=3$：1个组合（全部3个轴承，即用满
  剩余训练池，是这个扫描能达到的"覆盖度上限"，对应原清单里概念上的
  "$k=4$/最大覆盖度"角色，但在本项目的4轴承约束下实际就是$k=3$）。
  组合总数 $=3+3+1=7$。
- 臂：single（水平通道）/ dual，BM3 CE-only（不含L\_kin，与原清单一致，
  排除L\_kin作为混杂变量）。
- 选点：固定最终epoch（P10规则，与其余XJTU实验一致）。
- 指标：macro-recall（与LOBO主表同口径）。真实测试轴承只有OR一类，
  报告"该轴承的true single-class recall"（与`tab:xjtu_lobo_strat`的
  2×校正惯例一致，避免zero-guard假象再次出现）。
- run数：$7$组合 $\times 2$臂 $\times 5$ seeds $=$ **70 run**（原清单基于
  5轴承假设算出的150 run不成立，按修正后4轴承前提重算）。

**GPU-h估算**：复用BATCH5"LOBO n=8扩展"的实测吞吐——48 run（4臂×3新增
seed×4折）耗时6.16h，折合**7.7 min/run**。70 run × 7.7 min/run
$\approx$ **9.0 GPU-h**，低于清单设定的15 GPU-h降级阈值，**不触发降级方案**，
70-run设计按全量执行（批准后）。

**预注册假设与判定（跑前写死，按$k\in\{1,2,3\}$调整清单原文的4档判定，
概念对应关系：清单的"$k=4$/非负"在本设计中对应"$k=3$（训练池上限）"）**：
- H1：dual−single的差值随$k$单调上升（$k=1$为负，$k=3$为非负或零）。
  检验：对每个$k$报dual−single配对差（组合×seed池）与两侧精确Wilcoxon；
  趋势用$k$与差值的Spearman（精确置换，3点）。
- 判定档：
  (a) 差值随$k$单调上升且$k=1$显著为负、$k=3$非负 → 负迁移归因于训练
      覆盖度，C1的"do not fuse when each fold trains on a single bearing"
      升级为"fusion gain increases with the number of training bearings;
      negative at $k=1$"。
  (b) 单调但$k=3$（训练池上限）仍为负 → 负迁移在该工况下不随覆盖度
      消失，C1改为"negative across all tested training-bearing counts on
      this condition"；跨工况的正增益与之并列报告，不再暗示两者由同一
      变量解释。
  (c) 非单调 → 如实报，C1维持现有描述性表述，Limitations写明覆盖度不是
      决定变量。
  (d) $k=1$结果与LOBO n=8的fold0结果（single 99.54±0.82%, dual
      51.52±22.38%, Δ≈−48pp）方向不一致 → 停止，升报，先查协议差异
      （$k=1$理论上应约等于LOBO fold0的单轴承训练场景，只是种子数从
      n=8降到n=5、且这里的"训练轴承"取自{3\_3,3\_4,3\_5}而非LOBO fold0
      本身排除的其余3折——方向应一致，量级可能因种子数/具体轴承组合
      略有差异，但符号必须一致）。
- 无论哪档，不新增"显著点"主张；本实验目的是解释而非再添一条显著性。
- 结果写入§4 XJTU小节新增"Training-coverage sweep"段 + 一张按$k$分组的表；
  LOBO表注引用它作为混杂变量的隔离证据。

**执行前置（严格遵守，不得跳过）**：本条目已完成"P14写入prereg + 报run数
GPU-h"两步（本节）。**下一步是向用户报告70 run / ≈9.0 GPU-h的估算，等待
用户批准；批准前不得smoke test、不得动GPU、不得生成config/run脚本。**
批准后的执行顺序：smoke（临时scratch目录，不进`results_dir`）→ 三件套
核验通路 → 全量70 run → 三件套核验 → 按上方判定档机械套用 → 写入
`BATCH_P14_RESULTS.md`（文件名待定，视批准时的批次编号而定）。

## 2026-09-28 事后备注：判定档(c)的适用范围收窄

70 run跑完后复核设计发现：训练池{Bearing3_3(IR), Bearing3_4(IR),
Bearing3_5(OR)}里只有**一个**OR轴承，而Cond3全部可用轴承里OR类型总共也
只有2个（Bearing3_5 + 被固定为测试轴承的Bearing3_1）。这意味着本设计
**从一开始就不可能把OR覆盖度作为自变量来扫描**——$k$从1升到3，实际改变
的是"训练集里加了几个IR轴承"，OR轴承数量在所有非退化组合里始终固定
为1（唯一的例外是k=1里恰好抽到Bearing3_5的那种情况，同样只有1个OR）。

这个设计局限直接影响判定档(c)的解读范围：跑出来的"负迁移不随k改善、
甚至在k=3最差"这个结果，只能证明**加同故障类型(IR)轴承救不了OR折的
负迁移**，不能证明"加更多OR轴承也救不了"——后者这个问题本设计压根没
测过，因为没有第二个OR轴承可用来测。论文正文（`tab:p14_coverage`表注、
`tab:xjtu_lobo_strat`表注、§V-E3(c)段落、discussion对应段、结论、摘要、
C1条目）已按这个收窄后的范围改写，统一表述为"加IR轴承救不了；OR覆盖度
本身没法在这个工况下变化，这个问题仍然开放"，不再用"training coverage
does not diminish / ruling out insufficient bearing count"这种听起来
覆盖了"任何轴承数"的过宽表述。

若未来想真正回答"OR覆盖度本身是否是决定变量"，需要换一个OR轴承数≥3的
工况/数据集重新设计——Cond3本身的轴承构成不支持这个问题。
