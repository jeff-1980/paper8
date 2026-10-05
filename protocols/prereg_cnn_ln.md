# 预注册：cnn1d_ln 臂（1D-CNN BatchNorm→GroupNorm 逐样本归一化消融）

日期：2026-07-09
撰写人：[redacted]（本轮任务，`e2_cnn_ln` 系列，接续 2026-07-08 的 `e1_dual_baselines`）
状态：先于任何 config/run 写入，mtime 早于本轮所有其他新产物（本文件写完立即 chmod 444）

## 规则来源与逐字继承内容（P2，任务原文，来源日期 2026-07-09）

> 0. 硬护栏：loop 内禁 >2ep/>2cell。workdir 自证：1D-CNN 模型定义与
>    `exp_mext_e13_1dcnn_xjtu_cross/` 存在。
> 0a. **预注册物化**：P2 规则逐字写入 `prereg_cnn_ln.md`（workdir 根，含来源日期），
>     mtime 早于一切新 run。
> 1. 实现 `cnn1d_ln` 臂：1D-CNN 的全部 8 处 BatchNorm1d 逐一替换为逐样本归一化
>    （推荐 `GroupNorm(1, C)` 即 conv 版 LayerNorm 等价物；语义要求：**不使用任何
>    batch 统计**）。其余结构（Conv/GELU/MaxPool/头）逐键不变——diff 清单仅限
>    归一化层。
> 2. 配置：a) XJTU Cond2→Cond3 跨工况（仿 exp_mext_e13 config，仅 backbone/归一化
>    异）——P2 的裁决格；b) CWRU within-condition SNR 网格同款 config（补 within 侧
>    证据链，评审2 点名）。SNR/seed 集逐字继承对应 BN 版配置。
> 3. 烟雾：cnn1d_ln 单格 clean 2ep×1seed，loss 降、无 NaN → `e2_unitcheck.json`
>    （含 diff 逐键清单 + 参数量 + "全项目 grep BatchNorm 在 ln 臂前向 0 命中"证据）。
> 4. 备 `run_cnn_ln.sh`（只写不跑）：XJTU cross × 5 seed + CWRU SNR 网格 × 5 seed；
>    输出 `results/e2_cnn_ln_<ts>/`；`status_e2.sh` glob 只匹配 `results/e2_cnn_ln_2*`。
> **明确不做**：不跑满量；不动 BN 版既有臂与 `results/**`；不写主张。

## GUARDRAILS（不可违反，任务原文逐字引用）

> 【GUARDRAILS 不可违反】只增不覆盖:所有新产物写进带时间戳的新目录
> (`<name>_<YYYYMMDD-HHMM>/`);严禁覆盖或删除任何已有 `results/` 文件;决策/结果
> json 写完立即 chmod 444;若必须修改既有实现代码,在报告里显式声明改了哪行、为何。
> 违反将被自动检测并回滚。

## 项：0 workdir 自证核查结果

- `baselines/cnn1d.py` 存在（BearCNN1D，8 处 `nn.BatchNorm1d(d_model)`：4 个
  `n_layers` block × 2 处/block）。
- `experiments/exp_mext_e13_1dcnn_xjtu_cross/config.yaml` 存在（backbone=cnn1d,
  mode=cross, XJTU Cond2→Cond3）。

## 项：1 归一化替换设计决策

- 替换目标层：`baselines/cnn1d.py` 中全部 8 处 `nn.BatchNorm1d(d_model)`。
- 替换为：`nn.GroupNorm(1, d_model)`（num_groups=1 ⇒ 对每个样本的全部 C 个通道
  联合归一化，逐样本统计量，不依赖 batch 维度，等价于 conv 场景下的 LayerNorm）。
- 新文件 `baselines/cnn1d_ln.py`，类名 `BearCNN1D_LN`，与 `baselines/cnn1d.py` 的
  `BearCNN1D` 逐键相同（Conv1d kernel/stride/padding、GELU、MaxPool1d、
  conv_embed、classifier 全部不变），仅归一化层类型不同。
- 沿用既有消融先例：`baselines/onedcnn_nobn.py`（Phase 2b `cnn1d_nobn` 臂）已建立
  "复制 cnn1d.py + 改归一化/去归一化 + 独立文件" 的模式，本次沿用同一模式。

## 项：2 配置来源核对（逐字继承，不自创网格）

### 2a. XJTU Cond2→Cond3 跨工况

来源文件：`experiments/exp_mext_e13_1dcnn_xjtu_cross/config.yaml`（逐字复制，仅改
`name` / `backbone: cnn1d_ln` / `results_dir`）：
- `mode: cross`, `train_condition: "37.5Hz11kN"`（Cond2, 2250rpm）,
  `test_condition: "40Hz10kN"`（Cond3, 2400rpm）
- `window_size: 2048`, `d_model: 64`, `n_layers: 4`, `n_sensors: 1`, `n_classes: 2`,
  `conv_stride: 2`
- `batch_size: 64`, `num_workers: 4`, `epochs: 50`, `lr: 3.0e-4`,
  `weight_decay: 1.0e-4`, `grad_clip: 1.0`, `scheduler: cosine`
- `lambda_kin: 0.0`（cnn1d 系列不支持 L_kin，与 BN 版一致）
- `fs_eff: 12800.0`（D26 强制 assert）
- `seeds: [0, 1, 2, 3, 4]`

### 2b. CWRU within-condition SNR 网格

来源文件：`experiments/exp07_baselines/config_cnn1d_{snr-8,snr-6,snr-4,snr-2,snr0,
clean}.yaml`（M8/exp07 六点网格，与 Phase 2b `exp_mext_e21b_1dcnn_nobn_cwru` 沿用
同一网格）——逐字复制除 `name`/`backbone: cnn1d_ln`/`results_dir` 外的全部字段：
- `data_dir: <DATA_ROOT>/cwru_12k_de`, `channels: [DE]`, `win_len: 2048`,
  `stride: 1024`, `val_ratio: 0.2`, `batch_size: 64`, `num_workers: 4`
- `d_model: 64`, `n_layers: 4`, `n_classes: 4`, `conv_stride: 2`
- `nhead: 4`, `dim_feedforward: 256`（cnn1d 分支忽略，保留以便与源 config 逐键对齐）
- `epochs: 50`, `lr: 3.0e-4`, `weight_decay: 1.0e-4`, `grad_clip: 1.0`,
  `scheduler: cosine`
- SNR 网格：`{-8, -6, -4, -2, 0}` dB + `clean`（无 `noise_snr_db` 字段，对应
  A1/E21b 的 "+10dB" 网格点）
- `lambda_kin: 0.0`
- `seeds: [0, 1, 2, 3, 4]`

⚠️ 不新增、不删减网格点；不新增 XJTU LOBO 臂（任务只要求 cross）。

## 护栏重申（不可违反）

- 本轮 loop 内任何训练命令 **epoch ≤ 2 且 step/cell ≤ 2**（`--smoke` 模式，
  由既有 `train.py` 内部强制：`n_epochs = 2 if smoke else cfg["epochs"]` +
  `if smoke and step >= 2: break`）。
- 所有新产物写入带时间戳的新目录（`results/e2_cnn_ln_<YYYYMMDD-HHMM>/` 供满量
  队列聚合；烟雾测试用 `results/e2_smoke_<ts>/`，命名空间与 `e2_cnn_ln_*` 严格
  隔离，防止 `status_e2.sh` 误判，沿用 e1 阶段 reviewer F2/major 修复的隔离模式）。
- 不覆盖、不删除任何已有 `results/**` 文件。
- 决策/结果 json 写完后立即 `chmod 444`。
- 若必须修改既有实现代码，预期改动（本轮预注册即声明，执行后在完成报告中逐行核对）：
  - `experiments/exp01_cwru_baseline/train.py::build_model` — 新增 `cnn1d_ln`
    分支（仿 `cnn1d_nobn` 分支写法），必要：否则 `backbone: cnn1d_ln` 会静默
    落到默认 `mamba3` 分支。
  - `experiments/exp_xjtu/train.py::build_model` — 同上新增 `cnn1d_ln` 分支。
  - `experiments/exp_xjtu/train.py::train_one_run` 的 `do_kin` 判定元组
    （`backbone not in ("cnn1d","transformer1d","mamba2")`）追加 `"cnn1d_ln"`，
    与 `cnn1d`/`transformer1d`/`mamba2` 保持一致（防御性，本轮配置
    `lambda_kin=0.0` 已使 `do_kin` 恒为 False，此项非本轮必需但为正确性一致性
    追加，非行为变更）。
  - 以上均为**新增分支/新增元组项**，不改动任何既有分支的既有逻辑，不影响
    `cnn1d`/`cnn1d_nobn`/`cnn1d_attnfusion`/`mamba2`/`mamba3`/`transformer1d`
    既有臂的行为。
- 不跑满量训练；不写主张/结论；不动 BN 版既有臂（`baselines/cnn1d.py` 不改）与
  既有 `results/**`。
