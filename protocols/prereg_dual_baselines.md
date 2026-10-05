# 预注册：dual-baseline 扩展臂（dual-BM2 / dual-CNN 覆盖核对 / cnn1d_attnfusion）

日期：2026-07-08
撰写人：[redacted]（本轮任务，`e1_dual_baselines` 系列）
状态：先于任何 config/run 写入，mtime 早于本轮所有其他新产物

## 规则来源与逐字继承内容（P1）

### CWRU dual-sensor 网格（逐字继承自 dual-BM3，不自创）

来源文件：`experiments/exp_b2_dual_sensor/run_b2_snr_curve.py`（`SNR_GRID` 变量）+
`experiments/exp_b2_dual_sensor/config_dual_{nokin,kin}.yaml` + [internal project notes, not released] B2 小节

- **SNR 网格**：`[-8, -6, -4, -2, 0]` dB（5 点，EAAI 网格子集）
  - ⚠️ **不含 +10dB**：[internal project notes, not released] B2 明确记录 "+10dB 排除：单传感器已达 100.00%，
    双传感器增益空间为零，无统计意义（需论文脚注说明）"——本轮沿用同一排除理由，
    不为新臂重新引入 +10dB 档位。
- **seed 集**：`[0, 1, 2, 3, 4]`（5 seeds，与全项目一致）
- **数据/模型段**（逐字抄自 `config_dual_nokin.yaml`）：
  ```yaml
  data_dir: <DATA_ROOT>/cwru_12k_de
  channels: [DE, FE]
  win_len: 2048
  stride: 1024
  val_ratio: 0.2
  batch_size: 64
  num_workers: 4
  d_model: 64
  d_state: 128        # mamba2 分支忽略此键，但保留以便与 BM3 侧配置逐字对齐
  n_layers: 4
  n_classes: 4
  conv_stride: 2
  epochs: 50
  lr: 3.0e-4
  weight_decay: 1.0e-4
  grad_clip: 1.0
  scheduler: cosine
  ```
- **λ_kin**：dual-BM2 臂只跑 **nokin**（`lambda_kin: 0.0`）。
  理由（D19 架构约束，已在 [internal project notes, not released] 风险登记记录）：Mamba-2 无 RoPE / 无 θ,Δt 激活，
  `baselines/mamba2.py::BearMamba2.forward` 的 `return_kin` 参数被忽略（恒返回
  `(logits, [])`），`experiments/exp01_cwru_baseline/train.py` L195
  `do_kin = (lambda_kin > 0) and (backbone != "mamba2")` 已对此做了架构级短路。
  因此不新写 dual-BM2 的 kin 配置——写了也不会真正生效，属于无意义产物。

### XJTU cross-condition dual 网格（逐字继承自 dual-BM3 cross）

来源文件：`experiments/exp_xjtu/config_cross_dual_nokin.yaml`

- **模式**：`cross`（Cond2→Cond3，D26 既定协议），**不含 LOBO**（本轮任务只要求 cross 臂）
- **seed 集**：`[0, 1, 2, 3, 4]`
- **数据/模型段**（逐字抄自 `config_cross_dual_nokin.yaml`）：
  ```yaml
  data_root: ~/data_xjtu/XJTU-SY_Bearing_Datasets
  train_condition: "37.5Hz11kN"   # Cond2: 2250rpm
  test_condition:  "40Hz10kN"     # Cond3: 2400rpm
  window_size: 2048
  n_sensors: 2          # horizontal + vertical (B4-5 dual-channel)
  d_model: 64
  d_state: 128
  n_layers: 4
  n_classes: 2          # OR=0, IR=1
  conv_stride: 2
  batch_size: 64
  num_workers: 4
  epochs: 50
  lr: 3.0e-4
  weight_decay: 1.0e-4
  grad_clip: 1.0
  scheduler: cosine
  fs_eff: 12800.0        # D26 mandatory assert (fs=25600 / conv_stride=2)
  ```
- **λ_kin**：`0.0`（同上，mamba2 无 L_kin 支持）

### dual-CNN 覆盖核对基准

来源目录：`experiments/exp_mext_e14_1dcnn_cwru_dual/config_snr{-8,-6,-4,-2,0}.yaml`
（逐一读取核对，channels=[DE,FE], seeds=[0,1,2,3,4]，SNR 覆盖 `{-8,-6,-4,-2,0}`）
→ 与 dual-BM3 网格完全一致，参见任务 2 的核对报告（`e1_config_audit.md`）。

### cnn1d_attnfusion 新臂网格

与 dual-CNN 相同的 CWRU 5-SNR × 5-seed 网格（同一基准比较臂），channels=[DE,FE]。
不新增 XJTU 配置（任务未要求 attnfusion 跑 XJTU）。

## 护栏重申（不可违反）

- 本轮 loop 内任何训练命令 **epoch ≤ 2 且 cell/step ≤ 2**（`--smoke` 模式）。
- 所有新产物写入带时间戳的新目录（`results/e1_dual_baselines_<YYYYMMDD-HHMM>/`），
  不覆盖、不删除任何已有 `results/**` 文件。
- 决策/结果 json 写完后立即 `chmod 444`。
- 若必须修改既有实现代码（本轮预期：`experiments/exp_xjtu/train.py` 的 `build_model`
  分派缺 mamba2 分支），在报告中显式声明改了哪一行、为何改。
- 不跑满量训练；不写主张/结论；不动单通道既有臂与 BM3 双通道既有臂。
