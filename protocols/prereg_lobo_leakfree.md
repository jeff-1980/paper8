# 预注册：XJTU-SY LOBO 无泄漏选点协议（E3）

- 编号：E3（exp_e3_lobo_leakfree）
- 登记时间：2026-07-09 15:32 CST（本文件 mtime 必须早于本任务下任何 run 产物；违反视为协议破坏）
- 范围：仅新协议 `train_lobo_leakfree.py` + 新结果目录 `results/e3_lobo_leakfree_2*/`。
  不改动、不复用、不清空 `experiments/exp_xjtu/train.py` 及其既有 `results/exp_xjtu_lobo_*` 产物。
- 动机：现有 `train_one_run()`（`experiments/exp_xjtu/train.py` L259-263）在训练循环内部
  每个 epoch 都对 `test_loader` 做 `eval_recall_f1()`，并取全程 `best_macro_recall` /
  `best_macro_f1` 作为该折最终指标——这是按测试折表现挑 epoch，构成选点泄漏。
  本预注册规则在触碰任何真实数据前逐字登记，事后不得修改数值取舍规则。

## P1｜Fixed-epoch 选点（零选择）

- 每个 (fold, arm, seed) 的最终指标 **只能来自训练循环结束后的最后一个 epoch**
  （`epoch == n_epochs`，即 config 里的 `epochs` 字段；smoke 模式下 `n_epochs=2`）。
- 训练循环内部**不得**根据任何测试折指标更新"当前最优"状态（不得出现
  `best_macro_recall = max(...)`、`if macro_f1 > best_macro_f1` 之类逐 epoch 比较逻辑）。
- 配置项 `selection_mode` 必须显式为 `fixed_epoch`；脚本对其他取值一律 `assert` 失败退出，
  不允许静默回退到早停或按测试折选优的隐藏路径。
- 验收方式：`grep -n "selection_mode" train_lobo_leakfree.py` 只应命中断言与读取，
  不应出现任何形式的“选出历史最优 epoch”比较代码。

## P2｜测试折全程不可见

- `test_loader`（或 `test_ds`）在训练循环（`for epoch in range(1, n_epochs+1): ...`）
  内部**不得被访问**：不得在循环体内调用 `eval_recall_f1(model, test_loader, ...)`，
  不得读取测试折做可视化快照（I1 snapshot 机制在本脚本中整体移除，避免任何形式的
  训练期测试折访问）。
- 对测试折的唯一一次访问，是训练循环全部结束之后、且仅一次的
  `eval_recall_f1(model, test_loader, device)` 调用，用于产出 P1 所定义的最终指标。
- 验收方式（grep 证据）：
  `grep -n "eval_recall_f1(" train_lobo_leakfree.py` 命中数必须恰为 1 处调用点
  （函数定义本身不计），且该调用点所在代码行位于训练 `for epoch` 循环的**缩进层级之外**
  （用 `grep -n "for epoch in range" train_lobo_leakfree.py` 与调用行号比较可人工核验）。
- 训练用 `DataLoader`（`train_loader`）与测试用 `DataLoader`（`test_loader`）在数据构造阶段
  即已按 `make_lobo_folds()` 的 bearing 级划分互斥，本规则只约束"训练期间的模型/优化器
  是否访问测试折"，不重复定义已有的 bearing 级划分规则。

## P3｜Checkpoint 落盘

- 每个 (fold, arm, seed) 训练完成后（P1 定义的最终 epoch），若 `save_checkpoint: true`，
  必须把该次运行的模型权重存至
  `{results_dir}/checkpoints/{fold_tag}_seed{seed}.pt`，内容至少包含
  `model_state_dict`、`epoch`（等于 `n_epochs`）、`fold_tag`、`seed`。
- 这是一次性基础设施投入：本次及以后所有 E3 leak-free LOBO 训练均落盘 checkpoint，
  避免未来复现指标时需要重新训练。
- Checkpoint 写入**不得**影响 P1/P2：即不得为了保存"最优权重"而在训练期间提前
  访问测试折——只保存最终 epoch 的权重。

## 冻结声明

以上 P1/P2/P3 为本任务（E3）的全部选点与落盘规则，登记后不再修改。
若执行中发现规则需要调整，须新开一版 `prereg_lobo_leakfree_v2.md`，
不得就地编辑本文件（本文件写完后应被置为只读）。

本预注册**不包含**任何关于 L_kin 效果、单/双传感器优劣的主张——
这些结论留待满量运行完成后另行分析，本文件只登记协议本身。
