# 同计算预算随机基线消融实验设计

## 目标

为现有基于初始 loss 排序的累计式课程学习增加一个非课程学习基线。基线与课程版保持相同的正式训练轮数、每轮 batch 数、模型、总 loss、优化器、验证和检查点流程，只移除初始难度评估、loss 排序与分阶段开放候选池。

该基线用于回答一个单独问题：在正式训练计算预算不变时，基于 loss 的课程调度是否优于从全体训练 batch 中均匀随机采样。

## 范围

- 保留现有课程学习行为，并将其作为默认训练调度。
- 新增可配置的随机基线调度。
- 新增一份独立的随机基线实验配置，避免覆盖课程实验配置。
- 抽取轻量、无 PyTorch 依赖的 batch 调度逻辑，以便在当前本地环境执行单元测试。
- 不改变模型结构、数据内容、batch 划分、训练总 loss、优化器、学习率、权重衰减、SRIP、验证或 checkpoint 逻辑。
- 不修复本次审查中发现的随机 mask、跨数据集 loss 归一化或 query batch 映射问题；两组实验继续共享这些行为，避免扩大消融变量。

## 配置接口

新增配置项 `training_schedule`，只接受两个值：

- `loss_curriculum`：保持现有行为。正式训练前计算每个 batch 的一次总 loss，按 loss 升序排列；前 80 个 epoch 分 8 个累计阶段开放候选池。
- `random_baseline`：跳过全部初始 loss 评估和排序。每个正式 epoch 的候选池始终是全部原始 batch。

`training_schedule` 的默认值为 `loss_curriculum`，因此没有该字段的旧配置保持兼容。现有 `conf/example.json` 显式设置为 `loss_curriculum`；新增 `conf/random_baseline/example.json`，除 `training_schedule`、实验名称和输出位置标识外，其余训练超参数与课程版一致。

随机基线配置单独放在 `conf/random_baseline/`，使 `Configuration` 生成的默认日志、checkpoint 和记录目录与课程版隔离。随机基线的 `pickle` 与 `result_path` 也指向独立目录，防止覆盖课程模型。预训练保存编码器时改为使用配置中的 `pickle` 路径，不再写死 `conf/our_pretrain.pkl`；课程版继续使用其原配置路径。

两种调度都要求预训练使用 `train_type="linearlycombine"`，继续复用当前训练目标。

## 调度组件

新增 `util/batch_schedule.py`，仅处理索引和整数，不导入 PyTorch。它提供三个职责明确的函数：

1. `batches_per_epoch(total_num_batches, divisor=10) -> int`
   - 返回 `max(1, total_num_batches // divisor)`。
   - 空训练集直接报错。

2. `candidate_batch_indices(schedule, ordered_indices, epoch, curriculum_epochs=80, stages=8) -> list[int]`
   - `random_baseline` 始终返回全部 `ordered_indices`。
   - `loss_curriculum` 保留现有累计前缀规则：每 10 个 epoch 开放一个阶段，最多 8 个阶段；第 80 个 epoch 以后返回全部索引。
   - 输入和输出都使用原始 batch 索引，不转换成排序位置。

3. `select_batch_indices(candidate_indices, batch_count, shuffled_positions) -> list[int]`
   - `shuffled_positions` 由调用方使用 `torch.randperm` 生成。
   - 候选池不大于预算时返回全部候选项；否则按给定随机位置选择固定数量。
   - 验证位置范围和唯一性，保证单个 epoch 内无放回采样。

该拆分保留 PyTorch 的随机采样来源，同时允许使用标准库测试调度规则。

## 训练数据流

### Loss 课程版

1. 构建 `train_total_loader` 和原始 `index_list`。
2. 在 `eval()`、`torch.no_grad()` 下计算每个 batch 的初始总 loss。
3. 生成按 loss 升序排列的原始 batch 索引。
4. 每个 epoch 根据累计课程阶段取得候选池。
5. 从候选池内无放回抽取固定预算的 batch。
6. 进入现有 query 构造、训练、验证和 checkpoint 流程。

### 随机基线

1. 构建相同的 `train_total_loader` 和原始 `index_list`。
2. 不调用 curriculum loss，不执行额外前向计算，不生成 loss 排名。
3. 使用 `range(total_num_batches)` 作为原始 batch 索引。
4. 每个 epoch 都以全部原始 batch 为候选池。
5. 从候选池内无放回抽取与课程版相同数量的 batch。
6. 进入完全相同的 query 构造、训练、验证和 checkpoint 流程。

## 计算预算

正式训练预算保持为：

```text
batch_size_per_round = max(1, total_num_batches // 10)
```

课程版和随机基线都训练配置中的 `num_epoch` 次。这里的“同计算预算”指正式反向传播的 batch 数相同；课程版用于生成难度排名的无梯度初始前向评估不计入正式训练预算，随机基线不会执行这次额外评估。

## 日志与错误处理

- 启动时记录 `training_schedule`、总 batch 数和每轮选择数量。
- `loss_curriculum` 保留 loss 范围和阶段候选池日志。
- `random_baseline` 输出一次明确日志，说明初始 loss 评估已跳过，候选池始终覆盖全部 batch。
- 空训练集、未知调度名称、非正采样除数、非法 epoch 或非法随机位置立即抛出明确异常。
- 两种实验使用互不相同的 checkpoint、日志和编码器 pickle 路径，随机基线不得自动加载课程版 checkpoint。

## 测试与验证

新增 `tests/test_batch_schedule.py`，使用 Python 标准库 `unittest`，覆盖：

- 课程版第 0、9、10、69、70、79、80 个零基 epoch 的候选池边界。
- 随机基线在第 0、50、99 个 epoch 始终返回全部索引。
- 两种模式的 `batches_per_epoch` 完全相同。
- 候选池大于预算时返回固定数量且无重复。
- 候选池小于等于预算时返回全部。
- 空训练集、未知调度、负 epoch、重复/越界随机位置的异常。

完成实现后执行：

```bash
python -m unittest discover -s tests -v
python -m compileall -q run.py util model tests
```

当前本地环境没有 PyTorch，因此不能执行真实模型训练。最终还需在训练服务器分别运行课程配置和随机基线配置，核对两组日志中的 epoch 数和每轮 batch 数完全一致，并确认随机基线日志中没有初始 loss 评估与课程阶段输出。

## 验收标准

- 旧配置不增加字段时仍运行现有 loss 课程学习。
- 随机基线配置在正式训练前不调用 curriculum loss。
- 随机基线每个 epoch 都从全部原始 batch 中随机无放回抽取。
- 两组模式每个 epoch 的训练 batch 数和正式 epoch 总数一致。
- 两组模式共用同一个 `__train`、`__validate` 和 checkpoint 处理代码。
- 两组配置的输出目录互相隔离，不能读取或覆盖对方的 checkpoint 与编码器 pickle。
- 调度单元测试全部通过，所有 Python 文件通过语法编译检查。
