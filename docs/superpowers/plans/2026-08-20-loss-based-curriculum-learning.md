# 基于 Loss 的累计式课程学习实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将人工分段方差排序替换为首次无梯度总 loss 排序，并按从易到难的累计候选池进行正式训练。

**Architecture:** 在 `Experiment` 中新增一个只负责计算单个 batch 难度 loss 的私有方法，首次遍历全部训练 batch 时调用该方法完成排序。正式训练循环保留原有每轮 batch 数量，将阶段选择从“当前单一区间”改为“截至当前阶段的全部区间”，并用原始 batch 索引维护数据集来源映射。

**Tech Stack:** Python、PyTorch、现有 `Experiment` 训练框架。

## Global Constraints

- 首次难度评估不得反向传播、更新优化器、调整学习率或增加 `self.epoch`。
- 难度 loss 必须等于正式 `linearlycombine` 训练中的 `trans_error + orth_term + recons_term + return_l2`。
- 前 80 个正式训练 epoch 分成 8 个阶段，每 10 个 epoch 累计开放一个难度区间。
- 每轮训练 batch 数保持为 `max(1, total_num_batches // 10)`。
- 80 个 epoch 后继续从全部 batch 中随机抽取。
- 不执行训练、单元测试或语法运行检查，只做静态 diff 和逻辑复核。

---

## 文件结构

- Modify: `util/experiment.py`：新增难度 loss 计算、替换排序指标、实现累计候选池并修正原始索引映射。
- No test file：按用户要求不建立依赖本地运行环境的测试流程。

### Task 1: 新增首次 batch 难度评估

**Files:**
- Modify: `util/experiment.py`

**Interfaces:**
- Consumes: `batch: torch.Tensor`、`alpha: float`、现有模型与四项 loss 配置。
- Produces: `Experiment.__calculate_curriculum_loss(batch, alpha) -> float`。

- [ ] **Step 1: 在 `shuffle_batch_inter` 后新增难度 loss 方法**

```python
def __calculate_curriculum_loss(self, batch: torch.Tensor, alpha: float) -> float:
    db_batch = batch[torch.randperm(batch.size(0))]
    query_batch1 = batch[torch.randperm(batch.size(0))]
    query_batch2 = batch[torch.randperm(batch.size(0))]

    query_embedding1 = self.model.encode(query_batch1)[0]
    query_embedding2 = self.model.encode(query_batch2)[0]
    db_embedding, db_orig = self.model.encode(db_batch)

    trans_error = self.trans_loss(
        alpha,
        db_batch,
        query_batch1,
        query_batch2,
        db_embedding,
        query_embedding1,
        query_embedding2,
    )
    return_l2 = mean(self.__l2(squeeze(db_orig), squeeze(db_batch))) * self.__conf.getHP("func_b")

    if self.encoder_only:
        recons_term = torch.zeros(1).to(self.device)
    else:
        db_reconstructed = self.model.decode(db_embedding)
        recons_term = self.recons_weight * self.recons_reg(db_batch, db_reconstructed)

    loss = trans_error + self.__orth_reg() + recons_term + return_l2
    return loss.detach().item()
```

- [ ] **Step 2: 约束调用环境**

调用方必须先验证 `train_type == "linearlycombine"`，并将全部调用放入 `torch.no_grad()`。方法自身不访问优化器，不执行 `backward()`，不修改 epoch。

- [ ] **Step 3: 静态检查 loss 公式一致性**

对照 `__train` 中正式训练公式，确认四项名称和权重完全一致：转换损失、正交项、解码重构项、编码器重构项。

### Task 2: 用首次 loss 替换人工方差排序

**Files:**
- Modify: `util/experiment.py`

**Interfaces:**
- Consumes: `self.train_total_loader` 和 Task 1 的 `__calculate_curriculum_loss`。
- Produces: 按 loss 升序排列的原始 batch 索引 `total_indices: list[int]`。

- [ ] **Step 1: 删除运行入口中的方差排序块**

删除 `batch[0]`、`batch[-1]`、`calculate_sample_variance` 和 `batch_metrics` 的方差计算调用；保留无关方法不影响本次训练入口。

- [ ] **Step 2: 添加首次无梯度难度评估**

```python
if self.__conf.getHP('train_type') != 'linearlycombine':
    raise ValueError("loss-based curriculum requires train_type='linearlycombine'")

if not self.train_total_loader:
    raise ValueError('cannot build curriculum from an empty training set')

alpha = self.__conf.getHP('alpha')
was_training = self.model.training
self.model.eval()
try:
    with torch.no_grad():
        batch_metrics = [
            (self.__calculate_curriculum_loss(batch, alpha), idx)
            for idx, batch in enumerate(self.train_total_loader)
        ]
finally:
    self.model.train(was_training)

for loss_value, batch_index in batch_metrics:
    if not math.isfinite(loss_value):
        raise ValueError(f'non-finite curriculum loss at batch {batch_index}: {loss_value}')

batch_metrics.sort(key=lambda item: item[0])
total_indices = [idx for _, idx in batch_metrics]
```

- [ ] **Step 3: 输出评估摘要**

输出总 batch 数、最小 loss、最大 loss 和每轮抽取数量；不得输出单个样本或完整训练数据。

### Task 3: 改为固定预算的累计课程池

**Files:**
- Modify: `util/experiment.py`

**Interfaces:**
- Consumes: Task 2 的 `total_indices`。
- Produces: 每个正式 epoch 的原始 batch 索引 `selected_indices: list[int]`。

- [ ] **Step 1: 用累计候选池替换单区间逻辑**

```python
if self.epoch < 80:
    stage_index = min(self.epoch // 10, 7)
    pool_end = max(1, math.ceil(total_num_batches * (stage_index + 1) / 8))
    curriculum_pool = total_indices[:pool_end]
else:
    stage_index = None
    curriculum_pool = total_indices

if len(curriculum_pool) <= batch_size_per_round:
    selected_indices = curriculum_pool
else:
    random_positions = torch.randperm(len(curriculum_pool))[:batch_size_per_round].tolist()
    selected_indices = [curriculum_pool[position] for position in random_positions]
```

- [ ] **Step 2: 添加阶段日志**

仅当 `self.epoch < 80 and self.epoch % 10 == 0` 时打印阶段编号、累计候选池 batch 数及总 batch 数，避免每轮重复输出。

- [ ] **Step 3: 修正数据集来源映射**

将排序位置映射替换为原始索引映射：

```python
dataset_index = self.index_list[selected_indices[i]]
self.train_list[dataset_index].append(i)
self.train_query_list[dataset_index].append(i)
```

删除 `self.index_list_total`，避免把原始 batch 索引误当作排序后的位置。

### Task 4: 静态复核和交付

**Files:**
- Review: `util/experiment.py`

**Interfaces:**
- Consumes: Tasks 1-3 的最终 diff。
- Produces: 不依赖运行环境的静态审查结论。

- [ ] **Step 1: 检查变更范围**

运行 `git diff -- util/experiment.py`，确认没有模型、配置、验证或检查点的无关修改。

- [ ] **Step 2: 检查禁止出现在评估阶段的调用**

人工检查首次评估块中不存在 `optimizer.step()`、`backward()`、`__adjust_lr()`、`__adjust_wd()` 或 `self.epoch += 1`。

- [ ] **Step 3: 检查索引语义**

确认 `total_indices`、`curriculum_pool` 和 `selected_indices` 始终保存原始 batch 索引，只有随机位置变量用于访问候选池。

- [ ] **Step 4: 检查 epoch 行为**

确认首次 loss 评估发生在 `while self.epoch < self.max_epoch` 之前，正式循环仍完整执行配置的 `max_epoch` 次训练。
