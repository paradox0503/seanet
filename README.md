# SEAnet

SEAnet is a novel architecture especially designed for data series representation learning (DEA).

Codes were developed and tested under Linux environment.

## Train SEAnet

1. _**Compile Coconut Sampling**_

```bash
cd lib/
make
```

2. **Add a configuration file**

An example configuration for SEAnet is given in *conf/example.json*.
Two fields with *TO_BE_CHANGED* are required to get changed.

> **database_path**: indicates the dataset to be indexed \
> **query_path**: indicates the query set

Other fields could be left by default.
Please refer to *util/conf.py* for all possible configurations.

3. **Train SEAnet**
conda activate jlh
export CUDA_VISIBLE_DEVICES=4
conda activate /home/liangzhiyu/miniconda3/jlh
```bash
python run.py -C conf/example.json
```

### 课程学习消融：同训练预算随机基线

从项目根目录运行两组实验：

```bash
# 原有 loss 课程学习（旧配置省略 training_schedule 时也使用此模式）
python run.py -C conf/example.json

# 非课程学习：每轮从全部 batch 中随机无放回采样
python run.py -C conf/random_baseline/example.json
```

`training_schedule` 只接受 `loss_curriculum` 和 `random_baseline`。
随机基线不执行初始 loss 评估、排序或分阶段难度筛选，所有 epoch 的候选池均为全部原始 batch。
两组每轮都训练 `max(1, total_num_batches // 10)` 个 batch，默认 100 个 epoch，
共用模型、loss、优化器、query 构造和验证代码。这里的“同预算”是正式训练的更新 batch 数相同，
不是总耗时相同：课程版仍多一次无梯度难度评估。每轮也不是遍历整个数据集。

两份配置除调度、实验名称和输出位置外，其余超参数相同。两组共享 `conf/samples/` 中的训练/验证缓存；
对照实验应保留相同缓存，不要分别重新抽取数据。现有随机 mask 和 seed 行为没有改变，
相同配置 seed 不代表所有随机操作已完全可复现。

输出隔离：

- 课程版继续在 `conf/` 下保存默认 checkpoint、`fit.log` 和 `0time.log`。
- 随机基线在 `conf/random_baseline/` 下保存 checkpoint、`fit.log`、`0time.log`、导出的 `our_pretrain.pkl` 和配置快照。
- 预训练结束后的编码器保存位置现在遵循配置中的 `pickle` 字段；课程版示例保留了原服务器绝对路径，运行前请确认该路径适合当前机器。

注意：同一实验目录内已有 checkpoint 时，程序仍会自动续训。全新消融实验必须使用没有旧 checkpoint 的输出目录；
先自行备份已有结果，不要让课程版与随机版指向同一个目录。相对路径均按运行时的项目根目录解析。

不依赖 PyTorch 的调度与控制流测试：

```bash
python -m unittest discover -s tests -v
python -m compileall -q run.py util model tests
```

控制流测试执行实际的配置/训练入口方法，以轻量替身代替 GPU 运算；它们不等同于真实 PyTorch 训练验证。

## Approximate Similarity Search

The indexing and query answering of DEA is in https://github.com/qtwang/isax-modularized

## Cite this work

```latex
@inproceedings{kdd21-Wang-SEAnet,
  author    = {Wang, Qitong and
               Palpanas, Themis},
  title     = {Deep Learning Embeddings for Data Series Similarity Search},
  booktitle = {{KDD} '21: The 27th {ACM} {SIGKDD} Conference on Knowledge Discovery
               and Data Mining, Virtual Event, Singapore, August 14-18, 2021},
  publisher = {{ACM}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3447548.3467317},
  doi       = {10.1145/3447548.3467317},
  timestamp = {Thu, 05 Aug 2021 09:46:47 +0800}
}
```

