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

```bash
python run.py -C conf/example.json
```

## 多数据集 residual 补零基准线

复制 `conf/multi_residual.example.json` 到独立的实验目录（例如
`experiments/mixed/config.json`），填写 `datasets` 中各个文件的路径和原始
`dim_series`，然后运行：

```bash
python run.py -C experiments/mixed/config.json
```

这里的 `data96`、`data128`、`data256` 是占位名称，不代表截图中文件的实际维度。
文件格式沿用原程序：无头信息、连续存储的 float32。此格式无法可靠地自动推断记录
维度，所以每个数据集必须填写原始维度；最外层的 `dim_series` 固定为 256。

读取时按各自原始维度切分记录，96 维末尾补 160 个零、128 维末尾补 128 个零，
256 维直接使用。合并训练样本后，由现有 DataLoader 打乱，训练同一个 residual
模型。没有额外的输入归一化、插值、掩码或损失屏蔽；保留模型已有的归一化层，
重建和距离损失直接使用完整的 256 维向量。超过 256 维时报错，不截断数据。
编码器也支持直接传入 `[batch, 1, 原始维度]` 的张量并在末尾补零。

顶层 `size_train`、`size_val` 表示**每个数据集**的抽样数，各条目可以单独覆盖。
例如三个数据集各抽 20000 条，合并后训练集为 60000 条。沿用均匀有放回抽样，
训练与验证样本可能重复。多数据集模式按随机种子重新抽样，不使用旧的单数据集
样本缓存，也不修改或预先转换原始 `.bin` 文件；仅将抽样结果补零后放入内存。
`size_db` 可在每个条目内填写来限制读取的记录范围，省略则按文件大小计算。

默认示例只训练。需要训练后导出向量时，把顶层 `to_embed` 设为 `true`，并为
每个条目补上 `query_path`，可选填 `size_query`。同一个模型会分别编码各数据集
及其查询集，默认输出 `<dataset_name>-database-embedding.bin` 和
`<dataset_name>-query-embedding.bin`，也可在条目内指定 `db_embedding_path`
和 `query_embedding_path`。输出维度是 `dim_embedding`。查询集应与对应数据库
使用相同的原始维度。按照现有配置逻辑，默认日志、模型和导出结果写在配置文件
所在目录；建议每次实验使用独立目录，避免加载先前的模型。

不配置 `datasets` 时仍使用原来的单数据集流程。

验证（需要 NumPy、PyTorch 和 Matplotlib）：

```bash
python -m unittest discover -s tests
```

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
