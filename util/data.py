# coding = utf-8
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os
import struct
import platform
import subprocess
from os.path import isfile
from pathlib import Path
from ctypes import CDLL, c_char_p, c_long
from _ctypes import dlclose

import torch
import numpy as np
from torch.utils.data import Dataset

from util.conf import Configuration


class TSDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, indices):
        return self.data[indices]



import numpy as np

def calculate_segment_variance_sums(train_data, segments=16):
    """
    计算train_data中每个元素分段后的方差和，明确按照先求均值再求平方差的步骤

    参数:
        train_data: 包含多个样本的列表，每个样本是一个numpy数组
        segments: 分段数量，默认为16

    返回:
        variance_sums: 每个样本的分段方差和列表
    """
    variance_sums = []

    for sample in train_data:
        # 确保样本长度可以被分段数整除
        sample_length = len(sample)
        if sample_length % segments != 0:
            raise ValueError(f"样本长度 {sample_length} 不能被 {segments} 整除")

        # 计算每段的长度
        segment_length = sample_length // segments

        # 初始化方差和
        total_variance = 0.0

        # 遍历每个分段并计算方差
        for i in range(segments):
            start = i * segment_length
            end = start + segment_length
            segment = sample[start:end]

            # 显式计算平均值
            segment_sum = np.sum(segment)
            mean = segment_sum / segment_length

            # 显式计算每个元素与平均值的平方差之和
            squared_diff_sum = np.sum((segment - mean) ** 2)

            # 计算该段的方差
            variance = squared_diff_sum / segment_length

            # 累加方差
            total_variance += variance

        variance_sums.append(total_variance)
        print(f"Sample variance sum: {total_variance}")

    return variance_sums

# 使用示例：
# 假设train_data已经通过你提供的代码加载完成
# variance_sums = calculate_segment_variance_sums(train_data)


def getSamples(conf: Configuration,path_db,train_indices_path,val_indices_path,dim_seq,size_train,size_val,size_db,train_path,val_path):
    dim_series =dim_seq
    # size_train = conf.getHP('size_train')
    # size_val = conf.getHP('size_val')
    device = conf.getHP('device')

    mode=conf.getHP('mode')
    # if mode=="pretrain":
    if 1:
        print("pretrain_data")
        train_path = train_path
        val_path = val_path

        if os.path.exists(train_path) and os.path.exists(val_path):


            train_samples = torch.from_numpy(np.fromfile(train_path, dtype=np.float32, count=dim_series * size_train))
            val_samples = torch.from_numpy(np.fromfile(val_path, dtype=np.float32, count=dim_series * size_val))
        else:
            if conf.getHP('sampling_name') == 'coconut' or conf.getHP('sampling_name') == 'uniform':
                train_samples, val_samples = sample(conf,path_db,train_indices_path,val_indices_path,dim_seq,size_train,size_val,size_db,train_path,val_path)
            else:
                raise ValueError('sampling {:s} is not supported'.format(conf.getHP('sampling_name')))

        if conf.getHP('encoder') == 'gru' or conf.getHP('encoder') == 'lstm':#or conf.getHP('encoder') == 'transformer':
            train_samples = train_samples.view([-1, dim_series, 1])
            val_samples = val_samples.view([-1, dim_series, 1])
        else:
            train_samples = train_samples.view([-1, 1, dim_series])
            val_samples = val_samples.view([-1, 1, dim_series])

        train_samples = train_samples.to(device)
        val_samples = val_samples.to(device)


    return train_samples, val_samples



def sample(conf: Configuration, path_db, train_indices_path, val_indices_path, dim_seq, size_train, size_val, size_db, train_path, val_path):
    dataset_path = path_db
    dim_series = dim_seq

    # 创建必要的目录
    os.makedirs(Path(train_path).parent, exist_ok=True)
    os.makedirs(Path(val_path).parent, exist_ok=True)
    os.makedirs(Path(train_indices_path).parent, exist_ok=True)
    os.makedirs(Path(val_indices_path).parent, exist_ok=True)

    dim_coconut = conf.getHP('dim_coconut')
    sampling_method = conf.getHP('sampling_name')

    if sampling_method == 'coconut':
        raise NotImplementedError("Coconut sampling is not implemented in this function.")

    elif sampling_method == 'uniform':
        if not (os.path.exists(train_indices_path) and isfile(train_indices_path)) or not (os.path.exists(val_indices_path) and isfile(val_indices_path)):
            # 生成0到1亿之间的随机整数作为索引
            train_sample_indices = np.random.randint(0, size_db, size=size_train, dtype=np.int64)
            val_samples_indices = np.random.randint(0, size_db, size=size_val, dtype=np.int64)

            # 对随机索引进行排序，实现顺序读取
            train_sample_indices.sort()

            # 保存排序后的索引
            train_sample_indices.tofile(train_indices_path)
            val_samples_indices.tofile(val_indices_path)

    else:
        raise ValueError(f'sampling {sampling_method} is not supported')

    # 加载训练集索引并验证
    train_sample_indices = np.fromfile(train_indices_path, dtype=np.int64)
    assert len(train_sample_indices) == size_train, f"训练集索引数量不匹配: {len(train_sample_indices)} vs {size_train}"

    # 按排序后的索引读取训练数据
    loaded = []
    for index in train_sample_indices:
        # 计算偏移量：每个float32占4字节，每个样本有dim_series个数据点
        sequence = np.fromfile(dataset_path, dtype=np.float32, count=dim_series, offset=4 * dim_series * index)

        # 过滤包含NaN的序列
        if not np.isnan(np.sum(sequence)):
            loaded.append(sequence)

    # 保存并转换为PyTorch张量
    train_samples = np.asarray(loaded, dtype=np.float32)
    train_samples.tofile(train_path)
    train_samples = torch.from_numpy(train_samples)

    # 加载验证集索引并验证
    val_samples_indices = np.fromfile(val_indices_path, dtype=np.int64)
    assert len(val_samples_indices) == size_val, f"验证集索引数量不匹配: {len(val_samples_indices)} vs {size_val}"

    # 按排序后的索引读取验证数据
    loaded = []
    for index in val_samples_indices:
        sequence = np.fromfile(dataset_path, dtype=np.float32, count=dim_series, offset=4 * dim_series * index)

        if not np.isnan(np.sum(sequence)):
            loaded.append(sequence)

    # 保存并转换为PyTorch张量
    val_samples = np.asarray(loaded, dtype=np.float32)
    val_samples.tofile(val_path)
    val_samples = torch.from_numpy(val_samples)

    return train_samples, val_samples



class FileContainer(object):
    def __init__(self, filename, binary=True):
        self.filename = filename
        self.binary = binary
        if self.binary:
            self.f = open(filename, "wb")
        else:
            self.f = open(filename, "w")

    def write(self, ts):
        if self.binary:
            s = struct.pack('f' * len(ts), *ts)
            self.f.write(s)
        else:
            self.f.write(" ".join(map(str, ts)) + "\n")

    def close(self):
        self.f.close()


#这段Python代码的实现功能是将原始数据（data_filepath）嵌入到某种编码空间（embedding_filepath）中。
#实现原理是通过训练好的模型（model）对原始数据进行编码，将编码后的结果保存到文件中。
def embedData(model, data_filepath, embedding_filepath, data_size, batch_size = 2000, original_dim = 256,
              embedded_dim = 16, device = 'cuda', is_rnn = False, encoder = ''):
    if encoder == 'gru' or encoder == 'lstm':
        is_rnn = True

    num_segments = int(data_size / batch_size)

    if data_size < batch_size:
        num_segments = 1
        batch_size = data_size
    else:
        assert data_size % batch_size == 0

    nan_replacement_original = np.array([0.] * original_dim).reshape([original_dim, 1] if is_rnn else [1, original_dim])
    nan_replacement_embedding = [0.] * embedded_dim

    writer = FileContainer(embedding_filepath)

    try:
        with torch.no_grad():
            total_nans = 0
            print("Embeding...")
            for segment in range(num_segments):
                if segment % 20 == 0:
                    print("segment",segment,"/",num_segments)
                # if segment>300:
                #     break
                batch = np.fromfile(data_filepath, dtype=np.float32, count=original_dim * batch_size, offset=4 * original_dim * batch_size * segment)

                # print(type(batch))
                if batch.shape[0] == 0:
                    continue  # 如果 x 是空的，跳过这一轮循环  #2024.10.31.6.21.新添加

                if is_rnn:
                    batch = batch.reshape([-1, original_dim, 1])
                else:
                    batch = batch.reshape([-1, 1, original_dim])

                nan_indices = set()
                for i, sequence in zip(range(batch.shape[0]), batch):
                    # print("batch",i)
                    if np.isnan(np.sum(sequence)):
                        nan_indices.add(i)
                        batch[i] = nan_replacement_original

                embedding = model.encode(torch.from_numpy(batch).to(device))[0].detach().cpu().numpy()

                for i in nan_indices:
                    embedding[i] = nan_replacement_embedding

                writer.write(embedding.flatten())

                total_nans += len(nan_indices)

            print('nans = {:d}'.format(total_nans))
    finally:
        writer.close()
