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


# def loadTrainValCocunut(dataset_name, dataset_path, dataset_size, train_size, val_size, series_length=256, sax_length=16, sax_cardinality=8):
def sample(conf: Configuration,path_db,train_indices_path,val_indices_path,dim_seq,size_train,size_val,size_db,train_path,val_path):#采样
    
    dataset_path = path_db
    # train_path =train_path
    # val_path =val_path
    # train_indices_path = val_indices_path
    # val_indices_path = val_indices_path
    dim_series = dim_seq
    # size_train = size_train
    # size_val = size_val
    # size_db = size_db

    
    os.makedirs(Path(train_path).parent, exist_ok=True)
    os.makedirs(Path(val_path).parent, exist_ok=True)
    os.makedirs(Path(train_indices_path).parent, exist_ok=True)
    os.makedirs(Path(val_indices_path).parent, exist_ok=True)

    dim_coconut = conf.getHP('dim_coconut')

    
    sampling_method = conf.getHP('sampling_name')

    if sampling_method == 'coconut':
        if not (os.path.exists(train_indices_path) and isfile(train_indices_path)) or not (os.path.exists(val_indices_path) and isfile(val_indices_path)):
            c_functions = CDLL(conf.getHP('coconut_libpath'))

            return_code = c_functions.sample_coconut(c_char_p(dataset_path.encode('ASCII')), 
                                                    c_long(size_db),
                                                    c_char_p(train_indices_path.encode('ASCII')), 
                                                    size_train,
                                                    c_char_p(val_indices_path.encode('ASCII')), 
                                                    size_val, 
                                                    dim_series, 
                                                    conf.getHP('coconut_cardinality'),
                                                    dim_coconut)
            dlclose(c_functions._handle)
            
            if return_code != 0:
                print(return_code)
    elif sampling_method == 'uniform':
        if not (os.path.exists(train_indices_path) and isfile(train_indices_path)) or not (os.path.exists(val_indices_path) and isfile(val_indices_path)):
            # print("inner")
            # train_sample_indices = np.random.randint(0, int(size_db/2), size=int(size_train/2), dtype=np.int64)
            # val_samples_indices = np.random.randint(0, int(size_db/2), size=int(size_val/2), dtype=np.int64)
            # train_sample_indices1= np.random.randint(int(size_db/2), size_db, size=int(size_train/2), dtype=np.int64)
            # val_samples_indices1= np.random.randint(int(size_db/2), size_db, size=int(size_val/2), dtype=np.int64)
            # # print(type(train_sample_indices))
            # # print(train_sample_indices.shape)
            # train_sample_indices = np.concatenate((train_sample_indices1, train_sample_indices))
            # val_samples_indices = np.concatenate((val_samples_indices1, val_samples_indices))
            
            # print(train_sample_indices.shape)
            # exit()
            # print(size_train)
            train_sample_indices = np.random.randint(0, size_db, size=size_train, dtype=np.int64)
            val_samples_indices = np.random.randint(0, size_db, size=size_val, dtype=np.int64)
            
            
            # train_sample_indices = np.random.randint(0, int(size_db/5), size=int(size_train/5), dtype=np.int64)
            # val_samples_indices = np.random.randint(0, int(size_db/5), size=int(size_val/5), dtype=np.int64)
            # train_sample_indices1= np.random.randint(int(size_db/5), int(2*size_db/5), size=int(size_train/5), dtype=np.int64)
            # val_samples_indices1= np.random.randint(int(size_db/5),  int(2*size_db/5), size=int(size_val/5), dtype=np.int64)
            # train_sample_indices2= np.random.randint(int(2*size_db/5), int(3*size_db/5), size=int(size_train/5), dtype=np.int64)
            # val_samples_indices2= np.random.randint(int(2*size_db/5),  int(3*size_db/5), size=int(size_val/5), dtype=np.int64)
            # train_sample_indices3= np.random.randint(int(3*size_db/5), int(4*size_db/5), size=int(size_train/5), dtype=np.int64)
            # val_samples_indices3= np.random.randint(int(3*size_db/5),  int(4*size_db/5), size=int(size_val/5), dtype=np.int64)
            # train_sample_indices4= np.random.randint(int(4*size_db/5), size_db, size=int(size_train/5), dtype=np.int64)
            # val_samples_indices4= np.random.randint(int(4*size_db/5),  size_db, size=int(size_val/5), dtype=np.int64)            
            # train_sample_indices = np.concatenate((train_sample_indices,train_sample_indices1,train_sample_indices2,train_sample_indices3,train_sample_indices4))
            # val_samples_indices = np.concatenate((val_samples_indices,val_samples_indices1,val_samples_indices2,val_samples_indices3,val_samples_indices4))


            train_sample_indices.tofile(train_indices_path)
            val_samples_indices.tofile(val_indices_path)
    else:
        raise ValueError('sampling {:s} is not supported'.format(sampling_method))

    train_sample_indices = np.fromfile(train_indices_path, dtype=np.int64)
    # print(len(train_sample_indices) )
    # print(size_train)
    assert len(train_sample_indices) == size_train
    
    loaded = []
    for index in train_sample_indices:
        sequence = np.fromfile(dataset_path, dtype=np.float32, count=dim_series, offset=4 * dim_series * index)

        if not np.isnan(np.sum(sequence)):
            loaded.append(sequence) 

    train_samples = np.asarray(loaded, dtype=np.float32)
    train_samples.tofile(train_path)
    train_samples = torch.from_numpy(train_samples)
            
    val_samples_indices = np.fromfile(val_indices_path, dtype=np.int64)
    assert len(val_samples_indices) == size_val

    loaded = []
    for index in val_samples_indices:
        sequence = np.fromfile(dataset_path, dtype=np.float32, count=dim_series, offset=4 * dim_series * index)

        if not np.isnan(np.sum(sequence)):
            loaded.append(sequence) 

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
