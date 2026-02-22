# coding = utf-8
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os
import io
import logging
from datetime import date
from timeit import default_timer as timer
import pandas as pd
import torch
import pickle
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import mean, squeeze
from model.initialization import LSUVinit
from model.normalization import getSRIPTerm
from model.loss import ScaledL2Recons, ScaledL2Trans
from model.builder import AEBuilder
from util.data import TSDataset, getSamples
from util.conf import Configuration
from util.data import embedData
import random
import math
from torch.nn import Module, PairwiseDistance

class DatasetConfig:
    def __init__(self, name, path_db, dim_seq, size_train, size_val, size_db):
        self.name = name
        self.path_db = path_db
        self.dim_seq = dim_seq
        self.size_train = size_train
        self.size_val = size_val
        self.size_db = size_db
class EmbedConfig:
    def __init__(self, name, dataset_path,query_path, dim_seq, size_query):
        self.name = name
        self.dataset_path = dataset_path
        self.query_path=query_path
        self.dim_seq = dim_seq
        self.size_query = size_query
DATASET_CONFIGS = [
    DatasetConfig("Astro00", "/data/user_jialinhan/data_big/astro-dataset.bin", 256, 2000, 10000, 100000000),
    # DatasetConfig("Deep1B", "/data/user_jialinhan/data_big/deep1b-dataset.bin", 96, 200000, 10000, 100000000),
    # DatasetConfig("F5", "/data/user_jialinhan/data_big/F5-dataset.bin", 256, 200000, 10000, 100000000),
    # DatasetConfig("F10", "/data/user_jialinhan/data_big/F10-dataset.bin", 256, 200000, 10000, 100000000),
    # DatasetConfig("origin", "/data/user_jialinhan/data_big/origin-dataset.bin", 256, 200000, 10000, 100000000),
    # DatasetConfig("sald", "/data/user_jialinhan/data_big/sald-dataset.bin", 128, 200000, 10000, 100000000),
    # DatasetConfig("seismic", "/data/user_jialinhan/data_big/seismic-dataset.bin", 256, 200000, 10000, 100000000)
]
embed_CONFIGS = [    #   database path                                               query path
    EmbedConfig("astro", "data_big/astro-dataset.bin",    "data_big/astro-query.bin",256,100),
    # EmbedConfig("deep1b", "data_big/deep1b-dataset.bin",    "data_big/deep1b-query.bin",96,1000),
    # EmbedConfig("F5", "data_big/F5-dataset.bin",    "data_big/F5-query.bin",256,1000),
    # EmbedConfig("F10", "data_big/F10-dataset.bin",    "data_big/F10-query.bin",256,1000),
    # EmbedConfig("origin", "data_big/origin-dataset.bin",    "data_big/origin-query.bin",256,1000),
    # EmbedConfig("sald", "data_big/sald-dataset.bin",    "data_big/sald-query.bin",128,1000),
    # EmbedConfig("seismic", "data_big/seismic-dataset.bin",    "data_big/seismic-query.bin",256,1000),
    ]

class Experiment:
    def __init__(self, conf: Configuration):
        self.__conf = conf
        self.has_setup = False
        self.epoch = 0

        self.device = conf.getHP('device')

        self.mode=conf.getHP('mode')
        if self.mode=='fine':
            self.max_epoch = self.__conf.getHP('fine_epoch')
        elif self.mode=='pretrain':
            self.max_epoch = self.__conf.getHP('num_epoch')
        elif self.mode=='embed':
            self.max_epoch = 0
        else:
            print("config mode error")
            exit()

        self.checkpoint_folder = conf.getHP('checkpoint_folder')
        self.checkpoint_postfix = conf.getHP('checkpoint_postfix')
        self.__l2 = PairwiseDistance(p=2).cuda()


    def loadModel(self, epoch: int = -1) -> nn.Module:
        assert epoch < self.max_epoch
        checkpoint_path = os.path.join(self.checkpoint_folder, str(epoch) + '.' + self.checkpoint_postfix)

        if epoch == -1:
            epoch = self.max_epoch
            checkpoint_path = os.path.join(self.checkpoint_folder, str(epoch) + '.' + self.checkpoint_postfix)

            while not os.path.exists(checkpoint_path) and epoch >= 0:
                epoch -= 1
                checkpoint_path = os.path.join(self.checkpoint_folder, str(epoch) + '.' + self.checkpoint_postfix)

            if epoch != -1:
                print('loading checkpoint at epoch {:d}'.format(epoch))
        else:
            assert os.path.exists(checkpoint_path)

        if epoch == -1:
            return None

        self.epoch = epoch

        model = AEBuilder(self.__conf)
        with open(checkpoint_path, 'rb') as fin:
            model.load_state_dict(torch.load(fin))

        return model


    def getSample(self, size: int = -1) -> torch.Tensor:
        _, val_samples = getSamples(self.__conf)

        if size == -1:
            return val_samples
        elif size <= self.__conf.getHP('size_val'):
            indices = torch.randperm(val_samples.shape[0])
            return val_samples[indices][: size].to(self.device)
        else:
            raise ValueError('cannot provide {:d} samples ({:d} valset)'.format(size, self.__conf.getHP('size_val')))


    def setup(self) -> None:
        self.has_setup = True

        #设置日志记录，包括文件名、模式、格式和级别等。
        logging.basicConfig(filename=self.__conf.getHP('log_filepath'),
                            filemode='a+',
                            format='%(asctime)s,%(msecs)d %(levelname).3s [%(filename)s:%(lineno)d] %(message)s',
                            level=logging.DEBUG,
                            datefmt='%m/%d/%Y:%I:%M:%S')

        self.logger = logging.getLogger(self.__class__.__name__)

        torch.manual_seed(self.__conf.getHP('torch_rdseed'))#设置随机数种子，以确保结果的可重复性。
        if self.device == 'cuda':
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.manual_seed_all(self.__conf.getHP('cuda_rdseed'))
            else:
                raise ValueError('cuda is not available')

        batch_size = self.__conf.getHP('size_batch')
        num_data_base = len(DATASET_CONFIGS)

        self.train_db_loader = []
        self.train_query_loader1 = []
        self.train_query_loader2 = []
        self.val_db_loader = []
        self.val_query_loader1 = []
        self.val_query_loader2 = []
        for config in DATASET_CONFIGS:
            size_train = int(config.size_train / num_data_base)
            size_val = int(config.size_val / num_data_base)
            train_samples, val_samples = getSamples(self.__conf, config.path_db,
                                                    f"conf/samples/{config.name}_train_indices.bin",
                                                    f"conf/samples/{config.name}_val_indices.bin",
                                                    config.dim_seq, size_train, size_val, config.size_db,
                                                    f"conf/samples/{config.name}_train_samples.bin",
                                                    f"conf/samples/{config.name}_val_samples.bin")

            self.train_db_loader.extend(DataLoader(TSDataset(train_samples), batch_size=batch_size, shuffle=True))

            self.train_query_loader1.extend(DataLoader(TSDataset(train_samples), batch_size=batch_size, shuffle=True))
            self.train_query_loader2.extend(DataLoader(TSDataset(train_samples), batch_size=batch_size, shuffle=True))
            self.val_db_loader.extend(DataLoader(TSDataset(val_samples), batch_size=batch_size, shuffle=True))
            self.val_query_loader1.extend(DataLoader(TSDataset(val_samples), batch_size=batch_size, shuffle=True))
            self.val_query_loader2.extend(DataLoader(TSDataset(val_samples), batch_size=batch_size, shuffle=True))
            print(config.name, "加载数据完毕")

        print("合并完毕")



        dim_series = self.__conf.getHP('dim_series')
        dim_embedding = self.__conf.getHP('dim_embedding')

        if self.__conf.getHP('to_scale_lc'):
            self.trans_loss = ScaledL2Trans(dim_series, dim_embedding, to_scale=True).to(self.device)
        else:
            self.trans_loss = ScaledL2Trans().to(self.device)

        if self.__conf.getHP('to_scale_lr'):
            self.recons_reg = ScaledL2Recons(dim_series, to_scale=True).to(self.device)
        else:
            self.recons_reg = ScaledL2Recons().to(self.device)

        self.model = self.loadModel()
        if self.model is None:
            self.model = AEBuilder(self.__conf)
            if self.__conf.getHP('mode')=='pretrain':
                self.model = self.__init_model(self.model, val_samples)
            else:
                # pass
                self.model = self.__init_model(self.model, val_samples)#初始化encoder和decoder
                pkl = self.__conf.getHP('pkl_file')# pkl只有encoder的参数
                with open(pkl, 'rb') as f:
                    enc = pickle.load(f)
                self.model._AEBuilder__encoder=enc#覆盖掉初始化的encoder

        self.optimizer = self.__getOptimizer()

        self.checkpoint_mode = self.__conf.getHP('checkpoint_mode')
        if self.checkpoint_mode != 'none':
            if self.checkpoint_mode == 'everyk':
                self.checkpoint_k = self.__conf.getHP('checkpoint_k')

        if self.__conf.getHP('if_record'):
            indices = torch.randperm(val_samples.shape[0])

            self.samples2plot = val_samples[indices][: self.__conf.getHP('num_record')].to(self.device)
            self.record_folder = self.__conf.getHP('record_folder')

        self.detch_query = self.__conf.getHP('train_detach_query')

        self.encoder_only = self.__conf.getHP('decoder') == 'none'
        if not self.encoder_only:
            self.recons_weight = self.__conf.getHP('recons_weight')

        self.orth_regularizer = self.__conf.getHP('orth_regularizer')
        if self.orth_regularizer == 'srip':
            if self.__conf.getHP('srip_mode') == 'fix':
                self.srip_weight = self.__conf.getHP('srip_cons')
            elif self.__conf.getHP('srip_mode') == 'linear':
                self.srip_weight = self.__conf.getHP('srip_max')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def run(self) -> None:
        if not self.has_setup:
            self.setup()

        self.__checkpoint(persist_model=False)

        mode=self.__conf.getHP('mode')

        if mode=="fine":
            print("fine")
            self.__model_change()
            while self.epoch < self.max_epoch:
                batches = [batch for batch in self.train_db_loader]
                indices = torch.randperm(len(batches))  # 随机打乱批次的索引
                self.train_db_loader = [batches[i] for i in indices]  # 根据打乱后的索引重新排列批次

                batches = [batch for batch in self.train_query_loader1]
                self.train_query_loader1 = [batches[i] for i in indices]  # 根据打乱后的索引重新排列批次

                batches = [batch for batch in self.train_query_loader2]
                self.train_query_loader2 = [batches[i] for i in indices]  # 根据打乱后的索引重新排列批次


                batches = [batch for batch in self.val_db_loader]
                indices = torch.randperm(len(batches))  # 随机打乱批次的索引
                self.val_db_loader = [batches[i] for i in indices]  # 根据打乱后的索引重新排列批次

                batches = [batch for batch in self.val_query_loader1]
                self.val_query_loader1 = [batches[i] for i in indices]  # 根据打乱后的索引重新排列批次

                batches = [batch for batch in self.val_query_loader2]
                self.val_query_loader2 = [batches[i] for i in indices]  # 根据打乱后的索引重新排列批次
                start = timer()

                #调整学习率和权重衰减
                self.__adjust_lr()
                self.__adjust_wd()
                #根据需要调整随机梯度规则
                if self.orth_regularizer == 'srip':
                    self.__adjust_srip()

                # if self.epoch==15:
                #     self.__model_change()

                self.epoch += 1
                #a=random（0.1,0.9）
                print("第",self.epoch,"周期ing")
                # func_a = 0.95
                func_a = self.model._AEBuilder__encoder.fuc
                self.__train(func_a)
                self.__validate(func_a)
                self.logger.info('e{:d} time = {:.3f}s'.format(self.epoch, timer() - start))
                self.__checkpoint()

            import pickle
            with open('conf/fine.pkl', 'wb') as f:
                pickle.dump(self.model._AEBuilder__encoder, f)
        else:
            import pickle

            # 从配置中获取 pickle 文件路径
            pickle_path = os.path.abspath(self.__conf.getHP('pickle'))  # 假设 getHP 方法返回的是一个有效路径

            # 确保路径有效性
            if not pickle_path:
                raise ValueError("配置中未找到 'pickle' 文件路径")

            try:
                # 打开并加载 pickle 文件
                with open(pickle_path, 'rb') as f:
                    enc = pickle.load(f)
                    self.model._AEBuilder__encoder = enc
                print("编码器加载成功")
            except FileNotFoundError:
                raise FileNotFoundError(f"指定的 pickle 文件 '{pickle_path}' 不存在")
            except pickle.UnpicklingError:
                raise ValueError(f"无法加载 pickle 文件 '{pickle_path}'，可能文件已损坏")


        #-------------------------------------------------------------------------------

        if self.__conf.getHP('to_embed'):
            import sys
            # 手动添加系统中fvcore的安装路径（你的路径）
            sys.path.append("/home/liangzhiyu/.local/lib/python3.8/site-packages")
            from fvcore.nn import FlopCountAnalysis, parameter_count
            import time
            # if self.__conf.getHP('to_embed'):
            # from fvcore.nn import FlopCountAnalysis, parameter_count
            inputs = torch.randn(256,1,256).to(self.device)
            flops = FlopCountAnalysis(self.model, inputs)
            params = parameter_count(self.model)
            print(f"Total FLoPs:{flops.total() / 1e9:.2f} G")
            print(f"Total Params: {sum(params.values()) / 1e6:.2f} M")
            # time.sleep (10000)
            for i in range(len(embed_CONFIGS)):
                print("query")
                main_path=self.__conf.getHP('main_path')
                result_path=self.__conf.getHP('result_path')
                query_path=os.path.abspath(main_path+embed_CONFIGS[i].query_path)
                query_embed_path=os.path.abspath(result_path+embed_CONFIGS[i].name+"-query.bin")
                database_path=os.path.abspath(main_path+embed_CONFIGS[i].dataset_path)
                database_embed_path=os.path.abspath(result_path+embed_CONFIGS[i].name+"-database.bin")
                embedData(self.model, query_path, query_embed_path,
                        embed_CONFIGS[i].size_query, batch_size=self.__conf.getHP('embed_batch'), original_dim=embed_CONFIGS[i].dim_seq,
                        embedded_dim=self.__conf.getHP('dim_embedding'), device=self.device, encoder=self.__conf.getHP('encoder'))
                print("database")
                embedData(self.model, database_path, database_embed_path,
                        DATASET_CONFIGS[i].size_db, batch_size=self.__conf.getHP('embed_batch'), original_dim=embed_CONFIGS[i].dim_seq,
                        embedded_dim=self.__conf.getHP('dim_embedding'), device=self.device, encoder=self.__conf.getHP('encoder'))


    def __model_change(self):
        print("这里暂停！！！")
        # print(dir(self.model))
        self.model._AEBuilder__encoder.forward = self.model._AEBuilder__encoder.new_forward.__get__(self.model._AEBuilder__encoder)
        print("更换成功！！！")


    def __train(self, func_a: float) -> None:#用来对图像进行编码和解码，以便在encode-decode过程中学习到embedding
        recons_errors = []
        orth_terms = []
        trans_errors = []
        return_l2s = []
        losses=[]

        if self.__conf.getHP('train_type') == 'interleaving':
            if not self.encoder_only:
                for batch in self.train_db_loader:

                    self.optimizer.zero_grad()
                    # print("batch")
                    # print(batch.shape)

                    embedding = self.model.encode(batch)
                    # reconstructed = self.model.decode(embedding)       ！！！！！！！！！！

                    # recons_term = self.recons_weight * self.recons_reg(batch, reconstructed)           ！！！！！！！！！！
                    orth_term = self.__orth_reg()
                    # regularization = recons_term + orth_term          ！！！！！！！！！！
                    regularization= orth_term

                    regularization.backward()
                    self.optimizer.step()

                    # recons_errors.append(recons_term.detach().item())       ！！！！！！！！！！
                    orth_terms.append(orth_term.detach().item())

                # self.logger.info('t{:d} recons = {:.4f}'.format(self.epoch, np.mean(recons_errors)))           ！！！！！！！！！！
                self.logger.info('t{:d} orth = {:.4f}'.format(self.epoch, np.mean(orth_terms)))

            for db_batchf, query_batchf1,query_batchf2 in zip(self.train_db_loader, self.train_query_loader1,self.train_query_loader2):
                self.optimizer.zero_grad()
                db_batch = db_batchf.transpose(0,2).transpose(0,1)
                query_batch1 = query_batchf1.transpose(0,2).transpose(0,1)
                query_batch2 = query_batchf2.transpose(0,2).transpose(0,1)
                if self.detch_query:
                    with torch.no_grad():
                        # print("query_batch1")
                        # print(query_batch1.shape)
                        query_embedding1 = self.model.encode(query_batch1).detach()
                        query_batch1 = query_batch1.detach()
                        query_embedding2 = self.model.encode(query_batch2).detach()
                        query_batch2 = query_batch2.detach()
                else:
                    query_embedding1 = self.model.encode(query_batch1)
                    query_embedding2 = self.model.encode(query_batch2)

                db_embedding = self.model.encode(db_batch)
                # print("db_batch")
                # print(db_batch.shape)
                # 给trans_erro加一个a的参数
                trans_error = self.trans_loss(func_a,db_batch, query_batch1, query_batch2,db_embedding, query_embedding1,query_embedding2)

                trans_error.backward()
                self.optimizer.step()

                trans_errors.append(trans_error.detach().item())

            logging.info('t{:d} trans = {:.4f}'.format(self.epoch, np.mean(trans_errors)))
            logging.info('t{:d} func_a = {:.4f}'.format(self.epoch, func_a))

        elif self.__conf.getHP('train_type') == 'linearlycombine':
            # tp=0
            for db_batchf, query_batchf1,query_batchf2 in zip(self.train_db_loader, self.train_query_loader1,self.train_query_loader2):

                # 打乱第一个维度（即维度大小为 bs 的部分）
                db_batchf = db_batchf[torch.randperm(db_batchf.size(0))]
                query_batchf1 = query_batchf1[torch.randperm(query_batchf1.size(0))]
                query_batchf2 = query_batchf2[torch.randperm(query_batchf2.size(0))]

                self.optimizer.zero_grad()
                db_batch = db_batchf
                query_batch1 = query_batchf1#.transpose(0,2).transpose(0,1)
                query_batch2 = query_batchf2#.transpose(0,2).transpose(0,1)
                if self.detch_query:#true
                    with torch.no_grad():
                        query_embedding1 = self.model.encode(query_batch1)[0].detach()
                        query_batch1 = query_batch1.detach()
                        query_embedding2 = self.model.encode(query_batch2)[0].detach()
                        query_batch2 = query_batch2.detach()
                else:
                    query_embedding1 = self.model.encode(query_batch1)
                    query_embedding2 = self.model.encode(query_batch2)

                db_embedding = self.model.encode(db_batch)
                db_orig = db_embedding[1]
                db_embedding = db_embedding[0]

                    #加个参数a

                trans_error = self.trans_loss(func_a,db_batch, query_batch1,query_batch2, db_embedding, query_embedding1,query_embedding2)
                if self.__conf.getHP("mode")=="pretrain":
                    # print("jisuanl")
                    return_l2=mean(self.__l2(squeeze(db_orig), squeeze(db_batch)))*(0.0001+abs(self.model._AEBuilder__encoder.fucb))
                else:
                    return_l2=0
                    # ,db_orig,self.model._AEBuilder__encoder.fucb,self.__conf.getHP('mode')
                # trans_error = self.trans_loss(func_a,db_batch, query_batch1, db_embedding, query_embedding1)
                print(trans_error)
                # recons_term = torch.zeros(1).to(self.device)
                # orth_term = torch.zeros(1).to(self.device)
                if not self.encoder_only:
                    # print("????-------------------------------------------------------------------------!!")
                    db_reconstructed = self.model.decode(db_embedding)
                    recons_term = self.recons_weight * self.recons_reg(db_batch, db_reconstructed)#计算重构误差
                else:
                    # print("!!!!-------------------------------------------------------------------------!!")
                    recons_term = torch.zeros(1).to(self.device)

                orth_term = self.__orth_reg()

                loss = trans_error  + orth_term + recons_term   #+return_l2
                # print("backward")
                # print("")
                loss.backward()
                # print("backward1")
                # print("")
                self.optimizer.step()
                # print("backward2")
                # print("")
                recons_errors.append(recons_term.detach().item())
                orth_terms.append(orth_term.detach().item())
                trans_errors.append(trans_error.detach().item())
                # return_l2s.append(return_l2.detach().item())
                losses.append(loss.detach().item())
                #func_as.append(func_a.detach().item())

            self.logger.info('t{:d} recons = {:.4f}'.format(self.epoch, np.mean(recons_errors)))#重构误差
            self.logger.info('t{:d} orth = {:.4f}'.format(self.epoch, np.mean(orth_terms)))#正交化项  正交化项（orthogonalization term）的平均值记录到日志中
            self.logger.info('t{:d} trans = {:.4f}'.format(self.epoch, np.mean(trans_errors)))#转换误差
            self.logger.info('t{:d} recon_encoder = {:.4f}'.format(self.epoch, np.mean(return_l2s)))#转换误差
            # self.logger.info('t{:d} fuca = {:.4f}'.format(self.epoch, self.model._AEBuilder__encoder.fuc.detach().item()))#转换误差
            self.logger.info('t{:d} fucb = {:.4f}'.format(self.epoch, self.model._AEBuilder__encoder.fucb.detach().item()))#转换误差
            # self.logger.info('t{:d} fucc = {:.4f}'.format(self.epoch, self.model._AEBuilder__encoder.fc.detach().item()))
            self.logger.info('t{:d} loss = {:.4f}'.format(self.epoch, np.mean(losses)))#转换误差
            # self.logger.info('t{:d} func_a = {:.4f}'.format(self.epoch, func_a))

        else:
            raise ValueError('cannot train')


    def __validate(self,func_a: float) -> None:
        trans_errors = []
        jlh_recons=[]
        jlh_regulars=[]
        jlh_losss=[]

        with torch.no_grad():
            for db_batch, query_batch1,query_batch2 in zip(self.val_db_loader, self.val_query_loader1,self.val_query_loader2):
                db_batch = db_batch[torch.randperm(db_batch.size(0))]
                query_batch1 = query_batch1[torch.randperm(query_batch1.size(0))]
                query_batch2 = query_batch2[torch.randperm(query_batch2.size(0))]
                db_embedding = self.model._AEBuilder__encoder.new_forward(db_batch)
                query_embedding1 = self.model._AEBuilder__encoder.new_forward(query_batch1)[0]
                query_embedding2 = self.model._AEBuilder__encoder.new_forward(query_batch2)[0]
                db_orig=db_embedding[1]
                db_embedding=db_embedding[0]
                #zijixied
                if not self.encoder_only:
                    # print("????-------------------------------------------------------------------------!!")
                    jlh_db_recon = self.model.decode(db_embedding)#zijixied
                    jlh_recons_term = self.recons_weight * self.recons_reg(db_batch, jlh_db_recon)#计算重构误差
                else:
                    # print("!!!!-------------------------------------------------------------------------!!")
                    jlh_recons_term = torch.zeros(1).to(self.device)
                # t = min(db_batch.shape[0], jlh_db_recon.shape[0])
                # jlh_recons_term = self.recons_weight * self.recons_reg(db_batch[:t, :], jlh_db_recon[:t, :])
                jlh_orth_term = self.__orth_reg()
                jlh_regularization = jlh_recons_term + jlh_orth_term
                if self.__conf.getHP("mode")=="pretrain":
                    return_l2=mean(self.__l2(squeeze(db_orig), squeeze(db_batch)))*(0.0001+abs(self.model._AEBuilder__encoder.fucb))
                    # print("encoder_recon",return_l2)
                else:
                    return_l2=0
                trans_error = self.trans_loss(func_a,db_batch, query_batch1,query_batch2, db_embedding, query_embedding1, query_embedding2)#转换误差
                # trans_error = self.trans_loss(func_a,db_batch, query_batch1, db_embedding, query_embedding1)
                loss=jlh_regularization+trans_error   #+return_l2 #jiade
                # print(trans_error)
                trans_errors.append(trans_error.detach().item())
                jlh_recons.append(jlh_recons_term.detach().item())
                jlh_regulars.append(jlh_regularization.detach().item())
                jlh_losss.append(loss.detach().item())
        print("本轮trans_error = ",np.mean(trans_errors))
        print("本轮全部的loss = ",np.mean(jlh_losss))
        self.logger.info('v{:d} trans = {:.4f}'.format(self.epoch, np.mean(trans_errors)))#搞出trans error的值选择a
        self.logger.info('v{:d} recons = {:.4f}'.format(self.epoch, np.mean(jlh_recons)))
        self.logger.info('v{:d} regular = {:.4f}'.format(self.epoch, np.mean(jlh_regulars)))
        self.logger.info('v{:d} loss = {:.4f}'.format(self.epoch, np.mean(jlh_losss)))


    def __checkpoint(self, persist_model: bool = True) -> None:
        if self.__conf.getHP('if_record'):
            if self.encoder_only:
                fig, ax = plt.subplots(2, 1, figsize=(12, 6))
            else:
                fig, ax = plt.subplots(3, 1, figsize=(12, 9))

            with torch.no_grad():
                for series in torch.squeeze(self.samples2plot).detach().cpu():
                    ax[0].plot(series)

                embedding = self.model.encode(self.samples2plot)
                for series in embedding.detach().cpu():
                    ax[1].plot(series)

                if not self.encoder_only:
                    reconstructed = self.model.decode(embedding)
                    for series in torch.squeeze(reconstructed).detach().cpu():
                        ax[2].plot(series)

            fig.tight_layout()
            plt.savefig(self.record_folder + str(self.epoch) + '.eps', dpi=456)

        if persist_model and self.checkpoint_mode != 'none' and (self.epoch == self.max_epoch or (self.checkpoint_mode == 'everyk' and self.epoch % self.checkpoint_k == 0)):
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder, str(self.epoch) + '.' + self.checkpoint_postfix))


    def __orth_reg(self) -> torch.Tensor:
        if self.orth_regularizer == 'srip':
            return self.srip_weight * getSRIPTerm(self.model, self.device)

        return torch.zeros(1).to(self.device)


    def __adjust_lr(self) -> None:
        # should be based on self.epoch and hyperparameters ONLY for easily resumming

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
            break

        new_lr = current_lr

        if self.__conf.getHP('lr_mode') == 'linear':
            lr_max = self.__conf.getHP('lr_max')
            lr_min = self.__conf.getHP('lr_min')

            new_lr = lr_max - self.epoch * (lr_max - lr_min) / self.max_epoch
        elif self.__conf.getHP('lr_mode') == 'exponentiallyhalve':
            lr_max = self.__conf.getHP('lr_max')
            lr_min = self.__conf.getHP('lr_min')

            for i in range(1, 11):
                if (self.max_epoch - self.epoch) * (2 ** i) == self.max_epoch:
                    new_lr = lr_max / (10 ** i)
                    break

            if new_lr < lr_min:
                new_lr = lr_min
        elif self.__conf.getHP('lr_mode') == 'exponentially':
            lr_max = self.__conf.getHP('lr_max')
            lr_min = self.__conf.getHP('lr_min')
            lr_k = self.__conf.getHP('lr_everyk')
            lr_ebase = self.__conf.getHP('lr_ebase')

            lr_e = int(np.floor(self.epoch / lr_k))
            new_lr = lr_max * (lr_ebase ** lr_e)

            if new_lr < lr_min:
                new_lr = lr_min
        elif self.__conf.getHP('lr_mode') == 'plateauhalve':
            raise ValueError('plateauhalve is not yet supported')

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


    def __adjust_wd(self):
        # should be based on self.epoch and hyperparameters ONLY for easily resumming

        for param_group in self.optimizer.param_groups:
            current_wd = param_group['weight_decay']
            break

        new_wd = current_wd

        if self.__conf.getHP('wd_mode') == 'linear':
            wd_max = self.__conf.getHP('wd_max')
            wd_min = self.__conf.getHP('wd_min')

            new_wd = wd_min + self.epoch * (wd_max - wd_min) / self.max_epoch

        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = new_wd


    def __adjust_srip(self):
        # should be based on self.epoch and hyperparameters ONLY for easily resumming

        if self.__conf.getHP('srip_mode') == 'linear':
            srip_max = self.__conf.getHP('srip_max')
            srip_min = self.__conf.getHP('srip_min')

            self.srip_weight = srip_max - self.epoch * (srip_max - srip_min) / self.max_epoch



    def __getOptimizer(self) -> optim.Optimizer:
        if self.__conf.getHP('optim_type') == 'sgd':
            if self.__conf.getHP('lr_mode') == 'fix':
                initial_lr = self.__conf.getHP('lr_cons')
            else:
                initial_lr = self.__conf.getHP('lr_max')

            if self.__conf.getHP('wd_mode') == 'fix':
                initial_wd = self.__conf.getHP('wd_cons')
            else:
                initial_wd = self.__conf.getHP('wd_min')

            momentum = self.__conf.getHP('momentum')

            return optim.SGD(self.model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=initial_wd)

        raise ValueError('cannot obtain optimizer')


    def __init_model(self, model: nn.Module, samples: torch.Tensor = None) -> nn.Module:
        if self.__conf.getHP('model_init') == 'lsuv':
            assert samples is not None

            return LSUVinit(model, samples[torch.randperm(samples.shape[0])][: self.__conf.getHP('lsuv_size')],
                            needed_mean=self.__conf.getHP('lsuv_mean'), needed_std=self.__conf.getHP('lsuv_std'),
                            std_tol=self.__conf.getHP('lsuv_std_tol'), max_attempts=self.__conf.getHP('lsuv_maxiter'),
                            do_orthonorm=self.__conf.getHP('lsuv_ortho'))

        return model
