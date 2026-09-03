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
from util.batch_schedule import batches_per_epoch, candidate_batch_indices, select_batch_indices
import random
import math
from torch.nn import Module, PairwiseDistance

class DatasetConfig:
    def __init__(self, name, path_db, dim_seq, size_train, size_val, size_db,index_name):
        self.name = name
        self.path_db = path_db
        self.dim_seq = dim_seq
        self.size_train = size_train
        self.size_val = size_val
        self.size_db = size_db
        self.index_name=index_name


class EmbedConfig:
    def __init__(self, name, dataset_path,query_path, dim_seq, size_query):
        self.name = name
        self.dataset_path = dataset_path
        self.query_path=query_path
        self.dim_seq = dim_seq
        self.size_query = size_query

DATASET_CONFIGS = [
    DatasetConfig("Astro00", "/data/user_jialinhan/data_big/astro-dataset.bin", 256, 400000, 10000, 100000000,0),
    DatasetConfig("Deep1B", "/data/user_jialinhan/data_big/deep1b-dataset.bin", 96, 400000, 10000, 100000000,1),
    DatasetConfig("F5", "/data/user_jialinhan/data_big/F5-dataset.bin", 256, 400000, 10000, 100000000,2),
    # DatasetConfig("F10", "/data/user_jialinhan/data_big/F10-dataset.bin", 256, 400000, 10000, 100000000,3),
    DatasetConfig("origin", "/data/user_jialinhan/data_big/origin-dataset.bin", 256, 400000, 10000, 100000000,3),
    DatasetConfig("sald", "/data/user_jialinhan/data_big/sald-dataset.bin", 128, 400000, 10000, 100000000,4),
    DatasetConfig("seismic", "/data/user_jialinhan/data_big/seismic-dataset.bin", 256, 400000, 10000, 100000000,5)
]
embed_CONFIGS = [    #   database path                                               query path
    EmbedConfig("astro", "data_big/astro-dataset.bin",    "data_big/astro-query.bin",256,100),
    EmbedConfig("deep1b", "data_big/deep1b-dataset.bin",    "data_big/deep1b-query.bin",96,1000),
    EmbedConfig("F5", "data_big/F5-dataset.bin",    "data_big/F5-query.bin",256,1000),
    # EmbedConfig("F10", "data_big/F10-dataset.bin",    "data_big/F10-query.bin",256,1000),
    EmbedConfig("origin", "data_big/origin-dataset.bin",    "data_big/origin-query.bin",256,1000),
    EmbedConfig("sald", "data_big/sald-dataset.bin",    "data_big/sald-query.bin",128,1000),
    EmbedConfig("seismic", "data_big/seismic-dataset.bin",    "data_big/seismic-query.bin",256,1000),
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
        self.train_total_loader = []
        self.train_db_loader = []
        self.train_query_loader1 = []
        self.train_query_loader2 = []

        self.train_dbl=[]
        self.val_dbl=[]

        self.val_db_loader = []
        self.val_query_loader1 = []
        self.val_query_loader2 = []
        self.index_list=[]
        len1=0
        lent=0
        for config in DATASET_CONFIGS:
            size_train = int(config.size_train / num_data_base*10)
            size_val = int(config.size_val / num_data_base)

            train_samples, val_samples = getSamples(self.__conf, config.path_db,
                                                    f"conf/samples/{config.name}_train_indices.bin",
                                                    f"conf/samples/{config.name}_val_indices.bin",
                                                    config.dim_seq, size_train, size_val, config.size_db,
                                                    f"conf/samples/{config.name}_train_samples.bin",
                                                    f"conf/samples/{config.name}_val_samples.bin")
            self.train_total_loader.extend(DataLoader(TSDataset(train_samples), batch_size=batch_size, shuffle=False))
            len1=len(self.train_total_loader)-lent
            lent+=len1
            for _ in range(len1):
                self.index_list.append(config.index_name)
            # self.train_db_loader.extend(DataLoader(TSDataset(train_samples), batch_size=batch_size, shuffle=False))
            # self.train_query_loader1.extend(DataLoader(TSDataset(train_samples), batch_size=batch_size, shuffle=False))
            # self.train_query_loader2.extend(DataLoader(TSDataset(train_samples), batch_size=batch_size, shuffle=False))
            # import pdb; pdb.set_trace()
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
                self.model = self.__init_model(self.model, val_samples)

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

    def shuffle_batch_inter(self,batch_list):
        """
        打乱每个batch内部的数据顺序，但保持batch之间的顺序不变

        参数:
            batch_list: 包含多个batch的列表，每个batch是一个张量或数组
        返回:
            处理后的batch列表，每个batch内部已打乱
        """
        shuffled_batches = []
        for batch in batch_list:
            # 生成随机索引来打乱当前batch
            if isinstance(batch, torch.Tensor):
                # 处理PyTorch张量
                shuffle_indices = torch.randperm(batch.size(0))
                shuffled_batch = batch[shuffle_indices]
            else:
                # 处理其他类型（如numpy数组）
                shuffle_indices = random.sample(range(len(batch)), len(batch))
                shuffled_batch = batch[shuffle_indices]

            shuffled_batches.append(shuffled_batch)

        return shuffled_batches

    def __calculate_curriculum_loss(self, batch: torch.Tensor, alpha: float) -> float:
        """Calculate one batch's difficulty without updating model parameters."""
        db_batch = batch[torch.randperm(batch.size(0))]
        query_batch1 = batch[torch.randperm(batch.size(0))]
        query_batch2 = batch[torch.randperm(batch.size(0))]

        query_embedding1 = self.model.encode(query_batch1)[0]
        query_embedding2 = self.model.encode(query_batch2)[0]

        db_output = self.model.encode(db_batch)
        db_embedding = db_output[0]
        db_orig = db_output[1]

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

    def run(self) -> None:
        if not self.has_setup:
            self.setup()

        self.__checkpoint(persist_model=False)

        mode=self.__conf.getHP('mode')

        if mode=="pretrain":
            print("pretrain")
            if self.__conf.getHP('train_type') != 'linearlycombine':
                raise ValueError("pretraining requires train_type='linearlycombine'")
            if not self.train_total_loader:
                raise ValueError('cannot train with an empty training set')

            training_schedule = self.__conf.getHP('training_schedule')
            if training_schedule not in ('loss_curriculum', 'random_baseline'):
                raise ValueError(f'unknown training schedule: {training_schedule}')
            total_num_batches = len(self.train_total_loader)
            batch_size_per_round = batches_per_epoch(total_num_batches)
            schedule_message = (
                f"Training schedule={training_schedule}: "
                f"batches={total_num_batches}, "
                f"batches_per_epoch={batch_size_per_round}"
            )
            print(schedule_message)
            self.logger.info(schedule_message)

            if training_schedule == 'loss_curriculum':
                alpha = self.__conf.getHP('alpha')
                # The initial pass scores difficulty only; it is not a training epoch.
                batch_metrics = []
                was_training = self.model.training
                self.model.eval()
                try:
                    with torch.no_grad():
                        for idx, batch in enumerate(self.train_total_loader):
                            loss_value = self.__calculate_curriculum_loss(batch, alpha)
                            if not math.isfinite(loss_value):
                                raise ValueError(
                                    f'non-finite curriculum loss at batch {idx}: {loss_value}'
                                )
                            batch_metrics.append((loss_value, idx))
                finally:
                    self.model.train(was_training)

                batch_metrics.sort(key=lambda item: item[0])
                total_indices = [idx for _, idx in batch_metrics]
                print(
                    "Loss difficulty evaluation completed: "
                    f"batches={total_num_batches}, "
                    f"min_loss={batch_metrics[0][0]:.6f}, "
                    f"max_loss={batch_metrics[-1][0]:.6f}, "
                    f"batches_per_epoch={batch_size_per_round}"
                )
            else:
                total_indices = list(range(total_num_batches))
                baseline_message = (
                    "Random baseline: initial loss evaluation skipped; "
                    "all batches are eligible in every epoch."
                )
                print(baseline_message)
                self.logger.info(baseline_message)
            print(f"每轮选取的批次数量: {batch_size_per_round}")

            while self.epoch < self.max_epoch:
                batches = [batch for batch in self.train_total_loader]


                curriculum_pool = candidate_batch_indices(
                    training_schedule, total_indices, self.epoch
                )
                if training_schedule == 'loss_curriculum' and self.epoch < 80:
                    stage_index = min(self.epoch // 10, 7)

                    if self.epoch % 10 == 0:
                        print(
                            f"Curriculum stage {stage_index + 1}/8: "
                            f"candidate_batches={len(curriculum_pool)}/{total_num_batches}"
                        )
                random_positions = []
                if len(curriculum_pool) > batch_size_per_round:
                    random_positions = torch.randperm(
                        len(curriculum_pool)
                    )[:batch_size_per_round].tolist()
                selected_indices = select_batch_indices(
                    curriculum_pool, batch_size_per_round, random_positions
                )

                self.train_db_loader = [batches[i] for i in selected_indices]
                self.train_query_loader1 = [batches[i] for i in selected_indices]
                self.train_query_loader2 = [batches[i] for i in selected_indices]

                self.train_list=[]
                self.train_query_list=[]
                for i in range(len(DATASET_CONFIGS)):
                    self.train_list.append([])
                    self.train_query_list.append([])
                for i in range(len(selected_indices)):
                    dataset_index = self.index_list[selected_indices[i]]
                    self.train_list[dataset_index].append(i)
                    self.train_query_list[dataset_index].append(i)

                # 分配 train_query_loader1，确保相同数据集内打乱
                for i in range(len(self.train_list)):
                    indices = self.train_list[i][:]
                    random.shuffle(indices)
                    for j in range(len(indices)):
                        self.train_query_loader1[indices[j]] = self.train_db_loader[indices[j]]

                # 分配 train_query_loader2，确保相同数据集内打乱
                for i in range(len(self.train_list)):
                    indices = self.train_list[i][:]
                    random.shuffle(indices)
                    for j in range(len(indices)):
                        self.train_query_loader2[indices[j]] = self.train_db_loader[indices[j]]

                self.train_db_loader = self.shuffle_batch_inter(self.train_db_loader)
                self.train_query_loader1 = self.shuffle_batch_inter(self.train_query_loader1)
                self.train_query_loader2 = self.shuffle_batch_inter(self.train_query_loader2)


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


                self.epoch += 1
                print("第",self.epoch,"周期ing")
                # alpha = self.model._AEBuilder__encoder.fuc
                alpha=self.__conf.getHP('alpha')


                self.__train(alpha)
                self.__validate(alpha)

                self.logger.info('e{:d} time = {:.3f}s'.format(self.epoch, timer() - start))

                self.__checkpoint()
            self.__model_change()

            import pickle
            pickle_path = os.path.abspath(self.__conf.getHP('pickle'))
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
            with open(pickle_path, 'wb') as f:
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

        if self.__conf.getHP('to_embed')=="nono":
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

    def __model_mask(self):
        print("这里暂停！！！")
        # print(dir(self.model))
        self.model._AEBuilder__encoder.forward = self.model._AEBuilder__encoder.old_forward.__get__(self.model._AEBuilder__encoder)
        print("更换成功！！！")


    def __train(self, alpha: float) -> None:#用来对图像进行编码和解码，以便在encode-decode过程中学习到embedding
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
                trans_error = self.trans_loss(alpha,db_batch, query_batch1, query_batch2,db_embedding, query_embedding1,query_embedding2)

                trans_error.backward()
                self.optimizer.step()

                trans_errors.append(trans_error.detach().item())

            logging.info('t{:d} trans = {:.4f}'.format(self.epoch, np.mean(trans_errors)))
            logging.info('t{:d} alpha = {:.4f}'.format(self.epoch, alpha))

        elif self.__conf.getHP('train_type') == 'linearlycombine':
            # tp=0
            for db_batchf, query_batchf1,query_batchf2 in zip(self.train_db_loader, self.train_query_loader1,self.train_query_loader2):

                # 打乱第一个维度（即维度大小为 bs 的部分）
                db_batchf = db_batchf[torch.randperm(db_batchf.size(0))]
                query_batchf1 = query_batchf1[torch.randperm(query_batchf1.size(0))]
                query_batchf2 = query_batchf2[torch.randperm(query_batchf2.size(0))]

                self.optimizer.zero_grad()
                # tp=tp+1
                # print(tp)
                # print("db_batchf.shape=", db_batchf.shape)
                db_batch = db_batchf

                # db_batch = db_batch.squeeze(1)
                # print(db_batch.shape)
                # df = pd.DataFrame(db_batch.cpu().numpy())
                # df.to_excel('tensor_data.xlsx', index=False)
                # exit()
                # print("test")
                # print("")


                #.transpose(0,2).transpose(0,1)
                # print("db_batch.shape=", db_batch.shape)
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

                trans_error = self.trans_loss(alpha,db_batch, query_batch1,query_batch2, db_embedding, query_embedding1,query_embedding2)
                if self.__conf.getHP("mode")=="pretrain":
                    # print("jisuanl")

                    return_l2=mean(self.__l2(squeeze(db_orig), squeeze(db_batch)))*self.__conf.getHP("func_b")
                else:
                    return_l2=0
                    # ,db_orig,self.model._AEBuilder__encoder.fucb,self.__conf.getHP('mode')
                # trans_error = self.trans_loss(alpha,db_batch, query_batch1, db_embedding, query_embedding1)
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

                loss = trans_error  + orth_term + recons_term+return_l2
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
                return_l2s.append(return_l2.detach().item())
                losses.append(loss.detach().item())
                #alphas.append(alpha.detach().item())

            self.logger.info('t{:d} recons = {:.4f}'.format(self.epoch, np.mean(recons_errors)))#重构误差
            self.logger.info('t{:d} orth = {:.4f}'.format(self.epoch, np.mean(orth_terms)))#正交化项  正交化项（orthogonalization term）的平均值记录到日志中
            self.logger.info('t{:d} trans = {:.4f}'.format(self.epoch, np.mean(trans_errors)))#转换误差
            self.logger.info('t{:d} recon_encoder = {:.4f}'.format(self.epoch, np.mean(return_l2s)))#转换误差
            self.logger.info('t{:d} alpha = {:.4f}'.format(self.epoch, alpha))#转换误差
            self.logger.info('t{:d} fucb = {:.4f}'.format(self.epoch, self.model._AEBuilder__encoder.fucb.detach().item()))#转换误差
            self.logger.info('t{:d} loss = {:.4f}'.format(self.epoch, np.mean(losses)))#转换误差
            # self.logger.info('t{:d} alpha = {:.4f}'.format(self.epoch, alpha))

        else:
            raise ValueError('cannot train')


    def __validate(self,alpha: float) -> None:
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
                    return_l2=mean(self.__l2(squeeze(db_orig), squeeze(db_batch))) * self.__conf.getHP("func_b")
                    # print("encoder_recon",return_l2)
                else:
                    return_l2=0
                trans_error = self.trans_loss(alpha,db_batch, query_batch1,query_batch2, db_embedding, query_embedding1, query_embedding2)#转换误差
                # trans_error = self.trans_loss(alpha,db_batch, query_batch1, db_embedding, query_embedding1)
                loss=jlh_regularization+trans_error+return_l2 #jiade
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
