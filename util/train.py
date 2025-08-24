#coding=utf-8
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from ray import tune
from apex import amp
from model.loss import L2ScaledTransformation, L2Rreconstruction
from util.data import TSDataset


class AlternativeTrainingAutoencoder(tune.Trainable):
    # def setup(self, config):
    def _setup(self, config):
        torch.manual_seed(97)#首先对PyTorch进行了随机种子设置，以确保模型收敛性
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(29)
        
        self.epoch = 0
        self.ALTERNATIVE_EPOCHES = int(config["epoch"] / 5 * 4)
        self.ADJUSTION_EPOCHES = int(config["epoch"] / 4)
        #根据配置文件中的epoch参数，计算了训练过程中的主要迭代次数ALTERNATIVE_EPOCHES和调整迭代次数ADJUSTION_EPOCHES。



        #该函数创建了数据加载器（DataLoader）对象，用于在训练和验证过程中加载数据。
        #训练集和验证集的数据分别被分配给train_db_loader、train_query_loader、val_db_loader和val_query_loader
        self.train_dataset = TSDataset(config["train_samples"].cuda())
        self.train_db_loader = DataLoader(self.train_dataset, batch_size=config.get('batch_size', 64), shuffle=True)
        self.train_query_loader = DataLoader(self.train_dataset, batch_size=config.get('batch_size', 64), shuffle=True)

        self.val_dataset = TSDataset(config["val_samples"].cuda())
        self.val_db_loader = DataLoader(self.val_dataset, batch_size=config.get('batch_size', 64), shuffle=True)
        self.val_query_loader = DataLoader(self.val_dataset, batch_size=config.get('batch_size', 64), shuffle=True)
        #定义了损失函数，包括变换损失（transformation loss）和重构损失（reconstruction loss）
        self.transformation_loss = L2ScaledTransformation().cuda()
        self.reconstruction_loss = L2Rreconstruction().cuda()

        #根据配置文件中的模型参数，创建并初始化了一个PyTorch模型对象
        if 'negative_slope' in config:
            self.model = config["model"](dim_embedding=config.get('dim_embedding', 16),
                                         dim_sequence=config.get('dim_sequence', 256),
                                         negative_slope=config.get('negative_slope', 1e-2)).cuda()
        else:
            self.model = config["model"](dim_embedding=config.get('dim_embedding', 16),
                                         dim_sequence=config.get('dim_sequence', 256)).cuda()

        #定义了优化器（optimizer）的参数，并创建了一个优化器对象
        self.optimizer = optim.SGD(self.model.parameters(), 
                                   lr=config.get('lr', 1e-3), 
                                   momentum=config.get('momentum', 0.9), 
                                   weight_decay=config.get('weight_decay', 1e-5))
                                   
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=config.get('opt_level', 'O0'))

         
    # def step(self):
    def _train(self):   
        self.__adjust_learning_rate()#调整学习率：使用self.__adjust_learning_rate()来调整学习率，以便在训练过程中动态地调整模型收敛速度。
        self.epoch += 1#增加迭代次数

        train_reconstruction_batch = []#训练重构损失
        if self.epoch < self.ALTERNATIVE_EPOCHES:
            for batch in self.train_db_loader:
                self.optimizer.zero_grad()

                reconstruction_error = self.reconstruction_loss(batch, self.model(batch))
                
                # reconstruction_error.backward()
                with amp.scale_loss(reconstruction_error, self.optimizer) as scaled_loss:
                    scaled_loss.backward()

                self.optimizer.step()

                train_reconstruction_batch.append(reconstruction_error.detach().item())
        else:
            with torch.no_grad():
                for batch in self.train_db_loader:
                    train_reconstruction_batch.append(self.reconstruction_loss(batch, self.model(batch)).detach().item())

        train_diffences_batch = []#训练转换损失
        for db_batch, query_batch in zip(self.train_db_loader, self.train_query_loader):
            self.optimizer.zero_grad()
            
            transformation_error = self.transformation_loss(db_batch, query_batch, self.model.encode(db_batch), self.model.encode(query_batch))

            # transformation_error.backward()
            with amp.scale_loss(transformation_error, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            self.optimizer.step()
            
            train_diffences_batch.append(transformation_error.detach().item())
        
        val_diffences_batch = []#验证损失，在训练过程中，使用val_diffences_batch列表来存储每个批次的验证损失。这个过程在训练过程中不会影响模型性能。
        with torch.no_grad():
            for db_batch, query_batch in zip(self.val_db_loader, self.val_query_loader):
                val_diffences_batch.append(self.transformation_loss(db_batch, query_batch, self.model.encode(db_batch), self.model.encode(query_batch)).detach().item())                
        
        
        #返回中间结果：函数返回一个字典，包含验证损失、训练损失和重构损失的平均值，以便在训练过程中监控模型性能。
        return {'val_diff': np.mean(val_diffences_batch), 'train_diff': np.mean(train_diffences_batch), 'rec_error': np.mean(train_reconstruction_batch)}


    def __adjust_learning_rate(self):#当达到指定的训练次数（self.ADJUSTION_EPOCHES）时，将优化器（self.optimizer）中每个参数组的learning rate减小为原来的0.1倍。
        if self.epoch % self.ADJUSTION_EPOCHES == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1


    # def save_checkpoint(self, checkpoint_dir):
    def _save(self, checkpoint_dir):#用于保存当前模型的状态（模型参数、优化器状态和混合精度状态），将其保存到一个文件中。
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'amp': amp.state_dict()
        }
        
        # checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
        checkpoint_path = os.path.join(checkpoint_dir, 'amp_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path


    # def load_checkpoint(self, checkpoint_path):
    def _restore(self, checkpoint_path):#用于从保存的文件中恢复模型的状态。

        # self.model.load_state_dict(torch.load(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])

        
    # def cleanup(self):
    def _stop(self):
        self.train_dataset = None
        self.train_db_loader = None
        self.train_query_loader = None
        
        self.val_dataset = None
        self.val_db_loader = None
        self.val_query_loader = None
        
        self.model = None
        self.transformation_loss = None
        self.reconstruction_loss = None
        self.optimizer = None
        
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print('Welcome to where the training methods got defined!')
