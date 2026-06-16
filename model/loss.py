# coding = utf-8
import torch
from numpy import sqrt
from torch import mean, squeeze
from torch.nn import Module, PairwiseDistance


def ED_2(tensor1, tensor2):
    tensor1 = squeeze(tensor1)
    tensor2 = squeeze(tensor2)

    # 确保两个张量形状相同
    if tensor1.shape != tensor2.shape:
        raise ValueError("输入的张量形状不相同")

    # 计算逐元素差
    difference = tensor1 - tensor2

    # 计算逐元素差的平方
    squared_difference = difference ** 2

    # 求每行第二维上所有元素的平方总和
    row_sum = squared_difference.sum(dim=1, keepdim=True)

    return squeeze(row_sum)


# TODO squeeze is not time-comusing. While it's still good to remove it
class ScaledL2Trans(Module):
    def __init__(self, original_dimension:int = 256, embedding_dimension: int = 16, to_scale: bool = False):
        super(ScaledL2Trans, self).__init__()

        self.__l2 = PairwiseDistance(p=2).cuda()
        self.__l1 = PairwiseDistance(p=1).cuda()

        if to_scale:
            self.__scale_factor_original = sqrt(original_dimension)
            self.__scale_factor_embedding = sqrt(embedding_dimension)
        else:
            self.__scale_factor_original = 1
            self.__scale_factor_embedding = 1

    def forward(self,alpha, database, query1,query2, db_embedding, query_embedding1,query_embedding2):


        original_l2 = (self.__l2(squeeze(database), squeeze(query1))-self.__l2(squeeze(database), squeeze(query2)))/self.__scale_factor_original
        embedding_l2 = (self.__l2(squeeze(db_embedding), squeeze(query_embedding1))-self.__l2(squeeze(db_embedding), squeeze(query_embedding2)))/self.__scale_factor_embedding*alpha

        return self.__l1(original_l2.view([1, -1]), embedding_l2.view([1, -1]))[0] / database.shape[0]




class ScaledL2Recons(Module):
    def __init__(self, original_dimension: int = 256, to_scale: bool = False):
        super(ScaledL2Recons, self).__init__()

        self.__l2 = PairwiseDistance(p=2).cuda()

        if to_scale:
            self.__scale_factor = sqrt(original_dimension)
        else:
            self.__scale_factor = 1

    def forward(self, database, reconstructed):

        # print("database.shape", database.shape)

        # print("reconstructed.shape", reconstructed.shape)
        return mean(self.__l2(squeeze(database), squeeze(reconstructed))) / self.__scale_factor
#self.__l2(squeeze(database), squeeze(reconstructed)) 返回了两个张量之间的 L2 距离，然后 mean 函数计算了这些距离的平均值。





