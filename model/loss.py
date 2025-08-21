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

    def forward(self,jlh_a, database, query1,query2, db_embedding, query_embedding1,query_embedding2):
        
        original_l2 = (self.__l2(squeeze(database), squeeze(query1))-self.__l2(squeeze(database), squeeze(query2)))/self.__scale_factor_original
        embedding_l2 = (self.__l2(squeeze(db_embedding), squeeze(query_embedding1))-self.__l2(squeeze(db_embedding), squeeze(query_embedding2)))/self.__scale_factor_embedding*jlh_a

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


        return mean(self.__l2(squeeze(database), squeeze(reconstructed))) / self.__scale_factor*0.00001














'''
ScaledL2Trans 模块用于计算数据库和查询之间的 L1 距离，其中数据库和查询可以是原始数据或嵌入式表示。
如果需要，在计算距离之前，可以选择按比例缩放原始维度和嵌入维度。这有助于确保不同维度之间的距离具有可比性。

ScaledL2Recons 模块用于计算数据库和重构数据之间的 L2 距离。同样，可以选择按比例缩放原始维度，以确保距离具有可比性。

在 ScaledL2Trans 的构造函数中，使用了 PairwiseDistance 类来计算 L1 和 L2 距离。
如果设置了 to_scale 标志，则会计算原始维度和嵌入维度的缩放因子，并在前向传播时应用这些缩放因子。

在 ScaledL2Recons 的构造函数中，同样使用了 PairwiseDistance 类来计算 L2 距离。根据需要，也会计算原始维度的缩放因子。

在 forward 方法中，首先使用 squeeze 函数来压缩数据库和查询的张量维度。然后，计算原始数据和嵌入数据''' 