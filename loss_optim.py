import numpy as np
from torch import optim

# 定义损失函数
def dice_coefficient(y_pred, y_true):

    print(y_pred.shape, y_true.shape)
    eps = 1e-6  # 加一个极小值，避免分母为0
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    dice = (2. * intersection + eps) / (union + eps)
    return dice

def dice_loss(y_pred, y_true):
    return 1 - dice_coefficient(y_pred, y_true)

# 定义优化器
def get_optimiazer(net):
    return optim.Adam(net.parameters(), lr=0.001)