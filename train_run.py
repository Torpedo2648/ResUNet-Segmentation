import torch
from torch.utils.data import DataLoader

from loss_optim import get_optimiazer, dice_loss
from train import train
from data_load import get_dataset2
from UNet import UNet
from ResUNet import ResNetUNet
import segmentation_models_pytorch as smp

# net = smp.Unet(
#     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=1,                      # model output channels (number of classes in your dataset)
# )

# net = ResNetUNet(out_channels=1)

net = UNet(in_channels=1, out_channels=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device,dtype=torch.float32)

optimizer = get_optimiazer(net)
num_epochs = 48
batch_size = 32
train_dataset, test_dataset = get_dataset2('path_DataToProcess', train_prop=0.8)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

train(train_loader, net, dice_loss, num_epochs, device)
torch.save(net, 'net_SALIVA.pth')


"""
## 保存的模型说明
- 无加号的是用原始的2070张图像的训练结果
- 一个加号的是在上一个的基础上剔除了30张带字母的图片的训练结果
- 两个加号的是在上一个的基础上进行了数据增广的训练结果
- _797的是在上一个的基础上加入了动态学习率调整

- models/net_SALIVA1.pth (average dice coef: 0.774888)
- models/net_SALIVA2.pth (average dice coef: 0.667549)
- models/net_SALIVA2+.pth (average dice coef: 0.681677)
- models/net_SALIVA2++.pth (average dice coef: 0.666346)
- models/net_797.pth (average dice coef: 0.797)
"""
