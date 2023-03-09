import torch
from test import test
from data_load import get_dataset2
from loss_optim import dice_loss
from torch.utils.data import DataLoader
from UNet import DoubleConv, UNet

net = torch.load('models/net_797.pth', map_location=torch.device('cpu'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device,dtype=torch.float32)

train_dataset, test_dataset = get_dataset2('path_DataToProcess', train_prop=0.8)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
test(test_loader, net, dice_loss, device)