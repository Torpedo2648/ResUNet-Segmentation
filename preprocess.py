import os
import torch
from PIL import Image
from torchvision import transforms

from ima_crop import get_data


def prepro(data_src, data_dest = 'path_DataProcessed'):
    
    get_data(data_src, data_dest)

    # 图像变换函数
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),  # 水平翻转
        # transforms.RandomVerticalFlip(),  # 垂直翻转
        # transforms.RandomRotation(degrees=15),  # 随机旋转
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 色彩抖动
        # transforms.RandomCrop(size=(133, 135)),  # 随机裁剪
        # transforms.RandomResizedCrop(size=(133, 135), scale=(0.9, 1.2), ratio=(0.85, 1.2)), # 随机缩放
        transforms.ToTensor()  # 转为张量
    ])

    length = len(os.listdir(data_dest + '/inputs'))
    input_list = [data_dest + f"/inputs/img_{i+1}.jpg" for i in range(length)  if i<1200 or i>1229]
    label_list = [data_dest + f"/labels/img_{i+1}.jpg" for i in range(length)  if i<1200 or i>1229]

    input_batch = []
    label_batch = []

    for i, (input, label) in enumerate(zip(input_list, label_list)):
        img = Image.open(input)
        input_tensor = transform(img)
        input_batch.append(input_tensor)
        img = Image.open(label)
        label_tensor = transform(img)
        label_batch.append(label_tensor)

    # 将多张图片拼接成一个批次张量，维度为(N, C, H, W)，其中N为批次大小
    inputs = torch.stack(input_batch, dim=0)
    labels = torch.stack(label_batch, dim=0)
    # print(input.shape, label.shape)
    return inputs, labels