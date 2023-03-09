import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split

from preprocess import prepro


class MyDataset(Dataset):
    def __init__(self, input_data, label_data):
        self.input_data = input_data
        self.label_data = label_data

    def __getitem__(self, index):
        input_image = self.input_data[index]
        label_image = self.label_data[index]
        return input_image, label_image

    def __len__(self):
        return len(self.input_data)


# 将所有患者的数据混合。随机取样80%作为训练集，20%为测试集
def get_dataset1(data_src, train_prop):

    inputs, labels = prepro(data_src)
    dataset = MyDataset(inputs, labels)

    # 将数据集按照指定比例分割成训练集和测试集，size为分割后每个子集的长度
    train_size = int(len(dataset) * train_prop)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    return train_dataset, test_dataset
    

# 按照患者分类来训练。取前80%作为训练集，最后20%为测试集
def get_dataset2(data_src, train_prop, img_per_patient = 30):

    inputs, labels = prepro(data_src)
    train_count = int(inputs.shape[0] / img_per_patient * train_prop)

    train_input = inputs[:train_count * img_per_patient, :, :]
    train_label = labels[:train_count * img_per_patient, :, :]

    test_input = inputs[train_count * img_per_patient: , :, :]
    test_label = labels[train_count * img_per_patient: , :, :]

    train_dataset = MyDataset(train_input, train_label)
    test_dataset = MyDataset(test_input, test_label)

    return train_dataset, test_dataset