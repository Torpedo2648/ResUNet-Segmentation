import torch
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from sklearn.metrics import accuracy_score


# 定义Leave-one-out交叉验证函数
def loo_cross_valid(model, dataset, batch_size, num_epochs):
    predictions = []
    targets = []
    for i in range(len(dataset)):
        # 划分训练集和测试集
        train_set = Subset(dataset, [j for j in range(len(dataset)) if j != i])
        test_set = Subset(dataset, [i])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1)
        
        # 训练模型
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            for x_train, y_train in train_loader:
                optimizer.zero_grad()
                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()
        
        # 测试模型
        with torch.no_grad():
            for x_test, y_test in test_loader:
                y_pred = model(x_test)
                predictions.append(torch.argmax(y_pred).item())
                targets.append(y_test.item())
                
    # 计算模型性能指标
    accuracy = accuracy_score(targets, predictions)
    return accuracy


def kf_cross_valid(model, data, k, criterion, optimizer, batch_size, num_epochs):
    """
    K-fold交叉验证函数
    
    Args:
        model: 模型
        data: 数据集
        k: 折数
        criterion: 损失函数
        optimizer: 优化器
        batch_size: 每批次数据的数量
        num_epochs: 训练的轮数
    """
    # 将数据集拆分为k份
    data_splits = random_split(data, [len(data)//k]*k)
    
    for i in range(k):
        # 确定当前fold的训练集和测试集
        verif_data = data_splits[i]
        train_data = ConcatDataset(data_splits[:i] + data_splits[i+1:])
        
        # 构建DataLoader
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        verif_loader = DataLoader(verif_data, batch_size=batch_size, shuffle=True)
        
        # 训练模型
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                
            # 计算验证集上的损失
            model.eval()
            verif_loss = 0.0
            with torch.no_grad():
                for inputs, targets in verif_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    verif_loss += loss.item() * inputs.size(0)
                    
            # 打印每个fold上的训练损失和验证损失
            print('Fold [{}/{}] - Train Loss: {:.4f} - Val Loss: {:.4f}'.format(
                i+1, k, train_loss/len(train_data), verif_loss/len(verif_data)))
