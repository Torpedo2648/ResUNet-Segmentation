import matplotlib.pyplot as plt
import numpy as np
from torch import optim


# 训练过程
def train(train_loader, net, loss_fn, num_epochs, device):
    for epoch in range(num_epochs):

        # 设置动态学习率
        if epoch < 5:
            lr = 0.006
        else:
            lr = 0.006*np.exp(0.1*(5-epoch))
        optimizer = optim.Adam(net.parameters(), lr)

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 正向传播
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 打印损失值
            if (i + 1) % 1 == 0:
                correct = 0
                output, label = outputs[-1], labels[-1]
                output = output.permute(1,2,0).detach().cpu().numpy().squeeze()
                output = np.where(output > 0.5, 1, 0)
                label = label.permute(1,2,0).detach().cpu().numpy().squeeze()

                correct = (output == label).sum()
                print('Epoch [{}/{}], Step [{}/{}], Dice coefficient: {:.6f}, Dice Loss: {:.6f}, Correct pixels: {}/{}'
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), 1-loss.item(), loss.item(), correct, output.size))

                imgs = [output, label]
                titles = ['y_pred', 'y_true']
                for i, img in enumerate(imgs):
                    plt.subplot(1, len(imgs), i+1)
                    plt.imshow(img, cmap='gray')
                    plt.title(titles[i])
                    plt.axis('off')
                plt.show()
                # plt.savefig(f'comparison/fig_{epoch+1}.png')