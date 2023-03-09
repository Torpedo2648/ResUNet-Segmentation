import numpy as np
import torch
import matplotlib.pyplot as plt

# 测试过程
def test(test_loader, net, loss_fn, device):
    net.eval()
    with torch.no_grad():
        Loss = 0
        pic_order = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            print("Batch [{}/{}], Dice coefficient: {:.6f}, Dice Loss: {:.6f}"
            .format(pic_order+1, len(test_loader), 1-loss, loss))
            pic_order += 1
            Loss += loss
            
            for output, label in zip(outputs, labels):
                output = output.permute(1,2,0).detach().cpu().numpy().squeeze()
                output = np.where(output > 0.5, 1, 0)
                label = label.permute(1,2,0).detach().cpu().numpy().squeeze()

                imgs = [output, label]
                titles = ['y_pred', 'y_true']
                for i, img in enumerate(imgs):
                        plt.subplot(1, len(imgs), i+1)
                        plt.imshow(img, cmap='gray')
                        plt.title(titles[i])
                        plt.axis('off')
                # plt.show()

        print('Average Dice coeffitient: {:.6} Average Loss: {:.6}'
        .format(1-Loss/len(test_loader), Loss/len(test_loader)))