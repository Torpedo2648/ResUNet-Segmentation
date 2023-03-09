import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNetUNet(nn.Module):
    def __init__(self, out_channels):
        super(ResNetUNet, self).__init__()
        
        # 加载预训练的ResNet50网络
        resnet = models.resnet50(pretrained=True)
        
        # 编码器部分
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
        # 解码器部分
        self.decoder5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        )
        
        # 最终分类器
        self.classifier = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=1), nn.Sigmoid())

        
    def forward(self, x):
        # 编码器部分
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        # 解码器部分
        dec5 = self.decoder5(enc5)
        print(enc4.shape, dec5.shape)
        dec5 = nn.functional.pad(dec5, (0, enc4.shape[3]-dec5.shape[3], 0, enc4.shape[2]-dec5.shape[2]))
        dec4 = self.decoder4(torch.cat([enc4, dec5], dim=1))
        print(enc3.shape, dec4.shape)

        dec4 = nn.functional.pad(dec4, (0, enc3.shape[3]-dec4.shape[3], 0, enc3.shape[2]-dec4.shape[2]))
        dec3 = self.decoder3(torch.cat([enc3, dec4], dim=1))
        print(enc2.shape, dec3.shape)

        dec3 = nn.functional.pad(dec3, (0, enc2.shape[3]-dec3.shape[3], 0, enc2.shape[2]-dec3.shape[2]))
        dec2 = self.decoder2(torch.cat([enc2, dec3], dim=1))
        print(enc1.shape, dec2.shape)

        dec2 = nn.functional.pad(dec2, (0, enc1.shape[3]-dec2.shape[3], 0, enc1.shape[2]-dec2.shape[2]))
        dec1 = self.decoder1(torch.cat([enc1, dec2], dim=1))
        print(dec1.shape)
        
        # 最终分类器
        out = self.classifier(dec1)
        out = nn.functional.pad(out, (0, 135-out.shape[3], 0, 133-out.shape[2]))
        print(out.shape)
        return out