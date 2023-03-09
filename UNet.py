import torch
import torch.nn as nn

# 定义一个双卷积块，包含两个3x3的卷积层和ReLU激活函数
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# 定义一个U-Net模型，包含编码器和解码器
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()  # 编码器中的卷积层列表
        self.ups = nn.ModuleList()  # 解码器中的卷积层列表

        # 定义编码器中的卷积层
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 定义解码器中的卷积层
        for feature in reversed(features[:-1]):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2) # 利用转置卷积进行上采样
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # 最后的卷积层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []  # 用于保存编码器中的特征图

        # 编码器部分
        for i, down in enumerate(self.downs):
            x = down(x)
            print(x.shape, "1111111")
            skip_connections.append(x)
            if i != len(self.downs)-1:
                x = nn.MaxPool2d(kernel_size=2)(x)

        # 解码器部分
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i//2+1]
            if x.shape != skip.shape:
                x = nn.functional.pad(x, (0, skip.shape[3]-x.shape[3], 0, skip.shape[2]-x.shape[2]))
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)


        # 最后的卷积层
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        return x