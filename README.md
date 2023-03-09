# ResUNet-Segmentation
## Dataset
- Salivary gland SPECT image data set
- 由于数据集为医院所有，暂时无法公开，敬请谅解

## model
- 本项目手动搭建了UNet和ResUNet。其中UNet效果不错，而ResUNet效果并不理想，留待完善

## Description
- 无加号的是用原始的2070张图像的训练结果
- 一个加号的是在上一个的基础上剔除了30张带字母的图片的训练结果
- 两个加号的是在上一个的基础上进行了数据增广的训练结果
- _797的是在上一个的基础上加入了动态学习率调整

- models/net_SALIVA1.pth (average dice coef: 0.774888)
- models/net_SALIVA2.pth (average dice coef: 0.667549)
- models/net_SALIVA2+.pth (average dice coef: 0.681677)
- models/net_SALIVA2++.pth (average dice coef: 0.666346)
- models/net_797.pth (average dice coef: 0.797)
