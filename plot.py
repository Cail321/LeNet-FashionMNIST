import torch
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as Data


train_data = FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)
#获得一个batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()
class_label = train_data.classes
print(class_label)

#可视化一个batch的图像
plt.figure(figsize = (12,5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4,16,ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]],size = 10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()