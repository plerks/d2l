import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

import numpy as np
import matplotlib.pyplot as plt

def read_idx_images(filename):
    """读取 IDX 格式的图片文件"""
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        n_images = int.from_bytes(f.read(4), 'big')
        n_rows = int.from_bytes(f.read(4), 'big')
        n_cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8) # 读取文件中剩余的所有字节，转成 NumPy 数组。此时 data 是一维数组
        return data.reshape(n_images, n_rows, n_cols) # 转成 3 维数组

def read_idx_labels(filename):
    """读取 IDX 格式的标签文件"""
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        n_labels = int.from_bytes(f.read(4), 'big')
        return np.frombuffer(f.read(), dtype=np.uint8) # 每个图片是什么

# # 读取测试集数据
# images = read_idx_images("../data/FashionMNIST/raw/t10k-images-idx3-ubyte")
# labels = read_idx_labels("../data/FashionMNIST/raw/t10k-labels-idx1-ubyte")
# print(f"图片数量: {len(images)}")
# print(f"图片尺寸: {images[0].shape}")
# print(f"标签数量: {len(labels)}")
# # 显示前5张图片
# fig, axes = plt.subplots(1, 5, figsize=(10, 2))
# for i in range(5):
#     axes[i].imshow(images[i], cmap='gray')
#     axes[i].set_title(f"Label: {labels[i]}")
#     axes[i].axis('off')
# plt.show()

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, nums_rows, nums_cols, titles=None, scale=1.5):
    figsize = (nums_cols * scale, nums_rows * scale)
    _, axes = d2l.plt.subplots(nums_rows, nums_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img): # tensor
            ax.imshow(img.numpy())
        else: # PIL 图像
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
plt.show()

batch_size = 256

def get_dataloader_workers():
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop(): .2f} sec')

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans) # 将多个预处理操作组合成一个流水线，数据会按顺序依次经过这些操作
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), 
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break