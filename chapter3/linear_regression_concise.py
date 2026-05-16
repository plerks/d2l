import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000) # 样本输入输出

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays) # *data_arrays 解构 data_arrays 这个元组
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
# print(next(iter(data_iter)))

net = nn.Sequential(nn.Linear(2, 1)) # 2 和 1 是输入和输出的数量

# 初始化 w 和 b
net[0].weight.data.normal_(0, 1)
net[0].bias.data.fill_(0)

loss = nn.MSELoss() # loss 是 MSELoss 的实例，实现了 __call__ 方法

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y) # 将 X 传入 net 得到预测值，再计算与真实值 y 的 loss
        # 三步曲
        trainer.zero_grad() # 清零梯度
        l.backward() # 反向传播
        trainer.step() # 更新参数
    t = net(features)
    # print(t.shape)
    # print(labels.shape)
    l = loss(net(features), labels) # 训练一轮后的 loss
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('true_w.shape: ', true_w.shape)
print('w.shape: ', w.shape) # torch.Size([1, 2])，线性层的权重是二维的矩阵
print('w 的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b 的估计误差：', true_b - b)