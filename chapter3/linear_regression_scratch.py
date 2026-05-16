import torch
from d2l import torch as d2l
import random

# 合成数据
def synthetic_data(w, b, num_examples):
    # 生成 (num_examples, len(w)) 形状的矩阵，每个元素服从 N(0, 1)
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b # 理论值
    y += torch.normal(0, 0.01, y.shape) # 加上噪声
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) # feature的每一行是一组样本输入，labels的每个数是一个样本输出值。要学习 w 和 b
# print('features:', features[0], '\nlabel:', labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()

def data_iter(batch_size, features, labels):
    nums_examples = len(features)
    indices = list(range(nums_examples))
    random.shuffle(indices)
    for i in range(0, nums_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, nums_examples)])
        '''
        python 的特性，只要函数体里出现了 yield，python 就会把这个函数编译成 generator function 。
        这个函数会返回一个 generator ，通过 g.send(None) / next(g) ，g 会在原始的 yield 位置触发一次返回。
        这样就实现了懒生成多段数据。
        '''
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    '''线性回归模型'''
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    '''loss函数'''
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    '''Stochastic Gradient Descent 随机梯度下降'''
    with torch.no_grad(): # 临时关闭 自动求导 ，更新参数时不需要建计算图。torch.no_grad() 只在它的 with 代码块内部生效。
        for param in params:
            param -= lr * param.grad / batch_size # (w, b) -= 学习率 / |B| * (loss对各个参数的梯度) 
            param.grad.zero_() # 用一组数据更新后把 param 梯度清空，防止下一轮 backward 叠加梯度

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')