import torch
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
d2l.plt.show()

y.backward(torch.ones_like(x), retain_graph=True) # 标量才能求梯度，y 是向量的情况，会先把 y 和 gradient 参数求内积变成标量再求梯度
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
d2l.plt.show()