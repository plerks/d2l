import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# print(net)

loss = nn.CrossEntropyLoss(reduction='none') # 不聚合样本

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10 # 训练集重复 10 遍

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval() # 切换到推理(评估)模式
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel()) # 分别累加正确数和总数。y.numel() 返回 y 的元素个数
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = d2l.Accumulator(3) # metric[0] = loss 总和，metric[1] = 正确预测数，metric[2] = 总样本数
    total_images_processed = 0
    for X, y in train_iter: # X 是图片 flatten 后的结果，y 是输出
        total_images_processed += y.numel()
        print(f"已处理图片: {total_images_processed}")
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad() # 清空梯度
            l.mean().backward() # 求平均 loss ，然后反向传播
            updater.step() # 反向传播完了，更新参数。梯度会更新到 X, y tensor 对象里，step() 会适用更新
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs): # 跑多轮训练集。神经网络即使数据完全一样，重复训练多遍（多个 epoch）仍然会持续提升效果。
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc, ))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss # assert 条件, 错误消息
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

d2l.plt.show()

def predict_ch3(net, test_iter, n = 6):
    for X, y in test_iter: # 读一个 batch 的数据。python 只有函数级作用域，X, y 在循环结束后仍然存在
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1)) # axis=1，把列拍平找最大值
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
d2l.plt.show()