import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784 # 图片 flatten 成 28 * 28 = 784 个输入特征
num_outputs = 10 # 10 个输出类别

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) # 随机初始化是为了消除对称性
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    # reshape(-1, 784)，python 会先铺平成一维数组再按新形状填充
    # X.reshape 后每行是个 784 个像素值（一张铺平的图）
    # XW 的结果是，多个样本，每个都在分类位置上得到值
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    # 在 [] 内部，逗号会创建元组。这里 y_hat[range(len(y_hat)), y] 会得到每个样本在真实类别位置上的预测概率
    # y_hat[range(len(y_hat)), y] 依次取 y_hat[i, y[i]]
    return -torch.log(y_hat[range(len(y_hat)), y])

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
            updater.step() # 反向传播完了，更新参数
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc, ))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss # assert 条件, 错误消息
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

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
