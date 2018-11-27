# coding:utf-8
import numpy as np

def range_matrix(r,c):
    return np.arange(r*c).reshape((r, c))*0.1+0.1


input_len = 3
num_classes = 3
n, p = 0, 0
hidden_size = 2 # size of hidden layer of neurons
seq_length = 3 # number of steps to unroll the RNN for
learning_rate = 1

data_len = 50000
x = np.arange(data_len)+1

ground_truth = [(x[i-1] + x[i-2]) % 3 for i in range(data_len)]

# model parameters
U = range_matrix(hidden_size, input_len) # input to hidden
W = range_matrix(hidden_size, hidden_size) # hidden to hidden
V = range_matrix(num_classes, hidden_size) # hidden to output
bs = np.zeros((hidden_size, 1)) # hidden bias
bo = np.zeros((num_classes, 1)) # output bias


def forward_and_backprop(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    # seq_length:表示我们每一次进行反向传播之前需要多少个step
    # 我们传入的是3,表示传入三组数据之后进行一次反向传播,也就是它能学习到三个时间长度的内容
    for t in xrange(seq_length):
        xs[t] = inputs[t:t + 3].reshape(input_len, 1)  # make a matrix(rank 2)
        #     hs[t] = np.tanh('''Fill your code HERE''') #计算hidden state。激活函数使用tanh
        hs[t] = np.tanh(np.dot(U, xs[t]))
        #     ys[t] = '''Fill your code HERE''' #计算output logits。注意这里没有激活函数，我们将在下一步计算softmax
        ys[t] = np.dot(V, hs[t]) + bo

        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # softmax

        #     loss = '''Fill your code HERE''' # 计算交叉熵
        loss = np.sum(np.multiply(ps[t], targets[t:t + 3]))

    # 反向传播过程
    dU, dW, dV = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
    dbs, dbo = np.zeros_like(bs), np.zeros_like(bo)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(xrange(seq_length)):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # softmax-交叉熵delta： y-t
        dV = '''Fill your code HERE'''  # V-nabla
        dbo = '''Fill your code HERE'''  # bo-nabla
        dh = '''Fill your code HERE'''  # backprop into hidden-state
        dhraw = (1 - hs[t] * hs[t]) * dh  # tanh的导数是1-logits^2
        dbs = '''Fill your code HERE'''  # bs-nabla
        dU = '''Fill your code HERE'''  # U-nabla
        if t > 0:
            dW = '''Fill your code HERE'''  # W-nabla
        dhnext = dhraw

    return loss, dU, dW, dV, dbs, dbo, hs[seq_length - 1]







for n in range(5):
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(x) or n == 0:
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 2 # go from start of data
  inputs =  x[p-2:p+seq_length]
  targets = ground_truth[p:p+seq_length]
  loss, dU, dW, dV, dbs, dbo, hprev = forward_and_backprop(inputs, targets, hprev)
  # perform parameter update with Adagrad
  for param, dparam in zip([U, W, V, bs, bo],
                                [dU, dW, dV, dbs, dbo]):
    param += -learning_rate * dparam #sgd

  p += seq_length # move data pointer

print('U:')
print(U)
print('W:')
print(W)
print('V:')
print(V)