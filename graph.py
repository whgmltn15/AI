import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# Model
model = nn.Linear(1, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

total_epoch = 2000

def customLoss(pred, gt):
    loss = (pred - gt) * (pred - gt)

    return loss.mean()

for epoch in range(total_epoch):

    pred = model(x_train)

    #cost = F.mse_loss(pred, y_train)
    cost = F.mse_loss(pred, y_train)

    # gradient init 0
    optimizer.zero_grad()

    cost.backward()

    # a, b update
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost : {:.6f}'.format(epoch, total_epoch, cost.item()))
        plt.clf()
        plt.xlim(0, 15)
        plt.ylim(0, 12)
        plt.scatter(x_train.data.numpy(), y_train.data.numpy())
        plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'b--')
        plt.title(
            'loss={:.4}, w={:.4}, b={:.4}'.format(cost.data.item(), model.weight.data.item(), model.bias.data.item()))
        plt.show()