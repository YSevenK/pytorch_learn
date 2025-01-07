import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度:{}".format(train_data_size))
print("测试数据集长度:{}".format(test_data_size))

train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)

# 创建网络模型
baby = Baby()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# lr = 1e-2
learning_rate = 0.01
optimizer = torch.optim.SGD(baby.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练次数
total_train_step = 0

# 记录测试次数
total_test_step = 0

# 训练轮数
epoch = 10

writer = SummaryWriter('./logs_train')

for i in range(epoch):
    print("----------->第{}轮训练开始".format(i + 1))
    # 训练步骤开始
    baby.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = baby(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数: {},Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    baby.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = baby(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集Loss: {}".format(total_test_loss))
    print("整体测试集正确率: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1
    # 保存方式1
    torch.save(baby, "baby_{}.pth".format(i))

    # 保存方式2
    # torch.save(baby.state_dict(), "baby_{}.pth".format(i))
    print("模型已保存")

writer.close()
