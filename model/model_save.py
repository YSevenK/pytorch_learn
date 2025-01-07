import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(weights=None)

# 保存方式1 模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2 模型参数(官方推荐)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


# *注意1*
class Baby(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


baby = Baby()
torch.save(baby, "baby_method.pth")
