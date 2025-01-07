import torch
import torchvision
from model_save import *

# 加载方式1
model1 = torch.load("vgg16_method1.pth")
# print(model1)

# 加载方式2
vgg16 = torchvision.models.vgg16(weights=None)
model2 = vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# print(model2)
# print(vgg16)

# 注意
model = torch.load("baby_method.pth")
print(model)
