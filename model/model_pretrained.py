import torchvision
from torch import nn
from torchvision.models import VGG16_Weights
import os

os.environ['TORCH_HOME'] = 'D:/MyProject/PythonProject/learn_pytorch/model'

# train_data = torchvision.datasets.ImageNet("./data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('./data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
