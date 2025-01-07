import torchvision
from PIL import Image
import torch
from torch import nn

img_path = './imgs/img_2.png'
img = Image.open(img_path)
# print(img)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
img = transform(img)

# print(img.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 搭建神经网络
class Baby(nn.Module):
    def __init__(self):
        super(Baby, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("baby_49_gpu.pth")
model = model.to(device)

# print(model)
img = torch.reshape(img, (1, 3, 32, 32)).to(device)
model.eval()
with torch.no_grad():
    output = model(img)
output = output.cpu()
print(output)
print(output.argmax(1))
