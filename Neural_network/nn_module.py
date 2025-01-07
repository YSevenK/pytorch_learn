import torch
from torch import nn


class Neural(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


neural = Neural()
x = torch.tensor(1.0)
output = neural(x)
print(output)
