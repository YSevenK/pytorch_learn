from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
img_path = "data/val/bees/57459255_752774f1b2.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("train", img_array, 1, dataformats='HWC')

for i in range(1, 100):
    writer.add_scalar('y=x', i, i)

writer.close()
