from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = 'data/train/ants_image/49375974_e28ba6f17e.jpg'
img_path_abs = r'D:\MyProject\PythonProject\learn_pytorch\data\train\ants_image\49375974_e28ba6f17e.jpg'
img = Image.open(img_path)

writer = SummaryWriter('logs')

# print(img)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# print(tensor_img)

writer.add_image('Tensor_img', tensor_img)
writer.close()
