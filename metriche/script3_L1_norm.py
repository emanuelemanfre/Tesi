#L1 norm
import torch
import torchvision.transforms.functional as TF
import cv2
from PIL import Image
path_real = "C:\\Users\\mario\\Desktop\\UNI\\Tesi_magistrale\\GANs\\metriche\\images\\VIS_real\\speedboat (18).jpg"
path_fake = "C:\\Users\\mario\\Desktop\\UNI\\Tesi_magistrale\\GANs\\metriche\\images\\VIS_real\\ferry (3).jpg"

real_image = cv2.imread(path_real) #[3,256,256]
fake_image = cv2.imread(path_fake) #la considero fake per testare
N = TF.to_tensor(real_image).view(-1).size(0)//3 #total number of pixels in the image
img_real = Image.open(path_real).convert('L')
img_fake = Image.open(path_fake).convert('L')
img_real = img_real.resize((128, 128))
img_fake = img_fake.resize((128, 128))
Y = TF.to_tensor(img_real)
G = TF.to_tensor(img_fake)
Y = Y[0]
G = G[0]
L1=0


for i, j in enumerate(Y):
  L1 = L1 + 1/N * sum(abs((G[i] - j)))

print("L1 norm is: ", L1.item())


