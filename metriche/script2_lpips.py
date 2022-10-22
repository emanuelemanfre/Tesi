# LPIPS
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import lpips

path_real_IR = "images/IR_real/speedboat (18).jpg"
path_fake_IR = 'images/IR_real/ferry (1).jpg'
path_real_VIS = "images/IR_real/speedboat (18).jpg"
path_fake_VIS = 'images/IR_real/ferry (1).jpg'

loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

img0_IR = Image.open(path_real_IR).resize((128, 128)) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1_IR = Image.open(path_fake_IR).resize((128, 128))
img_real_IR = TF.to_tensor(img0_IR)
img_fake_IR = TF.to_tensor(img1_IR)
d_IR = loss_fn_alex(img_real_IR, img_fake_IR)
img0_VIS = Image.open(path_real_VIS).resize((128, 128)) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1_VIS = Image.open(path_fake_VIS).resize((128, 128))
img_real_VIS = TF.to_tensor(img0_VIS)
img_fake_VIS = TF.to_tensor(img1_VIS)
d_VIS = loss_fn_alex(img_real_VIS, img_fake_VIS)
print("d_VIS:  %.5f" %d_VIS.item())
print("d_IR:  %.5f" %d_IR.item())
