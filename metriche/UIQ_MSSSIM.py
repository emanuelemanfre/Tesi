#UIQ and MSSSIM
from sewar.full_ref import uqi, msssim
import cv2

path_real = "images/IR_real/ferry (1).jpg"
path_fake = 'images/IR_real/ferry (1).jpg'
imageA = cv2.imread(path_real)
# imageA = cv2.resize(imageA, (128, 128))
imageB = cv2.imread(path_fake)
msssim_ = msssim(imageA, imageB)
m = msssim_.real

print(" %.4f" %uqi(imageA, imageB))
print(" %.4f" %m)