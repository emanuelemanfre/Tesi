#SSIM
from skimage.metrics import structural_similarity as ssim
import imutils
import cv2
path_real = "image_real.jpg"
path_fake = 'image_fake.png'
imageA = cv2.imread(path_real)
imageA = cv2.resize(imageA, (128, 128))
imageB = cv2.imread(path_fake)

#Convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)


(score, diff) = ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

print("SSIM: {}".format(score))





