#PSNR
from math import log10, sqrt
import cv2
from PIL import Image
import numpy as np


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main():
    path_real = "image_real.jpg"
    path_fake = "image_fake.png"
    original = cv2.imread(path_real)
    compressed = cv2.imread(path_fake, 1)
    value = PSNR(original, compressed)
    print(f"PSNR value is {value} dB")


if __name__ == "__main__":
    main()
