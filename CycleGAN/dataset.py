from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class NIR_VISDataset(Dataset):
    def __init__(self, root_VIS, root_NIR, transform=None):
        self.root_VIS = root_VIS
        self.root_NIR = root_NIR
        self.transform = transform

        self.VIS_images = os.listdir(root_VIS)
        self.NIR_images = os.listdir(root_NIR)
        self.length_dataset = max(len(self.VIS_images), len(self.NIR_images)) # 1000, 1500
        self.VIS_len = len(self.VIS_images)
        self.NIR_len = len(self.NIR_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        vis_img = self.VIS_images[index % self.VIS_len]
        #dato che prima abbiamo preso la len massima tra i due dataset,
        #l'operazione sull'indice ci assicura di non ottenere errori.
        nir_img = self.NIR_images[index % self.NIR_len]
        #To load the immages
        vis_path = os.path.join(self.root_VIS, vis_img)
        nir_path = os.path.join(self.root_NIR, nir_img)

        vis_img = np.array(Image.open(vis_path).convert("RGB"))
        nir_img = np.array(Image.open(nir_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=vis_img, image0=nir_img)
            #Per effettuare le stesse transforms sulle due classi di immagini
            vis_img = augmentations["image"]
            nir_img = augmentations["image0"]

        return vis_img, nir_img



