import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
TRAIN_DIR = "C:\\Users\\mario\\Desktop\\UNI\\Tesi_magistrale\\GANs\\CycleGAN_SMD\\dataset_CT_cycle\\TRAIN"
VAL_DIR = "C:\\Users\\mario\\Desktop\\UNI\\Tesi_magistrale\\GANs\\CycleGAN_SMD\\dataset_CT_cycle\\TEST"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 0 #4 out of memory
NUM_EPOCHS = 30 #si dovrebbe vedere qualcosa di interessante dopo le 15/20 epoche
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_NIR = "genNIR.pth.tar"
CHECKPOINT_GEN_VIS = "genVIS.pth.tar"
CHECKPOINT_CRITIC_NIR = "criticNIR.pth.tar"
CHECKPOINT_CRITIC_VIS = "criticVIS.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=128, height=128), #[256x256] out of memory
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
