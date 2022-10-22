import torch
from dataset import NIR_VISDataset
from utils import save_checkpoint #load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(disc_NIR, disc_VIS, gen_VIS, gen_NIR, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    nir_reals = 0
    nir_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (vis, nir) in enumerate(loop):
        vis = vis.to(config.DEVICE)
        nir = nir.to(config.DEVICE)

        # Train Discriminators NIR and VIS
        with torch.cuda.amp.autocast():
            fake_NIR = gen_NIR(vis)
            D_NIR_real = disc_NIR(nir)
            D_NIR_fake = disc_NIR(fake_NIR.detach())
            nir_reals += D_NIR_real.mean().item()
            nir_fakes += D_NIR_fake.mean().item()
            #Real 1, Fake 0
            D_NIR_real_loss = mse(D_NIR_real, torch.ones_like(D_NIR_real))
            D_NIR_fake_loss = mse(D_NIR_fake, torch.zeros_like(D_NIR_fake))
            D_NIR_loss = D_NIR_real_loss + D_NIR_fake_loss

            fake_VIS = gen_VIS(nir)
            D_VIS_real = disc_VIS(vis)
            D_VIS_fake = disc_VIS(fake_VIS.detach())
            D_VIS_real_loss = mse(D_VIS_real, torch.ones_like(D_VIS_real))
            D_VIS_fake_loss = mse(D_VIS_fake, torch.zeros_like(D_VIS_fake))
            D_VIS_loss = D_VIS_real_loss + D_VIS_fake_loss

            # put it together
            D_loss = (D_NIR_loss + D_VIS_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators NIR and VIS
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_NIR_fake = disc_NIR(fake_NIR)
            D_VIS_fake = disc_VIS(fake_VIS)
            loss_G_NIR = mse(D_NIR_fake, torch.ones_like(D_NIR_fake)) #fake horse qui è reale
            loss_G_VIS = mse(D_VIS_fake, torch.ones_like(D_VIS_fake))

            # cycle loss
            cycle_VIS = gen_VIS(fake_NIR) #dovrebbe restituirci la zebra
            cycle_NIR = gen_NIR(fake_VIS)
            cycle_VIS_loss = l1(vis, cycle_VIS)
            cycle_NIR_loss = l1(nir, cycle_NIR)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            # identity_zebra = gen_Z(zebra)
            # identity_horse = gen_H(horse)
            # identity_zebra_loss = l1(zebra, identity_zebra)
            # identity_horse_loss = l1(horse, identity_horse)

            # add all togethor
            G_loss = (
                loss_G_VIS
                + loss_G_NIR
                + cycle_VIS_loss * config.LAMBDA_CYCLE
                + cycle_NIR_loss * config.LAMBDA_CYCLE
                # + identity_horse_loss * config.LAMBDA_IDENTITY
                # + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 1 == 0:
            #*0.5+0.5 servono per fare "l'inverso" della normalizzazione così da
            #avere l'immagine colorata correttamente
            save_image(fake_NIR*0.5+0.5, f"fake_ships/NIR_{idx}.png")
            save_image(fake_VIS*0.5+0.5, f"fake_ships/VIS_{idx}.png")

        loop.set_postfix(NIR_real=nir_reals/(idx+1), NIR_fake=nir_fakes/(idx+1))



def main():
    disc_NIR = Discriminator(in_channels=3).to(config.DEVICE) #NIR
    disc_VIS = Discriminator(in_channels=3).to(config.DEVICE) #VIS
    gen_VIS = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_NIR = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_NIR.parameters()) + list(disc_VIS.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999), #paper
    )

    opt_gen = optim.Adam(
        list(gen_VIS.parameters()) + list(gen_NIR.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    #Le due funzioni di loss: L1 e MSE
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    #if config.LOAD_MODEL: #Per caricare un checkpoint
    #    load_checkpoint(
     #       config.CHECKPOINT_GEN_NIR, gen_NIR, opt_gen, config.LEARNING_RATE,
       # )
      #  load_checkpoint(
       #     config.CHECKPOINT_GEN_VIS, gen_VIS, opt_gen, config.LEARNING_RATE,
      #  )
      #  load_checkpoint(
       #     config.CHECKPOINT_CRITIC_NIR, disc_NIR, opt_disc, config.LEARNING_RATE,
       # )
       # load_checkpoint(
       #     config.CHECKPOINT_CRITIC_VIS, disc_VIS, opt_disc, config.LEARNING_RATE,
      #  )

    dataset = NIR_VISDataset(
        root_NIR=config.TRAIN_DIR+"/NIR_train", root_VIS=config.TRAIN_DIR+"/VIS_train", transform=config.transforms
    )
    val_dataset = NIR_VISDataset(
        root_NIR=config.VAL_DIR + "/NIR_test", root_VIS=config.VAL_DIR + "/VIS_test", transform=config.transforms
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_NIR, disc_VIS, gen_NIR, gen_NIR, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config: #ogni epoca salva i due gen e i due discriminatori
            save_checkpoint(gen_NIR, opt_gen, filename=config.CHECKPOINT_GEN_NIR)
            save_checkpoint(gen_VIS, opt_gen, filename=config.CHECKPOINT_GEN_VIS)
            save_checkpoint(disc_NIR, opt_disc, filename=config.CHECKPOINT_CRITIC_NIR)
            save_checkpoint(disc_VIS, opt_disc, filename=config.CHECKPOINT_CRITIC_VIS)

if __name__ == "__main__":
    main()
