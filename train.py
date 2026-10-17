import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
# from local
import reg
import config
import dataset
import model
import utils


# train
def noise2match_train(cfg):
    # set device
    if torch.cuda.is_available():
        gpu_idx = 0
        device = "cuda"
        torch.cuda.set_device(device=gpu_idx)
        print(f"device: {device} - {torch.cuda.get_device_name(device=gpu_idx)}")
    else:
        device = "cpu"
        print(f"device: {device}")

    # set model
    unet = model.UNet(in_channels=1, out_channels=1).to(device)
    regnet = reg.RegNet(height=cfg.img_size, width=cfg.img_size, input_nc_a=1, input_nc_b=1).to(device)
    resampler = reg.ApplyRegTrans().to(device)

    # set optim
    optimizer_unet = optim.Adam(unet.parameters(), lr=cfg.lr)
    optimizer_regnet = optim.Adam(regnet.parameters(), lr=cfg.lr)

    # loss func
    mse_loss = nn.MSELoss().to(device)
    mse_loss_ = nn.MSELoss(reduction="none").to(device)

    # load weights
    epoch_begin = 0
    if os.path.exists(cfg.weight_path):
        # load model param
        weight = torch.load(cfg.weight_path)
        epoch_begin = weight["epoch"] + 1
        unet.load_state_dict(weight["unet"], strict=False)
        regnet.load_state_dict(weight["regnet"], strict=False)
        print("Weights load successfully!")
    else:
        print("No previous weight.")

    # set train dataloader
    train_dataloader = DataLoader(dataset.GetTrainDataset(cfg.data_root, cfg.k, cfg.hu_max, cfg.hu_min),
                                  batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    # start train
    for epoch in range(epoch_begin, cfg.epoch_num):
        train_iterator = tqdm(train_dataloader)
        total_loss = 0
        # set model mode
        unet.train()
        regnet.train()
        for index, data in enumerate(train_iterator):
            # load inputs
            ld_img1 = data[0].to(device)
            ld_img2 = data[1].to(device)
            # clear grad
            optimizer_unet.zero_grad()
            optimizer_regnet.zero_grad()
            # forward
            noise_pred1 = unet(ld_img1)
            fd_pred1 = ld_img1 - noise_pred1
            noise_pred2 = unet(ld_img2)
            fd_pred2 = ld_img2 - noise_pred2
            # x[i] -> x[i+1]
            phi_1 = regnet(fd_pred1, fd_pred2)
            fd_pred1_reg_corr = resampler(fd_pred1, phi_1)
            with torch.no_grad():
                phi_1_no_grad = phi_1.detach()
            fd_pred1_reg_rec = resampler(fd_pred1, phi_1_no_grad)
            # x[i+1] -> x[i]
            phi_2 = regnet(fd_pred2, fd_pred1)
            fd_pred2_reg_corr = resampler(fd_pred2, phi_2)
            with torch.no_grad():
                phi_2_no_grad = phi_2.detach()
            fd_pred2_reg_rec = resampler(fd_pred2, phi_2_no_grad)

            # loss
            # loss clip thresh
            th_clip = (cfg.th / (cfg.hu_max - cfg.hu_min)) ** 2
            # corr loss
            loss_corr1 = mse_loss_(fd_pred1_reg_corr, fd_pred2)
            loss_corr2 = mse_loss_(fd_pred2_reg_corr, fd_pred1)
            # get msk
            msk_corr1 = loss_corr1 > th_clip
            msk_corr2 = loss_corr2 > th_clip
            # clip corr loss
            loss_corr1 = torch.where(msk_corr1, th_clip, loss_corr1)
            loss_corr2 = torch.where(msk_corr2, th_clip, loss_corr2)
            loss_corr = torch.mean(loss_corr1) + torch.mean(loss_corr2)
            # rec loss
            loss_rec1 = mse_loss_(fd_pred1_reg_rec, ld_img2)
            loss_rec2 = mse_loss_(fd_pred2_reg_rec, ld_img1)
            loss_rec1 = torch.where(msk_corr1, 0, loss_rec1)
            loss_rec2 = torch.where(msk_corr2, 0, loss_rec2)
            loss_rec = torch.mean(loss_rec1) + torch.mean(loss_rec2)

            # smooth loss
            loss_smooth = reg.smooth_loss(phi_1) + reg.smooth_loss(phi_2)
            # loss total
            loss = loss_rec + cfg.lambda_corr * loss_corr + cfg.lambda_smooth * loss_smooth
            total_loss += loss.item()

            # backward
            loss.backward()
            # update
            optimizer_unet.step()
            optimizer_regnet.step()

            # display train img
            if index % 100 == 0:
                save_path = f"{cfg.train_img_root}/epoch{epoch}_train{index}.jpg"
                img_display = torch.cat([ld_img1, fd_pred1, ld_img2, fd_pred2], dim=-1)
                img_display = utils.trans_to_display_window(img_display[0])
                cv2.imwrite(save_path, img_display)
                msk_save_path = f"{cfg.train_img_root}/epoch{epoch}_train{index}_msk.jpg"
                msk_show = torch.cat([torch.abs(fd_pred1-fd_pred2)*10, torch.abs(fd_pred1_reg_corr-fd_pred2)*10, msk_corr1, msk_corr2], dim=-1)
                msk_show = msk_show[0].float() * 255.0
                msk_show = msk_show.squeeze().detach().cpu().numpy()
                cv2.imwrite(msk_save_path, msk_show)

            train_iterator.desc = "epoch{} loss:{}".format(epoch, loss.item())

        # finish one epoch
        # save weight
        weight_save_path = f"{cfg.weight_save_root}/epoch{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "unet": unet.state_dict(),
            "regnet": regnet.state_dict()
        }, weight_save_path)

        # calculate mean loss
        mean_loss = total_loss / len(train_dataloader)
        
        # print info
        print(f"loss:{mean_loss}")


if __name__ == '__main__':
    cfg = config.get_config()
    print(cfg)
    noise2match_train(cfg)











