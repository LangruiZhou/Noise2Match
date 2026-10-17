import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import SimpleITK as sitk


# train
class GetTrainDataset(Dataset):
    def __init__(self, data_root, k, hu_max, hu_min):
        # load param
        self.data_root = data_root
        self.k = k
        self.hu_max = hu_max
        self.hu_min = hu_min
        # get training set
        self.ld_root = os.path.join(self.data_root, "train", "ld_recon")
        # get name list
        self.ld_name_list = os.listdir(self.ld_root)
        self.ld_name_list.sort()
        # get img num
        self.img_num = len(self.ld_name_list)
        # set transforms
        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.img_num

    # normalize -> [0, 1]
    def norm(self, img):
        img = img + self.hu_min
        img[img < self.hu_min] = self.hu_min
        img[img > self.hu_max] = self.hu_max 
        img = (img - self.hu_min) / (self.hu_max - self.hu_min)
        return np.float32(img)      # data range: [0, 1]

    def __getitem__(self, index):
        # is last img?
        if index >= self.img_num - self.k:
            index = index - self.k
        # same volume?
        if self.ld_name_list[index][0:4] != self.ld_name_list[index + self.k][0:4]:
            index = index - self.k
        # get img path
        ld_img1_path = os.path.join(self.ld_root, self.ld_name_list[index])
        ld_img2_path = os.path.join(self.ld_root, self.ld_name_list[index+self.k])
        # load imgs
        ld_img1 = sitk.ReadImage(ld_img1_path)
        ld_img1 = sitk.GetArrayFromImage(ld_img1).squeeze()
        ld_img1 = np.float32(ld_img1)
        ld_img2 = sitk.ReadImage(ld_img2_path)
        ld_img2 = sitk.GetArrayFromImage(ld_img2).squeeze()
        ld_img2 = np.float32(ld_img2)
        # normalize
        ld_img1 = self.norm(ld_img1)
        ld_img2 = self.norm(ld_img2)
        # to tensor
        ld_img1 = self.to_tensor(ld_img1)
        ld_img2 = self.to_tensor(ld_img2)

        return ld_img1, ld_img2


# test
class GetTestDataset(Dataset):
    def __init__(self, data_root, hu_max, hu_min):
        # load param
        self.data_root = data_root
        self.hu_max = hu_max
        self.hu_min = hu_min
        # get testing set
        self.fd_root = os.path.join(self.data_root, "test", "fd_recon")
        self.ld_root = os.path.join(self.data_root, "test", "ld_recon")
        # get name list
        self.fd_name_list = os.listdir(self.fd_root)
        self.fd_name_list.sort()
        self.ld_name_list = os.listdir(self.ld_root)
        self.ld_name_list.sort()
        # set transforms
        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])


    def __len__(self):
        return len(self.fd_name_list)

    # normalize -> [0, 1]
    def norm(self, img):
        img = img + self.hu_min
        img[img < self.hu_min] = self.hu_min
        img[img > self.hu_max] = self.hu_max 
        img = (img - self.hu_min) / (self.hu_max - self.hu_min)
        return np.float32(img)      # data range: [0, 1]

    def __getitem__(self, index):
        # get img path
        fd_img_path = os.path.join(self.fd_root, self.fd_name_list[index])
        ld_img_path = os.path.join(self.ld_root, self.ld_name_list[index])
        # load imgs
        fd_img = sitk.ReadImage(fd_img_path)
        fd_img = sitk.GetArrayFromImage(fd_img).squeeze()
        fd_img = np.float32(fd_img)
        ld_img = sitk.ReadImage(ld_img_path)
        ld_img = sitk.GetArrayFromImage(ld_img).squeeze()
        ld_img = np.float32(ld_img)
        # norm -> [0, 1]
        fd_img = self.norm(fd_img)
        ld_img = self.norm(ld_img)
        # to tensor
        fd_img = self.to_tensor(fd_img)
        ld_img = self.to_tensor(ld_img)

        return fd_img, ld_img

