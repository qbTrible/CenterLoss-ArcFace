# -*- coding: utf-8 -*- 
# @Time : 2019-11-19 20:31 
# @Author : Trible 

from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image
import random

class FaceDataset(Dataset):

    def __init__(self, path):
        # self.path = path
        self.dataset = []
        per_list = os.listdir(path)[:10]
        print(per_list)
        self.dict = {per_list[per]: per for per in range(len(per_list))}
        # self.dict = {"李现": 0, "林允儿": 1}
        for key in self.dict:
            self.dataset.extend(os.path.join(path, key, file) for file in os.listdir(os.path.join(path, key)))

    def __getitem__(self, index):
        img = Image.open(self.dataset[index]).resize((128, 128), 1)
        img_data = torch.Tensor(np.array(img) / 255. - 0.5)
        img_data = img_data.permute(2, 0, 1)
        label = torch.tensor([self.dict[self.dataset[index].split("\\")[-2]]])

        return img_data, label

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':

    dataset = FaceDataset("E:\\face")
    print(dataset)
