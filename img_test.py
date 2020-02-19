# -*- coding: utf-8 -*- 
# @Time : 2019-12-02 13:55 
# @Author : Trible 

from test05 import DarkNet
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

test_imgs = os.listdir("test_img")
net = DarkNet([1, 1, 2, 2, 1]).cuda()
# if os.path.exists("model/center_net.pth"):
#     net.load_state_dict(torch.load("model/center_net.pth"))
if os.path.exists("model/center_net.pth"):
    net.load_state_dict(torch.load("model/center_net.pth"))

for img_name in test_imgs:
    max_file = ""
    max_dis = -float('inf')
    max_img = 0
    img = Image.open("test_img/%s" % img_name)
    test_img = img.resize((128, 128), 1)
    test_data = torch.Tensor((np.array(test_img)) / 255. - 0.5)
    test_data = test_data.permute(2, 0, 1).unsqueeze(0).cuda()
    feature, output = net(test_data)
    test_f = F.normalize(feature.detach().cpu())
    with open("vector.txt", "r") as f:
        for line in f.readlines():
            name, vector = eval(line)
            distance = test_f.mm(torch.Tensor(vector).t())
            if distance.item() > max_dis:
                max_dis = distance.item()
                max_file = name
                # max_img = np.array(img)
        flag = img_name.split(".")[0] == max_file.split(".")[0].split("0")[0]
        print(img_name, " 最相似图片：", max_file, " 相似度：", max_dis, flag)

