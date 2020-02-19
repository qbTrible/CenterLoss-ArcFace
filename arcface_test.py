# -*- coding: utf-8 -*- 
# @Time : 2019-11-20 11:14 
# @Author : Trible

# Using MTCNN for face detection

from test05 import DarkNet
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from MTCNN.detect import Detector


def convert_to_square(bbox):
    square_bbox = bbox.copy()
    if len(bbox) == 0:
        return []
    h = bbox[3] - bbox[1]
    w = bbox[2] - bbox[0]
    max_side = np.maximum(h, w)
    square_bbox[0] = bbox[0] + w * 0.5 - max_side * 0.5
    square_bbox[1] = bbox[1] + h * 0.5 - max_side * 0.5
    square_bbox[2] = square_bbox[0] + max_side
    square_bbox[3] = square_bbox[1] + max_side

    return square_bbox

detector = Detector()
test_img = Image.open(r"E:\01\face_recognition\test_img\古巨基.jpg").convert("RGB")
boxes = detector.detect(test_img)
for box in boxes:
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
crop = convert_to_square([x1, y1, x2, y2])
test_img = test_img.crop(crop).resize((128, 128), 1)

net = DarkNet([1, 1, 2, 2, 1]).cuda()
if os.path.exists("model/arcface01.pth"):
    net.load_state_dict(torch.load("model/arcface01.pth"))
test_data = torch.Tensor((np.array(test_img))/255. - 0.5)
test_data = test_data.permute(2, 0, 1).unsqueeze(0).cuda()
feature, output = net(test_data)
test_f = F.normalize(feature.detach().cpu())

img_list = os.listdir(r"face\古巨基")
max_file = ""
max_dis = -float('inf')
max_img = 0
for img_name in img_list:
    img_path = r"face\古巨基\%s" % img_name
    img = Image.open(img_path).resize((128, 128), 1)
    data = torch.Tensor((np.array(img))/255. - 0.5)
    data = data.permute(2, 0, 1).unsqueeze(0).cuda()
    feature1, output1 = net(data)
    lib_f = F.normalize(feature1.detach().cpu())
    distance = test_f.mm(lib_f.t())
    if distance.item() > max_dis:
        max_dis = distance.item()
        max_file = img_name
        max_img = np.array(img)
    print(img_name, "相似度：", distance.item())
print("最相似图片：", max_file, "相似度：", max_dis)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(test_img)
plt.subplot(1, 2, 2)
plt.imshow(max_img)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.suptitle("相似度：%.5f" % max_dis)
plt.show()


