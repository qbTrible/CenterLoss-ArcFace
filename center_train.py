# -*- coding: utf-8 -*- 
# @Time : 2019-12-02 11:26 
# @Author : Trible 

from test05 import DarkNet
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from CenterLoss import CenterLoss
from MyData import FaceDataset
import os
import torch.optim.lr_scheduler as lr_scheduler
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# Using t-SNE for dimention reduction
def tsne_process(X):
    tsne = TSNE(n_components=2, init="pca", random_state=0)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

def decet(feature, targets, epoch, save_path):
    color = ["red", "black", "yellow", "green", "pink", "gray", "lightgreen", "orange", "blue", "teal"]
    cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.ion()
    plt.clf()
    for j in cls:
        mask = [targets.numpy() == j]
        feature_ = feature[mask]
        x = feature_[:, 1]
        y = feature_[:, 0]
        label = cls
        plt.plot(x, y, ".", color=color[j])
        plt.legend(label, bbox_to_anchor=(1, 1), loc=2)  # 如果写在plot上面，则标签内容不能显示完整
        plt.title("epoch={}".format(str(epoch)))

    plt.savefig('%s/%03d.jpg' % (save_path, epoch))
    plt.axis('equal')
    plt.subplots_adjust(right=0.8)
    plt.draw()
    plt.pause(0.001)


if __name__ == "__main__":
    net = DarkNet([1, 1, 2, 2, 1]).cuda()
    if os.path.exists("model/center_net.pth"):
        net.load_state_dict(torch.load("model/center_net.pth"))
    arcloss = CenterLoss(10, 512).cuda()
    if os.path.exists("model/centerloss.pth"):
        arcloss.load_state_dict(torch.load("model/centerloss.pth"))
    optmizer = optim.RMSprop(net.parameters(), lr=0.001, alpha=0.9)
    optmizer_arc = optim.Adam(arcloss.parameters())
    scheduler = lr_scheduler.StepLR(optmizer, 20, gamma=0.8)
    nllloss = nn.CrossEntropyLoss().cuda()

    Batch_Size = 128
    face_data = FaceDataset("face")
    train_loader = data.DataLoader(face_data, batch_size=Batch_Size, shuffle=True, num_workers=4)

    count = 0
    while True:
        feat = []
        target = []
        scheduler.step()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda().squeeze()
            xs, ys = net(x)

            value = torch.argmax(ys, dim=1)
            arc_loss = arcloss(xs, y)
            nll_loss = nllloss(ys, y)
            loss = nll_loss + arc_loss
            acc = torch.sum((value == y).float())/len(y)
            optmizer.zero_grad()
            optmizer_arc.zero_grad()
            loss.backward()
            optmizer.step()
            optmizer_arc.step()
            feat.append(xs)
            target.append(y)
            print("epoch:", count, "batch:", i, "acc", acc.item())
            print("loss:", loss.item(), "nll_loss:", nll_loss.item(), "arcloss:", arc_loss.item())
            torch.save(net.state_dict(), "model/center_net.pth")
            torch.save(arcloss.state_dict(), "model/centerloss.pth")

        features = torch.cat(feat, 0)
        features = tsne_process(features.data.cpu())
        targets = torch.cat(target, 0)
        decet(features, targets.data.cpu(), count, "img/center_img")
        count += 1