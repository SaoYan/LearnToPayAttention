import numpy as np
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils

def visualize_attn_softmax(I, c, up_factor, nrow):
    if up_factor > 1:
        up_op = nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    N,C,W,H = c.size()
    a = F.softmax(c.view(N,C,-1), dim=2).view(N,C,W,H)
    if up_factor > 1:
        a = up_op(a)
    attn = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = img + attn
    min_val, max_val = np.min(vis), np.max(vis)
    vis = (vis - min_val) / (max_val - min_val)
    return torch.from_numpy(vis).permute(2,0,1)

def visualize_attn_sigmoid(I, c, up_factor, nrow):
    if up_factor > 1:
        up_op = nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    a = F.sigmoid(c)
    if up_factor > 1:
        a = up_op(a)
    attn = utils.make_grid(a, nrow=nrow, normalize=False)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = img + attn
    min_val, max_val = np.min(vis), np.max(vis)
    vis = (vis - min_val) / (max_val - min_val)
    return torch.from_numpy(vis).permute(2,0,1)

def compute_mac(result_file):
    # groundtruth
    with open('test.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        gt = [int(row[1]) for row in reader]

    # prediction
    pred = []
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = np.array(list(map(float, row)))
            pred.append(np.argmax(prob))

    # record muli class accuracy
    accuracy = []

    correct = 0
    for i in range(680): # NV
        if pred[i] == gt[i]:
            correct += 1
    accuracy.append(correct / 680)

    correct = 0
    for i in range(680, 793): # MEL
        if pred[i] == gt[i]:
            correct += 1
    accuracy.append(correct / 113)

    correct = 0
    for i in range(793, 904): # BKL
        if pred[i] == gt[i]:
            correct += 1
    accuracy.append(correct / 111)

    correct = 0
    for i in range(904, 956): # BCC
        if pred[i] == gt[i]:
            correct += 1
    accuracy.append(correct / 52)

    correct = 0
    for i in range(956, 989): # AKIEC
        if pred[i] == gt[i]:
            correct += 1
    accuracy.append(correct / 33)

    correct = 0
    for i in range(989, 1003): # VASC
        if pred[i] == gt[i]:
            correct += 1
    accuracy.append(correct / 14)

    correct = 0
    for i in range(1003, 1015): # DF
        if pred[i] == gt[i]:
            correct += 1
    accuracy.append(correct / 12)

    # print(accuracy)
    return np.mean(np.array(accuracy))
