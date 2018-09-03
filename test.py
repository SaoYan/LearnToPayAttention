import os
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from model2 import AttnVGG
from utilities import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description="LearnToPayAttn-test")
parser.add_argument("--outf", type=str, default="logs_test", help='path of log files')
parser.add_argument("--no_attention", action='store_true', help='turn down attention')

opt = parser.parse_args()

def main():
    # load data
    # CIFAR-100: 500 training images and 100 testing images per class
    print('\nloading the dataset ...\n')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4782, 0.4782, 0.4782), (0.2683, 0.2683, 0.2683))
    ])
    testset = torchvision.datasets.CIFAR100(root='CIFAR100_data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)
    print('done')
    # load network
    print('\nloading the network ...\n')
    net = AttnVGG(in_size=32, num_classes=100, attention=not opt.no_attention)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0,1]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.load_state_dict(torch.load('net.pth'))
    model.eval()
    print('done')

    # testing
    print('\nstart testing ...\n')
    writer = SummaryWriter(opt.outf)
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            if not opt.no_attention:
                __, c1, c2, c3 = model.forward(inputs)
                I = utils.make_grid(inputs, nrow=4, normalize=True, scale_each=True)
                # attn1 = visualize_attn_softmax(I, c1, up_factor=2, nrow=4)
                # attn2 = visualize_attn_softmax(I, c2, up_factor=4, nrow=4)
                # attn3 = visualize_attn_softmax(I, c3, up_factor=8, nrow=4)
                attn1 = visualize_attn_sigmoid(I, c1, up_factor=2, nrow=4)
                attn2 = visualize_attn_sigmoid(I, c2, up_factor=4, nrow=4)
                attn3 = visualize_attn_sigmoid(I, c3, up_factor=8, nrow=4)
                writer.add_image('train/image', I, i)
                writer.add_image('train/attention_map_1', attn1, i)
                writer.add_image('train/attention_map_2', attn2, i)
                writer.add_image('train/attention_map_3', attn3, i)
                print("%d done" % i)
                if i >= 10:
                    break

if __name__ == "__main__":
    main()
