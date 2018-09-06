import os
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from model1 import AttnVGG_before
from model2 import AttnVGG_after
from utilities import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description="LearnToPayAttn-test")
parser.add_argument("--outf", type=str, default="logs_test", help='path of log files')
parser.add_argument("--initialize", type=str, default="xavierUniform", help='kaimingNormal or kaimingUniform or xavierNormal or xavierUniform')
parser.add_argument("--attn_mode", type=str, default="after", help='insert attention modules before / after maxpooling layers')
parser.add_argument("--no_attention", action='store_true', help='turn down attention')

opt = parser.parse_args()

def main():
    # load data
    # CIFAR-100: 500 training images and 100 testing images per class
    print('\nloading the dataset ...\n')
    im_size = 32
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    testset = torchvision.datasets.CIFAR100(root='CIFAR100_data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=6)
    print('done')
    # load network
    print('\nloading the network ...\n')
    if not opt.no_attention:
        print('\nturn on attention ...\n')
        print('\npay attention %s maxpooling layers...\n' % opt.attn_mode)
    else:
        print('\nturn off attention ...\n')
    if opt.attn_mode == 'before':
        net = AttnVGG_before(im_size=im_size, num_classes=100, attention=not opt.no_attention, init=opt.initialize)
    elif opt.attn_mode == 'after':
        net = AttnVGG_after(im_size=im_size, num_classes=100, attention=not opt.no_attention, init=opt.initialize)
    else:
        raise NotImplementedError("Invalid attention mode!")
    # move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0,1]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.load_state_dict(torch.load('net.pth'))
    model.eval()
    print('done')

    # testing
    print('\nstart testing ...\n')
    writer = SummaryWriter(opt.outf)
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            print("[%d / %d]" % (i+1, len(testset)))
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            if not opt.no_attention:
                if opt.attn_mode == 'before':
                    min_up_factor = 1
                elif opt.attn_mode == 'after':
                    min_up_factor = 2
                pred, c1, c2, c3 = model.forward(inputs)
                # accuracy
                predict = torch.argmax(pred, 1)
                total += labels.size(0)
                correct += torch.eq(predict, labels).sum().double().item()
                # images
                I = utils.make_grid(inputs, nrow=10, normalize=True, scale_each=True)
                attn1 = visualize_attn_softmax(I, c1, up_factor=min_up_factor, nrow=10)
                attn2 = visualize_attn_softmax(I, c2, up_factor=min_up_factor*2, nrow=10)
                attn3 = visualize_attn_softmax(I, c3, up_factor=min_up_factor*4, nrow=10)
                writer.add_image('train/image', I, i)
                writer.add_image('train/attention_map_1', attn1, i)
                writer.add_image('train/attention_map_2', attn2, i)
                writer.add_image('train/attention_map_3', attn3, i)
        print("accuracy on test data: %.2f" % (100*correct/total))

if __name__ == "__main__":
    main()
