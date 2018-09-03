import os
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from model1 import AttnVGG
from utilities import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser(description="LearnToPayAttn-ImageNet")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--no_attention", action='store_true', help='turn down attention')

opt = parser.parse_args()

def main():
    # load data
    # CIFAR-100: 500 training images and 100 testing images per class
    print('\nloading the dataset ...\n')
    num_aug = 1
    im_size = 224
    imagenet_path = '/home/yiqiyan/projects/rrg-hamarneh/data/ILSVRC2012'
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(im_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(imagenet_path, 'train'), transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=6)
    testset = torchvision.datasets.ImageFolder(root=os.path.join(imagenet_path, 'val'), transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=6)
    print('done')
    # load network
    print('\nloading the network ...\n')
    net = AttnVGG(in_size=im_size, num_classes=1000, attention=not opt.no_attention)
    criterion = nn.CrossEntropyLoss()
    # move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0,1,2,3]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    criterion.to(device)
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
    lr_lambda = lambda epoch : np.power(0.1, int(epoch/30))
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    print('done')

    # training
    print('\nstart training ...\n')
    step = 0
    running_avg_accuracy = 0
    writer = SummaryWriter(opt.outf)
    for epoch in range(opt.epochs):
        images_test_disp = []
        # adjust learning rate
        scheduler.step()
        print("\nepoch %d learning rate %f\n" % (epoch, optimizer.param_groups[0]['lr']))
        # run for one epoch
        for aug in range(num_aug):
            for i, data in enumerate(trainloader, 0):
                # warm up
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                if i == 0: # archive images in order to save to logs
                    images_test_disp.append(inputs[0:16,:,:,:])
                # forward
                pred, __, __, __ = model.forward(inputs)
                # backward
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                # display results
                if i % 10 == 0:
                    model.eval()
                    pred, __, __, __ = model.forward(inputs)
                    predict = torch.argmax(pred, 1)
                    total = labels.size(0)
                    correct = torch.eq(predict, labels).sum().double().item()
                    accuracy = correct / total
                    running_avg_accuracy = 0.9*running_avg_accuracy + 0.1*accuracy
                    writer.add_scalar('train/loss', loss.item(), step)
                    writer.add_scalar('train/accuracy', accuracy, step)
                    writer.add_scalar('train/running_avg_accuracy', running_avg_accuracy, step)
                    print("[epoch %d][aug %d/%d][%d/%d] loss %.4f accuracy %.2f%% running avg accuracy %.2f%%"
                        % (epoch, aug, num_aug-1, i, len(trainloader)-1, loss.item(), (100*accuracy), (100*running_avg_accuracy)))
                step += 1
                if i >= 10:
                    break
        # the end of each epoch: test & log
        print('\none epoch done, saving records ...\n')
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
        if epoch == opt.epochs / 2:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net%d.pth' % epoch))
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            # log scalars
            for i, data in enumerate(testloader, 0):
                images_test, labels_test = data
                images_test, labels_test = images_test.to(device), labels_test.to(device)
                if i == 0: # archive images in order to save to logs
                    images_test_disp.append(inputs[0:16,:,:,:])
                pred_test, __, __, __ = model.forward(images_test)
                predict = torch.argmax(pred_test, 1)
                total += labels_test.size(0)
                correct += torch.eq(predict, labels_test).sum().double().item()
            writer.add_scalar('test/accuracy', correct/total, epoch)
            print("\n[epoch %d] accuracy on test data: %.2f%%\n" % (epoch, 100*correct/total))
            # log images
            if not opt.no_attention:
                # training data
                __, c1, c2, c3 = model.forward(images_test_disp[0])
                I = utils.make_grid(images_test_disp[0], nrow=4, normalize=True, scale_each=True)
                attn1 = visualize_attn_softmax(I, c1, up_factor=2, nrow=4)
                attn2 = visualize_attn_softmax(I, c2, up_factor=4, nrow=4)
                attn3 = visualize_attn_softmax(I, c3, up_factor=8, nrow=4)
                writer.add_image('train/image', I, epoch)
                writer.add_image('train/attention_map_1', attn1, epoch)
                writer.add_image('train/attention_map_2', attn2, epoch)
                writer.add_image('train/attention_map_3', attn3, epoch)
                # test data
                __, c1, c2, c3 = model.forward(images_test_disp[1])
                I = utils.make_grid(images_test_disp[1], nrow=4, normalize=True, scale_each=True)
                attn1 = visualize_attn_softmax(I, c1, up_factor=2, nrow=4)
                attn2 = visualize_attn_softmax(I, c2, up_factor=4, nrow=4)
                attn3 = visualize_attn_softmax(I, c3, up_factor=8, nrow=4)
                writer.add_image('test/image', I, epoch)
                writer.add_image('test/attention_map_1', attn1, epoch)
                writer.add_image('test/attention_map_2', attn2, epoch)
                writer.add_image('test/attention_map_3', attn3, epoch)

if __name__ == "__main__":
    main()
