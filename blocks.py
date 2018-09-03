import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features):
        super(LinearAttentionBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g).view(N,1,-1) # batch_sizex1x(WH)
        a = F.softmax(c, dim=2).view(N,1,W,H)#.add(1./W/H)
        g = torch.mul(a.expand_as(l), l).view(N,C,-1).sum(dim=2) # batch_sizexC
        return c.view(N,1,W,H), g

class GridAttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=True, output_transform=False):
        super(GridAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.output_transform = output_transform
        self.up_op = nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=True)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
        if self.output_transform:
            self.trans = nn.Sequential(
                nn.Conv2d(in_channels=in_features_l, out_channels=in_features_l, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(in_features_l)
            )
    def forward(self, l, g):
        N, C, W, H = l.size()
        up_g = self.up_op(g)
        c = self.phi(F.relu(self.W_l(l) + self.W_g(up_g))) # batch_sizex1xWxH
        # up_g = self.up_op(self.W_g(g))
        # c = self.phi(F.relu(self.W_l(l) + up_g))
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = F.sigmoid(c)
        f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
        if self.output_transform:
            f = self.trans(f)
        return c.view(N,1,W,H), f.view(N,C,-1).sum(dim=2)
