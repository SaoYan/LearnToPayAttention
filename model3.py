import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock, GridAttentionBlock
from initialize import *

'''
attention after max-pooling
'''

class AttnVGG_grid(nn.Module):
    def __init__(self, num_classes, normalize_attn=False, init='kaimingNormal'):
        super(AttnVGG_grid, self).__init__()
        # conv blocks
        self.conv_block1 = ConvBlock(3, 64, 2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.pool = nn.AvgPool2d(2, stride=1)
        self.attn1 = GridAttentionBlock(256, 512, 256, 4, normalize_attn=normalize_attn)
        self.attn2 = GridAttentionBlock(512, 512, 256, 2, normalize_attn=normalize_attn)
        # final classification layer
        self.classify = nn.Linear(in_features=512+512+256, out_features=num_classes, bias=True)
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")
    def forward(self, x):
        # feed forward
        block1 = self.conv_block1(x)       # /1
        pool1 = F.max_pool2d(block1, 2, 2) # /2
        block2 = self.conv_block2(pool1)   # /2
        pool2 = F.max_pool2d(block2, 2, 2) # /4
        block3 = self.conv_block3(pool2)   # /4
        pool3 = F.max_pool2d(block3, 2, 2) # /8
        block4 = self.conv_block4(pool3)   # /8
        pool4 = F.max_pool2d(block4, 2, 2) # /16
        block5 = self.conv_block5(pool4)   # /16
        # pay attention
        N, __, __, __ = block5.size()
        g = self.pool(block5).view(N,512)
        c2, g1 = self.attn1(block3, block5)
        c3, g2 = self.attn2(block4, block5)
        g_hat = torch.cat((g,g1,g2), dim=1)
        out = self.classify(g_hat)
        return [out, None, c2, c3]
