# LearnToPayAttention  

[![AUR](https://img.shields.io/aur/license/yaourt.svg?style=plastic)](LICENSE)   

PyTorch implementation of ICLR 2018 paper [Learn To Pay Attention](http://www.robots.ox.ac.uk/~tvg/publications/2018/LearnToPayAttention_v5.pdf)  

![](https://github.com/SaoYan/LearnToPayAttention/blob/master/fig/learn_to_pay_attn.png)

My implementation is based on "(VGG-att3)-concat-pc" in the paper, and I trained the model on CIFAR-100 DATASET.  
I implemented two version of the model, the only difference is whether to insert the attention module before or after the corresponding max-pooling layer.

## (New!) Pre-trained models

[Google drive link](https://drive.google.com/open?id=1-s0rXWSSTZ23-o3KjKaxqaol8MnA8PvR)  

## Dependences  

* PyTorch (>=0.4.1)
* OpenCV
* [tensorboardX](https://github.com/lanpa/tensorboardX)  

**NOTE** If you are using PyTorch < 0.4.1, then replace *torch.nn.functional.interpolate* by *[torch.nn.Upsample](https://pytorch.org/docs/stable/nn.html#upsample)*. (Modify the code in utilities.py).  

## Training  
1. Pay attention before max-pooling layers  
```
python train.py --attn_mode before --outf logs_before --normalize_attn --log_images
```

2. Pay attention after max-pooling layers  
```
python train.py --attn_mode after --outf logs_after --normalize_attn --log_images
```

## Results  

### Training curve - loss  

The x-axis is # iter

1. Pay attention before max-pooling layers  
![](https://github.com/SaoYan/LearnToPayAttention/blob/master/fig/loss_attn_before.png)  

2. Pay attention after max-pooling layers  
![](https://github.com/SaoYan/LearnToPayAttention/blob/master/fig/loss_attn_after.png)  

3. Plot in one figure  
![](https://github.com/SaoYan/LearnToPayAttention/blob/master/fig/loss_compare.png)  

### Training curve - accuracy on test data  

The x-axis is # epoch  

1. Pay attention before max-pooling layers  
![](https://github.com/SaoYan/LearnToPayAttention/blob/master/fig/test_acc_attn_before.png)  

2. Pay attention after max-pooling layers  
![](https://github.com/SaoYan/LearnToPayAttention/blob/master/fig/test_acc_attn_after.png)  

3. Plot in one figure  
![](https://github.com/SaoYan/LearnToPayAttention/blob/master/fig/test_acc_compare.png)  

### Quantitative results (on test data of CIFAR-100)  

|    Method   | VGG (Simonyan&Zisserman,2014) | (VGG-att3)-concat-pc (ICLR 2018) | attn-before-pooling (my code) | attn-after-pooling (my code) |
|:-----------:|:-----------------------------:|:--------------------------------:|:-----------------------------:|:----------------------------:|
| Top-1 error |             30.62             |               22.97              |             22.62             |             22.92            |

### Attention map visualization (on test data of CIFAR-100)  

From left to right: L1, L2, L3, original images

1. Pay attention before max-pooling layers  
![](https://github.com/SaoYan/LearnToPayAttention/blob/master/fig/attn_map_before.png)

2. Pay attention after max-pooling layers  
![](https://github.com/SaoYan/LearnToPayAttention/blob/master/fig/attn_map_after.png)
