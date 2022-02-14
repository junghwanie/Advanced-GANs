""" 
    This code is based on SNU-Advanced-GANs-master coursework
    GAN Basic: DCGAN training with celeba
    Refer to https://github.com/pytorch/examples/blob/master/dcgan/main.py
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

from __future__ import print_function
import argparse
from logging import root
import os
from pyexpat import model
import sys
import numpy as np
import random

import torchvision.transforms as transforms
import torchvision.datasets as dset
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable
import torchvision.utils as vutils

import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch

from dataset import DataSetFromDir
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='celeba | cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--n_epochs', type=int, default=25, help='# of epochs of training')
parser.add_argument('--batchSize', type=int, default=64, help='size of the batches')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--ndf', type=int, default=64, help='Number of features to be used in Discriminator network')
parser.add_argument('--ngf', type=int, default=64, help='Number of features to be used in Generator network')
parser.add_argument('--nz', type=int, default=100, help='Size of the noise')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--imageSize', type=int, default=64, help='size of each image dimension')
parser.add_argument('-f')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Configure data loader
if opt.dataset == 'celeba':
    dataset = DataSetFromDir(opt.dataroot, transform=transforms.Compose([
                          transforms.Resize(opt.imageSize),
                          transforms.CenterCrop(opt.imageSize),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]))
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), pin_memory=True)

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # input is Z, going into a convolution
        self.convt1 = nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0)
        self.convt1_bn = nn.BatchNorm2d(ngf*8)
        # state size. (ngf*8) x 4 x 4
        self.convt2 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1)
        self.convt2_bn = nn.BatchNorm2d(ngf*4)
        # state size. (ngf*4) x 8 x 8
        self.convt3 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1)
        self.convt3_bn = nn.BatchNorm2d(ngf*2)
        # state size. (ngf*2) x 16 x 16
        self.convt4 = nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1)
        self.convt4_bn = nn.BatchNorm2d(ngf)
        # state size. (ngf) x 32 x 32
        self.convt5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1)
        # state size. (nc) x 64 64
                    
    def forward(self, input):
        x = F.relu(self.convt1_bn(self.convt1(input)))
        x = F.relu(self.convt2_bn(self.convt2(x)))
        x = F.relu(self.convt3_bn(self.convt3(x)))
        x = F.relu(self.convt4_bn(self.convt4(x)))
        x = F.tanh(self.convt5(x))
        
        return x

G = Generator(ngpu).to(device)
G.apply(weights_init)
print(G)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # input is (nc) x 64 x 64
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        # state size. (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(ndf*2)
        # state size. (ndf) x 16 x 16
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(ndf*4)
        # state size. (ndf) x 8 x 8
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(ndf*8)
        # state size. (ndf) x 4 x 4
        self.conv5 = nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False)
        
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2) # ndfx32x32
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2) # ndf*2x16x16
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2) # ndf*4x8x8
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))
        
        return x

D = Discriminator(ngpu).to(device)
D.apply(weights_init)
print(D)

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Lists to keep track of progress
img_list = []
G_losses = []
G_grads_mean = []
G_grads_std = []
D_losses = []

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0
saturating = False

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)

for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        # Update D network
        # Train with all-real batch
        D.zero_grad()
        real_img = imgs[0].to(device)
        
        bs = real_img.size(0)
        valid = torch.full((bs,), real_label, dtype=real_img.dtype, device=device) # label
        output = D(real_img).view(-1)
        d_real_loss = criterion(output, valid)
        d_real_loss.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        noise = torch.randn(bs, nz, 1, 1, device=device)
        fake = G(noise)
        valid.fill_(fake_label)
        output = D(fake.detach()).view(-1)
        d_fake_loss = criterion(output, valid)
        d_fake_loss.backward()
        D_G_z1 = output.mean().item()
        
        d_loss = d_real_loss + d_fake_loss
        optimizerD.step()
        
        # Update G network
        if saturating:
            valid.fill_(fake_label)
        else:
            valid.fill_(real_label)
            
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = D(fake).view(-1)
        
        if saturating:
            g_loss = -criterion(output, valid)
        else:
            g_loss = criterion(output, valid)
        
        g_loss.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Save gradients
        G_grad = [p.grad.view(-1).cpu().numpy() for p in list(G.parameters())]
        G_grads_mean.append(np.concatenate(G_grad).mean())
        G_grads_std.append(np.concatenate(G_grad).std())

        # Output training stats
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))
        
        with torch.no_grad():
            if i % 100 == 0:
                vutils.save_image(real_img,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                fake = G(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
                
        if opt.dry_run:
            break
                
    # do checkpointing
    torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(D.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    

    
