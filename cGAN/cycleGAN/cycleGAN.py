import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt
from zmq import device

def reverse_normalize(image, mean_=0.5, std_=0.5):
    if torch.is_tensor(image):
        image = image.detach().numpy()
    un_normalized_img = image * std_ + mean_
    un_normalized_img = un_normalized_img * 255
    return np.uint8(un_normalized_img)

def show_test(fixed_Y, fixed_X, G_YtoX, G_XtoY, mean_=0.5, std_=0.5):
    """
    Shows results of generates based on test image input. 
    """
    #Identify correct device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #Create fake pictures for both cycles
    fake_X = G_YtoX(fixed_Y.to(device))
    fake_Y = G_XtoY(fixed_X.to(device))
    
    #Generate grids
    grid_x =  make_grid(fixed_X, nrow=4).permute(1, 2, 0).detach().cpu().numpy()
    grid_y =  make_grid(fixed_Y, nrow=4).permute(1, 2, 0).detach().cpu().numpy()
    grid_fake_x =  make_grid(fake_X, nrow=4).permute(1, 2, 0).detach().cpu().numpy()
    grid_fake_y =  make_grid(fake_Y, nrow=4).permute(1, 2, 0).detach().cpu().numpy()
    
    #Normalize pictures to pixel range rom 0 to 255
    X, fake_X = reverse_normalize(grid_x, mean_, std_), reverse_normalize(grid_fake_x, mean_, std_)
    Y, fake_Y = reverse_normalize(grid_y, mean_, std_), reverse_normalize(grid_fake_y, mean_, std_)
    
    #Transformation from X -> Y
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(20, 10))
    ax1.imshow(X)
    ax1.axis('off')
    ax1.set_title('X')
    ax2.imshow(fake_Y)
    ax2.axis('off')
    ax2.set_title('Fake Y  (Monet-esque)')
    plt.show()

class ImageDataset(Dataset):
    def __init__(self, img_path, img_size=256):
        self.img_path = img_path
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.img_idx = dict()
        for number_, img_ in enumerate(os.listdir(self.img_path)):
            self.img_idx[number_] = img_
                
    def __len__(self):
        return len(self.img_idx)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.img_idx[idx])
        img = Image.open(img_path)
        img = self.transform(img)
            
        return img
    
# Configure hyperparameters
batch_size=8
batch_size_test=4

""" mount colab and google drive 
    fix the data path 
    
    from google.colab import drive
    drive.mount('/content/drive') """
    
monet_dir = '/content/drive/MyDrive/dataset/dataset/gan-getting-started/monet_jpg'
photo_dir = '/content/drive/MyDrive/dataset/dataset/gan-getting-started/photo_jpg'
photo_ds = ImageDataset(photo_dir, img_size=256)
monet_ds = ImageDataset(monet_dir, img_size=256)
    
photo_dl = DataLoader(photo_ds, batch_size=batch_size, shuffle=True,
num_workers=0, pin_memory=True)
monet_dl = DataLoader(monet_ds, batch_size=batch_size, shuffle=True,
num_workers=0, pin_memory=True)
    
photo_dl_test = DataLoader(photo_ds, batch_size=batch_size_test, shuffle=False,
num_workers=0, pin_memory=True)
monet_dl_test = DataLoader(monet_ds, batch_size=batch_size_test, shuffle=False,
num_workers=0, pin_memory=True)

class FeatureMapBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3, use_bn=True):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
        padding=padding, padding_mode='reflect')
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(out_channels)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class ContractingBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, use_bn=True, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.contl = nn.Conv2d(in_channels, in_channels*2, kernel_size=kernel_size,
        stride=2, padding=1, padding_mode='reflect')
        if use_bn:
            self.contl_instance = nn.InstanceNorm2d(in_channels*2)
        self.use_bn = use_bn
        self.contl_activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU()

    def forward(self, x):
        x = self.contl(x)
        if self.use_bn:
            x = self.contl_instance(x)
        x = self.contl_activation(x)
        return x

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.expnl = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=kernel_size,
        stride=2, padding=1, output_padding=1)
        if use_bn:
            self.expnl_instance = nn.InstanceNorm2d(in_channels//2)
        self.use_bn = use_bn
        self.expnl_activation = nn.ReLU()

    def forward(self, x):
        x = self.expnl(x)
        if self.use_bn:
            x = self.expnl_instance(x)
        x = self.expnl_activation(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.resconv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
        padding=1, padding_mode='reflect')
        self.resconv_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
        padding=1, padding_mode='reflect')
        self.res_instance = nn.InstanceNorm2d(in_channels)
        self.res_activation = nn.ReLU()
    
    def forward(self, x):
        input_x = x.clone()
        x = self.resconv_1(x)
        x = self.res_instance(x)
        x = self.res_activation(x)

        x = self.resconv_2(x)
        x = self.res_instance(x)
        return input_x + x

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=64):
        super(Generator, self).__init__()
        self.upfeature = FeatureMapBlock(in_channels, mid_channels)
        self.contl_1 = ContractingBlock(mid_channels)
        self.contl_2 = ContractingBlock(mid_channels*2)
        res_mult = 4
        self.rsdul_0 = ResidualBlock(mid_channels*res_mult) # 128 -> 256
        self.rsdul_1 = ResidualBlock(mid_channels*res_mult)
        self.rsdul_2 = ResidualBlock(mid_channels*res_mult)
        self.rsdul_3 = ResidualBlock(mid_channels*res_mult)
        self.rsdul_4 = ResidualBlock(mid_channels*res_mult)
        self.rsdul_5 = ResidualBlock(mid_channels*res_mult)
        self.rsdul_6 = ResidualBlock(mid_channels*res_mult)
        self.rsdul_7 = ResidualBlock(mid_channels*res_mult)
        self.rsdul_8 = ResidualBlock(mid_channels*res_mult)
        self.expnl_1 = ExpandingBlock(mid_channels*4)
        self.expnl_2 = ExpandingBlock(mid_channels*2)
        self.downfeature = FeatureMapBlock(mid_channels, out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        upfeature_out = self.upfeature(x)
        contl_1_out = self.contl_1(upfeature_out)
        contl_2_out = self.contl_2(contl_1_out)
        rsdul_0_out = self.rsdul_0(contl_2_out)
        rsdul_1_out = self.rsdul_1(rsdul_0_out)
        rsdul_2_out = self.rsdul_2(rsdul_1_out)
        rsdul_3_out = self.rsdul_3(rsdul_2_out)
        rsdul_4_out = self.rsdul_4(rsdul_3_out)
        rsdul_5_out = self.rsdul_5(rsdul_4_out)
        rsdul_6_out = self.rsdul_6(rsdul_5_out)
        rsdul_7_out = self.rsdul_7(rsdul_6_out)
        rsdul_8_out = self.rsdul_8(rsdul_7_out)
        expnl_1_out = self.expnl_1(rsdul_8_out)
        expnl_2_out = self.expnl_2(expnl_1_out)
        downfeature_out = self.downfeature(expnl_2_out)
        return self.tanh(downfeature_out)

class Discriminator(nn.Module):
    def __init__(self, in_channels, mid_channels=64):
        super(Discriminator, self).__init__()
        self.disconv_1 = FeatureMapBlock(in_channels, mid_channels, kernel_size=4, stride=2, padding=1)
        self.disconv_2 = ContractingBlock(mid_channels, kernel_size=4, use_bn=False, activation='lrelu')
        self.disconv_3 = ContractingBlock(mid_channels*2, kernel_size=4, activation='lrelu')
        self.disconv_4 = ContractingBlock(mid_channels*4, kernel_size=4, activation='lrelu')
        #self.disconv_5 = ContractingBlock(mid_channels*8, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(mid_channels*8, 1, kernel_size=1)

    def forward(self, x):
        disconv_1_out = self.disconv_1(x)
        disconv_2_out = self.disconv_2(disconv_1_out)
        disconv_3_out = self.disconv_3(disconv_2_out)
        disconv_4_out = self.disconv_4(disconv_3_out)
        #disconv_5_out = self.disconv_5(disconv_4_out)
        final_out = self.final(disconv_4_out)
        return final_out
    
def init_weights_normal(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight'):
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

def build_model(dim_p=3, dim_m=3, device=device):
    G_PtoM = Generator(dim_p, dim_m).to(device)
    G_MtoP = Generator(dim_m, dim_p).to(device)
    
    D_P = Discriminator(dim_p).to(device)
    D_M = Discriminator(dim_m).to(device)

    G_PtoM.apply(init_weights_normal)
    G_MtoP.apply(init_weights_normal)
    D_P.apply(init_weights_normal)
    D_M.apply(init_weights_normal)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        G_PtoM.to(device)
        G_MtoP.to(device)
        D_P.to(device)
        D_M.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_PtoM, G_MtoP, D_P, D_M

# adv_criterion = nn.MSELoss()
# recon_criterion = nn.L1Loss()

def real_mse_loss(D_out, adversarial_weight):
    #mse_loss = adv_criterion(D_out, torch.ones_like(D_out))
    mse_loss = torch.mean((D_out-1)**2)*adversarial_weight
    return mse_loss

def fake_mse_loss(D_out, adversarial_weight):
    #mse_loss = adv_criterion(D_out, torch.zeros_like(D_out))
    mse_loss = torch.mean(D_out**2)*adversarial_weight
    return mse_loss

def cycle_consistency_loss(real_image, recon_image, lambda_weight=1):
    #forward cycle flow
    #fake_M = G_MtoP(real_image_m)
    #recon_P = G_MtoP(fake_M)
    #recon_loss = recon_criterion(real_image_p, recon_p)

    #backward cycle flow
    #fake_P = G_PtoM(real_image_p)
    #recon_M = G_PtoM(fake_P)
    #recon_loss = recon_criterion(real_image_m, recon_m)

    recon_loss = torch.mean(torch.abs(real_image-recon_image)*lambda_weight)
    return recon_loss

def identity_loss(real_image, gen_image, identity_weight=1):
    ident_loss = torch.mean(torch.abs(real_image-gen_image)*identity_weight)
    return ident_loss

def training_loop(dataloader_m, dataloader_p, test_dataloader_m, test_dataloader_p, 
    n_epochs=1000):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_lr = 0.0002
    beta1 = 0.5
    G_PtoM, G_MtoP, D_P, D_M = build_model(3, 3, device)

    losses = []
    adverserial_weight = 0.5
    lambda_weight = 10
    identity_weight = 5

    test_iter_M = iter(test_dataloader_m)
    test_iter_P = iter(test_dataloader_p)
    fixed_M = test_iter_M.next()
    fixed_P = test_iter_P.next()

    iter_M = iter(dataloader_m)
    iter_P = iter(dataloader_p)
    batches_per_epoch = min(len(iter_M), len(iter_P))

    d_total_loss_avg = 0.0
    g_total_loss_avg = 0.0

    for epoch in range(1, n_epochs+1):
        
        if epoch % batches_per_epoch == 0:
            iter_M = iter(dataloader_m)
            iter_P = iter(dataloader_p)

        images_M = iter_M.next()
        images_P = iter_P.next()

        images_M = images_M.to(device)
        images_P = images_P.to(device)

        ## Train the Discriminators
        ## First; D_P, real fake detection (loss)
        d_p_optimizer = optim.Adam(D_P.parameters(), base_lr, [beta1, 0.999])
        d_p_optimizer.zero_grad() # train with real images; photo
        
        # Discriminator loss for real images
        out_p = D_P(images_P)
        d_p_real_loss = real_mse_loss(out_p, adverserial_weight)

        # Generate fake images that look like domain P based on real images in domain M
        fake_p = G_MtoP(images_M)
        out_p = D_P(fake_p)
        d_p_fake_loss = fake_mse_loss(out_p, adverserial_weight)

        # Compute total loss and backprop
        d_p_loss = d_p_real_loss + d_p_fake_loss
        d_p_loss.backward()
        d_p_optimizer.step()

        ## Second; D_M, real fake loss
        d_m_optimizer = optim.Adam(D_M.parameters(), base_lr, [beta1, 0.999])
        d_m_optimizer.zero_grad()

        out_m = D_M(images_M)
        d_m_real_loss = real_mse_loss(out_m, adverserial_weight)
        fake_m = G_PtoM(images_P)
        out_m = D_M(fake_m)
        d_m_fake_loss = fake_mse_loss(out_m, adverserial_weight)

        d_m_loss = d_m_real_loss + d_m_fake_loss
        d_m_loss.backward()
        d_m_optimizer.step()

        d_total_loss = d_p_real_loss + d_p_fake_loss + d_m_real_loss + d_m_fake_loss

        ## Train the Generators
        ## First; fake image monet, reconstructed photo
        g_params = list(G_PtoM.parameters()) + list(G_MtoP.parameters())
        g_optimizer = optim.Adam(g_params, base_lr, [beta1, 0.999])
        g_optimizer.zero_grad()

        fake_m = G_PtoM(images_P)
        out_m = D_M(fake_m)
        g_PtoM_loss = real_mse_loss(out_m, adverserial_weight)

        recon_p = G_MtoP(fake_m)
        forward_cycle_loss = cycle_consistency_loss(images_P, recon_p, lambda_weight)
        forward_ident_loss = identity_loss(images_P, fake_m, identity_weight)

        ## Second; fake image photo, reconstructed monet
        fake_p = G_MtoP(images_M)
        out_p = D_P(fake_p)
        g_MtoP_loss = real_mse_loss(out_p, adverserial_weight)
        
        recon_m = G_PtoM(fake_p)
        backward_cycle_loss = cycle_consistency_loss(images_M, recon_m, lambda_weight)
        backward_ident_loss = identity_loss(images_M, fake_p, identity_weight)

        g_total_loss = g_PtoM_loss + g_MtoP_loss + forward_cycle_loss + backward_cycle_loss +\
        forward_ident_loss + backward_ident_loss
        g_total_loss.backward()
        g_optimizer.step()

        d_total_loss_avg = d_total_loss_avg + d_total_loss / batches_per_epoch
        g_total_loss_avg = g_total_loss_avg + g_total_loss / batches_per_epoch
        
        # Print log info
        print_every = batches_per_epoch
        if epoch % print_every == 0:
            # Append real and fake discriminator losses and the generator loss
            losses.append((d_total_loss_avg.item(), g_total_loss_avg.item()))
            true_epoch_n = int(epoch/batches_per_epoch)
            true_epoch_total = int(n_epochs/batches_per_epoch)
            print('Epoch [{:5d}/{:5d}] | d_total_loss_avg: {:6.4f} | g_total_loss: {:6.4f}'.format(
                    true_epoch_n, true_epoch_total, d_total_loss_avg.item(), g_total_loss_avg.item()))
        
        # Show the generated samples
        show_every = (batches_per_epoch*10)
        if epoch % show_every == 0:
            # set generators to eval mode for image generation
            G_MtoP.eval()
            G_PtoM.eval()
            test_images = show_test(fixed_P, fixed_M, G_MtoP, G_PtoM)
            # set generators to train mode to continue training
            G_MtoP.train()
            G_PtoM.train()
    
        # Reset average loss for each epoch
        if epoch % batches_per_epoch == 0:
            d_total_loss_avg = 0.0
            g_total_loss_avg = 0.0
    
    return losses

batches_per_epoch = min(len(photo_dl), len(monet_dl))
epoch_true = 25

n_epochs = epoch_true * batches_per_epoch
losses = training_loop(photo_dl, monet_dl, photo_dl_test, monet_dl_test, n_epochs=n_epochs)