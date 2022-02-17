## Conditional GAN
### Basic conditional GAN
<img width="770" alt="cGAN" src="https://user-images.githubusercontent.com/37526521/154440178-0bf1c125-16c4-4d46-a1a0-81b1f5d43c09.png">

+ cGAN ([Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)): was implemented by ``python cGAN.py``.   
  This network illustrates that conditional information (classes, labels, images, text descriptions) was added to the existing GANs.
  
+ ACGAN: Conditional Image Synthesis With Auxiliary Classifier GANs

### Supervised Approach
+ Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks

### Unsupervised Approach
+ CycleGAN: Unparied Image-to-Image Translation using Cycle-Consistent Adversarial Networks

## Dependencies
+ Python 3.8.12+
+ PyTorch 1.10.2+

## Usage
<pre><code>usage: cgan.py [-h] [--workers WORKERS] [--batchSize BATCHSIZE] [--imageSize IMAGESIZE] [--nz NZ]
               [--niter NITER] [--lr LR] [--beta1 BETA1] [--cuda] [--ngpu NGPU] [--netG NETG] [--netD NETD]
               [--outf OUTF] [--manualSeed SEED] [-f F]

optional arguments:
  -h, --help            show this help message and exit
  --workers WORKERS     number of data loading workers
  --batchSize BATCHSIZE input batch size
  --imageSize IMAGESIZE the height / width of the input image to network
  --nz NZ               size of the latent z vector
  --niter NITER         number of epochs to train for
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         beta1 for adam. default=0.5
  --cuda                enables cuda
  --ngpu NGPU           number of GPUs to use
  --netG NETG           path to netG (to continue training)
  --netD NETD           path to netD (to continue training)
  --outf OUTF           folder to output images and model checkpoints
  --manualSeed SEED     manual seed</code></pre>
  
  ## Citation
  The lecture of Advanced-GANs in Seoul National University by Naver AI lab.
