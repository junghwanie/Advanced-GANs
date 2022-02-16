## DCGAN with CelebA
This practice was given as first assignment in the lecture of Advanced GANs.  
The code implements the paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (2015 / Alec Radford, Luke Metz, Soumith Chintala)](https://arxiv.org/abs/1511.06434).

After every 100 training iterations, the files ``real_samples.png`` and ``fake_samples.png`` are written to disk with the samples from the generative model.  
After every epoch, models are saved to: ``netG_epoch_%d.pth`` and ``netD_epoch_%d.pth``

## Dependencies
+ Python 3.8.12+
+ PyTorch 1.10.2+

## Dataset
You can download the CelebA dataset by entering [the Large-scale CelebFaces Attributes (CelebA) Database website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Training networks
To train DCGAN on CelebA, run the simple training script below. If you change other arguments, 
you should also add specific arguments in reference to usage.
<pre><code>python dcgan.py --dataset celeba --dataroot path-to-dataset</code></pre>

## Usage
<pre><code>usage: dcgan.py [-h] --dataset DATASET [--dataroot DATAROOT] [--workers WORKERS] 
                [--n_epochs N_EPOCHS] [--batchSize BATCHSIZE] [--niter NITER] [--ndf NDF] 
                [--ngf NGF] [--nz NZ] [--lr LR] [--beta1 BETA1] [--n_cpu N_CPU] [--cuda] 
                [--ngpu NGPU] [--outf OUTF] [--dry-run] [--manualSeed SEED] 
                [--classes CLASSES] [--imageSize IMAGESIZE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     celeba
  --dataroot DATAROOT   path to dataset
  --workers WORKERS     number of data loading workers
  --n_epochs N_EPOCHS   number of epochs of training, default=25
  --batchSize BATCHSIZE size of the batches, default=64
  --niter NITER         number of epochs to train for
  --ndf NDF             number of features to be used in Discriminator network, default=64
  --ngf NGF             number of features to be used in Generator network, default=64
  --nz NZ               Size of the noise, default=100
  --lr LR               adam: learning rate, default=0.0002
  --beta1 BETA1         adam: decay of first order momentum of gradient, default=0.5
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --cuda                enables cuda
  --ngpu NGPU           number of GPUs to use
  --outf OUTF           folder to output images and model checkpoints
  --dry-run             check a single training cycle works
  --manualSeed SEED     manual seed
  --classes CLASSES     number of classes for dataset
  --imageSize IMAGESIZE size of each image dimension</code></pre>
