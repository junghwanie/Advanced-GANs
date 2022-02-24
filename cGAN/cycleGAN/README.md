## Style transfer with Cycle GAN
The code implements the paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (ICCV 2017/ Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros)](https://arxiv.org/abs/1703.10593).

So far, image-to-image translation has been trained only on datasets with pairs. In practive, however, paired training data does not exist for many tasks.
In this papaer, the mapping network G:X->Y and inverse network F:Y->X were trained to enable image-to-image translation for unpaired training data.

The objective of CycleGAN consists of adversarial loss (to make image realistic) and cycle-consistency loss (to preserve the content of the input image).

**Adversarial Losses**

<img width="372" alt="advloss" src="https://user-images.githubusercontent.com/37526521/155486892-f7af2ae1-12c4-483f-9de9-645ec5f9a401.png">
A similar adversarial loss is applied to mapping network F.


**Cycle Consistency Loss**

<img width="298" alt="cycleloss" src="https://user-images.githubusercontent.com/37526521/155486909-7c84efa4-f753-4864-a2f0-06f65f9b012f.png">

## Dependencies
+ Python 3.8.12+
+ PyTorch 1.10.2+

## Dataset
Description of dataset: I'm Something of a Painter Myself in kaggle competition.  
Monet painting and photo dataset are trained to generate monet-style photos.

You can download the kaggle dataset link below.  
[I'm Something of a Painter Myself: Use GANs to create art - will you be the next Monet?](https://www.kaggle.com/c/gan-getting-started/data)
