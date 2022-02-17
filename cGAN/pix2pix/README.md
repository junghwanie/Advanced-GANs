## Pix2Pix with facades
This code implements the paper: [Image-to-Image Translation with Conditional Adversarial Networks (CVPR 2017 / Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros)](https://arxiv.org/abs/1611.07004).

## Description
### Objective
1. **cGAN Loss**: As an object function, conditional GAN (cGAN) was used to train whether the input image and output image match well.
<img width="492" alt="pix2pix" src="https://user-images.githubusercontent.com/37526521/154443733-5b9dfc71-8f4b-472d-a739-479c00b789fd.png">

2. **L1 loss**: pix2pix adds an additional loss term. Using cGAN loss, Generator is trained to fool Discriminator well, but since Generator only serves to fool Discriminator, the ground truth output and the generated image may be different.
<img width="498" alt="pix2pix2" src="https://user-images.githubusercontent.com/37526521/154443825-f50b653f-86ed-4d07-abf4-bd0f0675de9f.png">
### Architecture
