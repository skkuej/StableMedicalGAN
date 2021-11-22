
# For model to synthesize contrast enhanced breast images
# key words : MRI, T1 contrast-enhanced, synthesized, Generative-Adversarial Network, Self-attention, Residual Network

<h1 align="center">
    <p> Stable Medical GAN </p>
</h1>

## Objective
We aimed to **synthesize contrast-enhanced T1-weighted MRI (ceT1) image from pre-contrast T1-weighted MRI (preT1)**. We develped a generative adversarial network (GAN) to synthesize the ceT1 image better in particular to tumor region of interests (ROI). It consisted of 1 generator, 2 discriminator and 1 segmentation network and was performed to synthesize more authentic ceT1 images.

## Data Description
This experiment has been conducted by using Breast dynamic contrast T1-weighted MRI (DCE MRI) supported by Samsung Medical Centor and Gil Hospital. DCE MRI have several series images, especially we used pre-contrast image (preT1) before the contrast agenet injection and contrast-enhanced image (ceT1) performing at 90 sec.

## Code Description
All the things about TSGAN are coded by Python 3.2 and Tensorflow 2.3v. We trained the Pix2Pix network as the baseline GAN model using U-net as the generator and patch GAN as the discriminator for comparison. We also trained other well-established networks, i.e., the residual GAN (ResGAN), edge-aware GAN (EaGAN), and CycleGAN, as the other baseline methods. ResGAN and EaGAN were implemented with a residual module and an edge map. CycleGAN, consisting of two GANs, was trained to use the cycle and identity losses with paired images. All models were implemented on the TensorFlow framework using four GeForce GTX 1080 GPUs.

## Executation

First, open Load.ipynb included Evaluation, Load_data, Load_model files.

Second, Change the path for data of Load_data.

Third, Choose one of them(Tumor ROI version, Multi-view Tumor ROI version) and Excute load.ipynb

Then, can get the images of results synthesized and NRMSE, PSNR, SSIM(manual) and SSIM2(tf.image.ssim).


## Lincense
/*******************************************************
Copyright (C) 2020-2021 Eunjin Kim dmswlskim970606@gmail.com This file is part of Stable Medical GAN (SMGAN). TSGAN Project can not be copied and/or distributed without the express permission of Eunjin Kim.
*******************************************************/

/*******************************************************

 * Copyright (C) 2020 Eunjin Kim <dmswlskim970606@gmail.com>
 * 
 * This file is part of T1ce synthesized using GAN Project.
 * 
 * This project and code can not be copied and/or distributed without the express
 * permission of EJK, skkuej.

 *******************************************************/
 
