# Pytorch-CycleGAN

## 1.Data

Inclued simulation Images （with label） and  real Imges (without label). All images are saved in .npy with two images (local and global).



## 2. Train 

1. Train DT-CycleGAN

   `python train_RetinaGAN.py`

2. Train CycleGAN

   `python train_cycleGAN.py`

3. Train RetinaGAN

   `python train_RetinaGAN.py`

## 3. Test

the **test.py** will produce the images generate by generator in GAN.

## 4. GAN Data Augment

For the methods of CycleGAN and RetinaGAN, they play roles as a data augment methods. Therefor, we need generate the augment images and train with grisper model. The **data_gan.py** will generate the augment data in a replaybuffer type. (The training of the model see *../Robot-FTC/train_post.py*)

## 4.Output

The output model of different methods in different setting.
