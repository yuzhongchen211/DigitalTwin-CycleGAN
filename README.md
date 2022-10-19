# Digital Twin (DT)-CycleGAN

Digital Twin (DT)-CycleGAN: Enabling Zero-Shot Sim-to-Real Transfer of Visual Grasping Models

## Requirements
The usage of python package for this project
numpy==1.21.6
pybullet==3.2.5
timm==0.4.12
torch==1.9.1
torchvision==0.10.1
tqdm==4.62.3
opencv_python_headless==4.5.5.64
Pillow==9.2.0
visdom==0.1.8.9

## File tree
```
All the files are as listed
├─GAN                    // Train the DT-CycleGAN, CyclGAN, RetinaGAN
└─Robot-FTC       // Train the Model on simulation enviroment
    ├─checkpoints
    ├─Env
    ├─ftc_robot     // the robots model for pybullet
    │  ├─config
    │  ├─launch
    │  ├─meshes
    │  └─urdf
    ├─Mesh           // background mesh figures
    │  └─swirly
    ├─objects         // grisp target
```

## Usage
- ./Robot-FTC files contains the codes and model files of robots trained on the simulation enviroments. More details see ./Robot-FTC/README.md
- ./GAN files contains the codes and data of our the DT-CycleGAN methods, also with RetinaGAN and CycleGAN. MOre details see ./GAN/README.md

## Contact

Yuzhong Chen - chenyuzhong211@gmail.com



