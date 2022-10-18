DEMO
===============

###########Requirements
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

###########File tree
All the files are as listed
├─GAN                    // Train the DT-CycleGAN, CyclGAN, RetinaGAN
│  ├─Data                // Data for GAN training 
│  │  └─train
│  │      ├─real_block
│  │      ├─real_blue
│  │      ├─real_red
│  │      └─simu_pure_1k
│  └─output             // Output trained GAN model
│      ├─CycleGAN
│      ├─DT-CycleGAN
│      │  ├─blocks-complex
│      │  ├─blocks-pure
│      │  ├─blue
│      │  └─red
│      └─RetinaGAN
└─Robot-FTC       // Train the Model on simulation enviroment
    ├─checkpoints
    ├─Env
    │  └─__pycache__
    ├─ftc_robot     // the robots model for pybullet
    │  ├─config
    │  ├─launch
    │  ├─meshes
    │  └─urdf
    ├─Mesh           // background mesh figures
    │  └─swirly
    ├─objects         // grisp target
    ├─runs             // record while training
    │  ├─Object_detect_Mesh_resnet26d
    │  ├─Object_detect_Mesh_swin_tiny
    │  ├─Object_detect_Mesh_vit_tiny
    │  ├─Object_detect_pure_swin-ti
    │  ├─Object_detect_pure_swin-ti-cycleagan
    │  ├─Object_detect_pure_swin-ti-cyclegan
    │  ├─Object_detect_pure_swin-ti-retinagan
    │  ├─Object_detect_pure_swin-ti_10k
    │  ├─Object_detect_pure_swin-ti_12k
    │  ├─Object_detect_pure_swin-ti_14k
    │  ├─Object_detect_pure_swin-ti_16k
    │  ├─Object_detect_pure_swin-ti_18k
    │  ├─Object_detect_pure_swin-ti_2000
    │  ├─Object_detect_pure_swin-ti_20k
    │  └─Object_detect_pure_swin-ti_4000
    └─__pycache__

###########Usage
- ./Robot-FTC files contains the codes and model files of robots trained on the simulation enviroments. More details see ./Robot-FTC/README.md
- ./GAN files contains the codes and data of our the DT-CycleGAN methods, also with RetinaGAN and CycleGAN. MOre details see ./GAN/README.md

###########Contact

Yuzhong Chen - chenyuzhong211@gmail.com



