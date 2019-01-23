# FakeFace

# Introduction

This repository holds the [PyTorch](http://pytorch.org)-based code for training a fake (face) image classifier.

We used this code to participate in the [2018 AI RnD Challenge](http://airndchallenge.com), hosted by the [Ministry of Science and ICT](http://msit.go.kr), Government of South Korea. Our team(Lomin) won the 2nd place of the challenge.

In this repository, we provide

- Training/Test code
- Our 'Photoshop dataset', which we specially made for this task
- Demo website for test

# Dependencies

- Python packages (We recommend using Conda environment)
    - tensorflow-gpu==1.12.0
    - torch==1.0.0
    - torchvision==0.2.1
    - Pillow==5.2.0
    - matplotlib==2.2.2
    - scipy==1.1.0
    - tqdm==4.24.0
    - opencv-python==3.4.3.18
    - pyyaml==4.2b1
    - addict==2.2.0
    - tensorboardX==1.4
    - mtcnn==0.0.8
- An NVIDIA GPU & CUDA 9.0

# Quick start

- To test our models, visit our demo website: [https://faceforensic.lomin.ai](https://faceforensic.lomin.ai) (Korean)
- If you want to manually test and get scores for each images, follow the steps:
    1. Place your images at any directory.
    2. Download our models from link.
    3. Run the script "test.py". Use at least one of —inference or —avg option.

        python test.py
        	--root_dset  (root directory for your test images)
        	--root_model (root directory where you downloaded models)
        	--batch_size (batch size for inference)
        	--inference  (if you want to run inference)
        	--avg        (if you want to ensemble results)

# Contact

Bee Lim (b.lim@lomin.ai)
