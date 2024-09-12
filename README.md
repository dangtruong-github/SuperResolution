# Super Resolution Project

This repository contains the implementation of various super-resolution (SR) techniques, including CNN-based methods, GAN-based models, and diffusion-based approaches. Below are the details of the main components:

## Members
- Chu Hữu Đăng Trường
- Nguyễn Kim Hoàng Anh
- Nguyễn Đức Anh
- Nguyễn Tiến Hùng
- Đặng Ngọc Minh

## Components

1. **sr-cnn.ipynb**  
   This Jupyter notebook contains the implementation of a basic super-resolution CNN (SRCNN) model. It demonstrates the training process and evaluates the performance of the model.

2. **srgan-orig**  
   This directory contains the implementation of the SRGAN model. The model is trained from scratch, starting with random weights, and learns to generate high-resolution images from low-resolution inputs.

3. **srgan-finetune**  
   This directory holds the code for fine-tuning a pre-trained SRGAN model. It leverages transfer learning to improve results on specific datasets by continuing the training process.

4. **sr-diffusion**  
   The SR diffusion model uses a probabilistic approach to generate high-resolution images through iterative refinement. This method differs from traditional CNN and GAN approaches by leveraging a diffusion process to generate realistic images.

5. **data_analysis_and_process**

   Contains code for train data analysis and train, test split.

## Evaluation

The following table summarizes the evaluation metrics for each model:

| Model           | PSNR  | SSIM_LR | SSIM  | LPIPS |
|-----------------|-------|---------|-------|-------|
| sr-cnn          | 28.75 | 0.79    | 0.73  | 0.12  |
| srgan-orig      | 22.09 | 0.75    | 0.71  | 0.08  |
| srgan-finetune  | 30.87 | 0.93    | 0.85  | 0.21  |
| sr-diffusion    | 31.49 | -       | 0.88  | -     |

Each model is evaluated using four key metrics:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **SSIM_LR** (SSIM for low-resolution input)
- **LPIPS** (Learned Perceptual Image Patch Similarity)
