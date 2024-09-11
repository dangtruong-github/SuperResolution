# SRGAN On Custom Dataset
Learn how to train SRGAN on Custom dataset

## Environment Setup
Open anaconda prompt and cd to the folder where you have your environment.yml file

conda env create -f environment.yml

conda activate srganenv_gpu 


Now Install Pytorch as per your resources

#### GPU
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

##### CPU Only
conda install pytorch-cpu==1.1.0 torchvision-cpu==0.3.0 cpuonly -c pytorch


#### Train your Model:
!python main.py --LR_path your/lr/path --GT_path your/hr/path

#### Test your Model:
!python main.py --mode test_only --LR_path your/lr/path --generator_path your/generator/path.pt


#### If you want to test the model which is trained on div2K dataset then download the model from here:  https://github.com/AarohiSingla/SRGAN_CustomDataset/tree/main/pretrained_models


