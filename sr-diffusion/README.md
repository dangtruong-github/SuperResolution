# SRDiff on CelebA 
Learn how to train SRGAN on Custom dataset

## Environment Setup
pip install <module> 

torch>=2.0.0
torchvision>=0.15.0
lightning>=2.0.0
torchmetrics>=0.11.4
hydra-core==1.3.2
hydra-colorlog==1.2.0
hydra-optuna-sweeper==1.2.0
rootutils       
pre-commit     
rich           
pytest          


## Config 
Config parameter of model

You can adjust file /config/train.yaml 
In train.yaml, you have modify some path to folderlike that: 

model: the config of model (default: srdiffution_module0)
data: the config of dataset (default: celeba)
logger: the config of logger (default: wandb) 
trainer: the config of trainer (default: default) 


## Train your Model:

After configurizing the config models, you only do: 

python train.py 

and enjoy your moment 

