"""
python main.py --LR_path /N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba/train/lr --GT_path /N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba/train/hr --fine_tuning True --generator_path ./pretrained_models/SRGAN.pt
python main.py --LR_path /N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba/val/lr --GT_path /N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba/val/hr --fine_tuning True --generator_path ./model/pre_trained_model_1600.pt
python main.py --mode test --LR_path /N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba/val/lr --GT_path /N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba/val/hr --generator_path ./model/SRGAN_gene_1500.pt

python main.py --mode test --LR_path /N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/celeba_hq_256/lr --GT_path /N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/celeba_hq_256/hr --generator_path ./model/SRGAN_gene_1500.pt

python main.py --mode test --LR_path /N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/celeba_hq_256/lr --GT_path /N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/celeba_hq_256/hr --generator_path ./model/sr-cnn.pt
"""