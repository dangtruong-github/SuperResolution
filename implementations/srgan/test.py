from PIL import Image


DATA_PATH = "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba/000011.jpg"

im = Image.open(DATA_PATH)

print(im.width)
print(im.height)

"""
python srgan.py --checkpoint_interval 1 --batch_size 64 --hr_width 178 --hr_height 218
"""