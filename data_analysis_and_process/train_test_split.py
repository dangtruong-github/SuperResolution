from sklearn.model_selection import train_test_split
import pandas as pd
import os
import shutil

PATH_FEATURES = "../../list_attr_celeba.csv"

df = pd.read_csv(PATH_FEATURES)

print(df)

# Split the original DataFrame into training and testing sets
train, eval = train_test_split(df, test_size=0.1, random_state=42)

train.to_csv("../../data/train_img.csv", index=False)
eval.to_csv("../../data/val_img.csv", index=False)

os.makedirs("../../data/img_align_celeba/train", exist_ok=True)
os.makedirs("../../data/img_align_celeba/val", exist_ok=True)

for index, img_path in enumerate(train["image_id"]):
    print(img_path)
    src_path = "../../data/img_align_celeba/" + img_path
    dst_path = "../../data/img_align_celeba/train/" + img_path
    print(src_path)
    print(dst_path)
    shutil.move(src_path, dst_path)

    if (index+1) % 50 == 0:
        print(img_path, "Success")

for index, img_path in enumerate(eval["image_id"]):
    print(img_path)
    src_path = "../../data/img_align_celeba/" + img_path
    dst_path = "../../data/img_align_celeba/val/" + img_path
    print(src_path)
    print(dst_path)
    shutil.move(src_path, dst_path)

    if (index+1) % 50 == 0:
        print(img_path, "Success")