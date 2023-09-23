import os
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm

train_folder_path = "./train/"
val_folder_path = "./val/"

# csv file path
train_csv_file_path = os.path.join(train_folder_path, 'annotations.csv')
val_csv_file_path = os.path.join(val_folder_path, 'annotations.csv')

train_annotations_df = pd.read_csv(train_csv_file_path)
val_annotations_df = pd.read_csv(val_csv_file_path)

def resize_and_scale_bbox(img, bbox, target_size) :
    img_width, img_height = img.size
    # print(img_width, img_height)

    img = img.resize(target_size, Image.LANCZOS)
    resize_img_width, resize_img_height = img.size

    # bounding box scale
    x, y , width, height = bbox
    x_scale = target_size[0] / img_width
    y_scale = target_size[1] / img_height

    x_center = (x + width / 2) * x_scale
    y_center = (y + height / 2) * y_scale
    scaled_width = width * x_scale
    scaled_height = height * y_scale

    scaled_bbox = (x_center, y_center, scaled_width, scaled_height)

    return img, scaled_bbox


# convert to yolo format
def convert_to_yolo_format(annotation_df, org_image_folder, output_folder, target_size) :
    for idx , row in tqdm(annotation_df.iterrows()) : # .iterrows() 데이터의 행-열 데이터 정보를 튜플 형태의 generator 객체로 변환
        image_name = row['filename']
        label = row['region_id']
        # print(image_name, label)

        img_path = os.path.join(org_image_folder, image_name)
        # print("img path >> " , img_path)
        new_img_path = os.path.join(output_folder, "images", image_name)
        # print("new img path >> " , new_img_path)

        # box info
        shape_attributes = json.loads(row['region_shape_attributes'])
        print("shape_attributes" , shape_attributes)
        x = shape_attributes['x']
        y = shape_attributes['y']
        width = shape_attributes['width']
        height = shape_attributes['height']
        print(x, y , width, height)

        # img read
        img = Image.open(img_path)

        # img resize and bounding box scale
        resize_and_scale_bbox(img,(x,y, width, height), target_size)


train_yolo_folder = "./yolo_dataset/train/"
val_yolo_folder = "./yolo_dataset/val/"

os.makedirs(os.path.join(train_yolo_folder, "images"), exist_ok=True)
os.makedirs(os.path.join(train_yolo_folder, "labels"), exist_ok=True)
os.makedirs(os.path.join(val_yolo_folder, "images"), exist_ok=True)
os.makedirs(os.path.join(val_yolo_folder, "labels"), exist_ok=True)

target_size = (1280, 720)
"""
yolo_dataset 
  train
    images
        aaa.png 
    labels
        aaa.txt
  val
    images
        bbb.png
    labels
        bbb.txt
"""

convert_to_yolo_format(train_annotations_df,train_folder_path, train_yolo_folder, target_size)