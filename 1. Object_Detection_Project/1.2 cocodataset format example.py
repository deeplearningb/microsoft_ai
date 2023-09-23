import os
import matplotlib.pyplot as plt
import random

from pycocotools.coco import COCO
from PIL import Image

# 경로설정
annfile_path = "./car_damage_dataset/train/COCO_train_annos.json"
mul_annfile_path = "./car_damage_dataset/train/COCO_mul_train_annos.json"
img_path = "./car_damage_dataset/img/"

# coco dataset api 초기화
coco = COCO(annfile_path)
mul_coco = COCO(mul_annfile_path)
"""
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
"""
# class info
cats = coco.loadCats(coco.getCatIds())
coco_class_name = [cat['name'] for cat in cats]
print("COCO categories for damages \n{}\n".format(', '.join(coco_class_name)))

mul_cats = mul_coco.loadCats(mul_coco.getCatIds())
mul_class_name = [cat['name'] for cat in mul_cats]
print("COCO categories for damages \n{}\n".format(', '.join(mul_class_name)))

"""
COCO categories for damages 
damage

COCO categories for damages 
headlamp, rear_bumper, door, hood, front_bumper
"""

# 특정 클래스의 이미지 id 가져오기
catIds = coco.getCatIds(catNms=['damage'])
imgIds = coco.getImgIds(catIds = catIds)
random_img_id = random.choice(imgIds)
# print(f"{random_img_id} image id was selected at random from the {imgIds} ")

# 랜덤으로 선택된 load the image
imgIds = coco.getImgIds(imgIds=[random_img_id])
print(imgIds)
img = coco.loadImgs(imgIds)[0]
# {'coco_url': '', 'date_captured': '2020-07-14 09:59:34.190485',
# 'file_name': '21.jpg', 'flickr_url': '', 'height': 1024, 'id': 10, 'license': 1, 'width': 1024}
image_path = os.path.join(img_path, img['file_name'])
image = Image.open(image_path)
image_org = image.copy()

# 이미지 화면 출력
plt.axis('off')
plt.imshow(image)
plt.show()

# get damage annotations
annIds = coco.getAnnIds(imgIds=imgIds, iscrowd=None)
anns = coco.loadAnns(annIds)

# plot damages
plt.imshow(image)
plt.axis('off')
coco.showAnns(anns, draw_bbox=True)
plt.show()

mul_annIds = mul_coco.getAnnIds(imgIds=imgIds, iscrowd=None)
mul_anns = mul_coco.loadAnns(mul_annIds)

# 카테고리 매핑
category_map = dict()

for ele in list(mul_coco.cats.values()):
    category_map.update({ele['id']:ele['name']})
# category_map >> {1: 'headlamp', 2: 'rear_bumper', 3: 'door', 4: 'hood', 5: 'front_bumper'}

# create a list of parts in the image
parts = []
for region in mul_anns :
    parts.append(category_map[region['category_id']])

# print("Parts are : ", parts)

# plot Parts
plt.imshow(image_org)
plt.axis('off')
mul_coco.showAnns(mul_anns, draw_bbox=True)
plt.show()