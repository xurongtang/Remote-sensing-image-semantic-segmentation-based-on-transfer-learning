import json
import numpy as np
import cv2
from tqdm import tqdm

def json_to_mask(json_file_path,save_path):
    # 读取JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    annotations = data['annotations']
    image_info_ls = data['images']
    # print(len(image_info))
    # print(image_info[2])
    # Image:3784, len(annotations)=148983
    for i,img in enumerate(tqdm(image_info_ls,colour='green')):
        img_name = img['file_name'].split('/')[-1].split('.')[0]
        img_id = img['id']
        # Create a empty mask array
        height,width = img['height'],img['width']
        temp_mask = np.zeros((height,width),dtype=np.uint8)
        for j,ann in enumerate(annotations):
            if ann['image_id'] == img_id:
                segment = ann['segmentation']
                for polygon in segment:
                    polygon = np.array(polygon,dtype=np.int32).reshape(-1,2)
                    cv2.fillPoly(temp_mask,[polygon],1)
        mask_file_path = save_path + 'label/' + img_name + '.jpg'
        np.save(mask_file_path.replace('.jpg','.npy'),temp_mask)
        cv2.imwrite(mask_file_path,temp_mask*255)

# 使用示例
rootpath = '/home/rton/pyproj/ieee_building_extract/MyProj/data/'
train_json_path = rootpath + 'train/train.json'
valid_json_path = rootpath + 'val/val.json'
train_save_path = '/home/rton/pyproj/ieee_building_extract/MyProj/data/train/'
val_save_path = '/home/rton/pyproj/ieee_building_extract/MyProj/data/val/'

json_to_mask(train_json_path,train_save_path)
json_to_mask(valid_json_path,val_save_path)
