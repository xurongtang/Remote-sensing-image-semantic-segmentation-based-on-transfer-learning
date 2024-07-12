import torch
import numpy as np
from PIL import Image
import glob
import cv2,sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
from torchvision import transforms
from model.base_model import Model

# 填充
def pad_image(image, target_size=(512, 512)):
    h, w, c = image.shape
    top = (target_size[0] - h) // 2
    bottom = target_size[0] - h - top
    left = (target_size[1] - w) // 2
    right = target_size[1] - w - left
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image, top, left

# 裁剪预测结果到原始尺寸
def crop_image(image, original_size=(500, 500), top=6, left=6):
    h, w = original_size
    return image[top:top+h, left:left+w]


def plotting_result(img,mask,img_name,save_path):
    name = img_name.split('.')[0]
    address = img_name.split('.')[-1]

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

    # 创建一个空白的图像作为绘制轮廓的目标
    contour_image = np.zeros_like(img, dtype=np.uint8)

    # 设置填充颜色和透明度
    fill_color = (0, 0, 255)  # 填充颜色为绿色
    alpha = 0.4  # 设置透明度

    # 绘制轮廓
    # cv2.drawContours(contour_image, contours, -1, fill_color, thickness=cv2.FILLED)
    cv2.drawContours(contour_image, contours, -1, (0,0,255),2)

    # 将绘制了轮廓的图像与原始图像进行叠加
    result_image = cv2.addWeighted(img, 1 - alpha, contour_image, alpha, 0)
    img = result_image
    cv2.imwrite(save_path + name+'_result.'+address,img)

# 比赛测试
# root_path = '/home/rton/pan1/competetion_data/test/images/*'
# model = torch.load('save_model/pick/competetion_model_epoch10.pt')
# model = torch.load('save_model/competition_Deeplabvpmodel_epoch5.pt')

# 实习测试
# model = torch.load('save_model/pick/intership_model_epoch5.pt')
# root_path = '/home/rton/pan1/CG_dataset/val/images/*'

# 
model_name = 'MAnet'
encoder_name = 'resnet34'
model = Model(model_name,encoder_name=encoder_name, in_channels=3, out_classes=1)
model.load_state_dict(torch.load('/home/rton/pyproj/ieee_building_extract/new_test/save_model/MAnet_resnet34_8_pretrain.pkl'))
model.eval()

root_path = '/home/rton/pan1/competetion_data/test/images/*'

file_path_ls = glob.glob(root_path)
for file_path in tqdm(file_path_ls,):
    image = np.array(Image.open(file_path).convert("RGB"))
    padding_image,top,left = pad_image(image,target_size=(512,512))
    # print(type(padding_image))
    # print(torch.from_numpy(padding_image).permute(2,0,1).shape)
    # print(model)
    # sys.exit()
    pr_masks = model(torch.from_numpy(padding_image).permute(2,0,1))
    pr_masks = pr_masks.sigmoid()
    mask_npy = pr_masks.detach().numpy().squeeze()
    # plt.imshow(mask_npy)
    mask = (mask_npy * 255).astype(np.uint8)
    _,binary_mask = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
    crop_mask = crop_image(binary_mask,original_size=(500,500),top=top,left=left)
    img_name = file_path.split('/')[-1].split('.')[0]
    cv2.imwrite('/home/rton/pyproj/ieee_building_extract/Pytorch-UNet/test_result/'+img_name+'.png',crop_mask)
    
    save_path = '/home/rton/pyproj/ieee_building_extract/Pytorch-UNet/test_result/'
    # cv2.imwrite(save_path + img_name +'.jpg',crop_mask)
    # save_path = '/home/rton/pyproj/ieee_building_extract/new_test/competetion_val_res/'
    # save_path = '/home/rton/pyproj/ieee_building_extract/new_test/intership_val_res/'
    img_name = file_path.split('/')[-1]
    plotting_result(image,crop_mask,img_name,save_path)
    # plt.colorbar()
    # plt.show()
    # break

