import torch
import numpy as np
from PIL import Image
import glob,sys
import cv2
import argparse
import tqdm
import torch
import numpy as np
from base_model import Model

def plotting_result(img,mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    # 创建一个空白的图像作为绘制轮廓的目标
    contour_image = np.zeros_like(img, dtype=np.uint8)
    # 设置填充颜色和透明度
    fill_color = (255, 0, 0)  # 填充颜色为绿色
    alpha = 0.4  # 设置透明度
    # 绘制轮廓
    cv2.drawContours(contour_image, contours, -1, fill_color, thickness=cv2.FILLED)
    # cv2.drawContours(contour_image, contours, -1, (0,0,255),2)
    # 将绘制了轮廓的图像与原始图像进行叠加
    result_image = cv2.addWeighted(img, 1 - alpha, contour_image, alpha, 0)    
    img = result_image
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


def big_image_test(file_path,model_path,window_size):
    
    model_name = 'MAnet'
    encoder_name = 'resnet34'

    big_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    # big_image = cv2.cvtColor(big_image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    # 调整清晰度 降采样
    # print(big_image.shape)
    # big_image = cv2.resize(big_image,None,fx=0.5, fy=0.5)
    # print(big_image.shape)

    # 加载你的模型
    model = Model(model_name,encoder_name=encoder_name, in_channels=3, out_classes=1)
    model.load_state_dict(torch.load(model_path))
    # model = torch.load('/home/rton/pyproj/ieee_building_extract/new_test/save_model/pick/competetion_model_epoch10.pt')
    model.eval()

    step_size = int(window_size/2)

    height, width, _ = big_image.shape
    # print(height,width)

    # 准备一个空的数组来保存预测结果
    predicted_image = np.zeros((height, width), dtype=np.float32)

    count_image = np.zeros((height, width), dtype=np.float32)
    # 滑动窗口预测

    for y in tqdm.tqdm(range(0, height, step_size)):
        for x in range(0, width, step_size):
            # 定义当前窗口的坐标
            y1, y2 = y, min(y + window_size, height)
            x1, x2 = x, min(x + window_size, width)

            # 提取当前窗口的图像
            window = big_image[y1:y2, x1:x2]

            # 如果窗口超出图像边界，则用零填充
            if y2 - y1 < window_size or x2 - x1 < window_size:
                padded_window = np.zeros((window_size, window_size, 3), dtype=np.uint8)
                padded_window[:y2-y1, :x2-x1] = window
            else:
                padded_window = window

            # 预处理图块
            image = np.array(padded_window)
            pr_masks = model(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float())
            pr_masks = pr_masks.sigmoid()
            mask_npy = pr_masks.detach().numpy().squeeze()

            # 将预测结果保存到对应位置
            # predicted_image[y1:y2, x1:x2] = mask_npy[:y2-y1, :x2-x1]
            predicted_image[y1:y2, x1:x2] += mask_npy[:y2-y1, :x2-x1]
            count_image[y1:y2, x1:x2] += 1

    # 直接保存预测结果
    # predicted_image = Image.fromarray(predicted_image)
    # predicted_image.save('predicted_large_image.png')

    predicted_image /= count_image
    predicted_image = (predicted_image * 255).astype(np.uint8)
    _,predicted_mask = cv2.threshold(predicted_image,1,255,cv2.THRESH_BINARY)
    predicted_image = plotting_result(big_image,predicted_mask)
    return predicted_image

    # # 分作3*3保存
    # sub_image_height = height // 3
    # sub_image_width = width // 3

    # for i in tqdm.tqdm(range(3)):
    #     for j in range(3):
    #         y1 = i * sub_image_height
    #         y2 = (i + 1) * sub_image_height if i < 2 else height
    #         x1 = j * sub_image_width
    #         x2 = (j + 1) * sub_image_width if j < 2 else width
    #         sub_image = predicted_image[y1:y2, x1:x2]
    #         sub_image_pil = Image.fromarray(sub_image)
    #         sub_image_pil.save(f'predicted_sub_image_{i}_{j}.tif')


def rs_big_pred(file_path,model_path):
    window_szie = 512
    test_image1 = file_path
    pred_mask = big_image_test(test_image1,model_path,window_szie)
    return pred_mask

if __name__ == "__main__":

    # model_path = '/home/rton/pyproj/ieee_building_extract/new_test/save_model/intership_finetune.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--encoder_name",type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--flag",type=str)
    args = parser.parse_args()
    rs_big_pred(args.model_name,
                args.encoder_name,
                args.model_path,
                args.flag)
