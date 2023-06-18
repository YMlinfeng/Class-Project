#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@author:  Zj Meng
@file:    1.py.py
@time:    2023-06-14 14:31
@contact: ymlfvlk@gmail.com 
@desc: "Welcome contact me if any questions"

"""
import utils
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from PIL import Image


import matplotlib.pyplot as plt

dataset_path = '/data/ay/pj/roadCLS/init/pre'
dataset_path_1 = '/data/ay/pj/roadCLS/init/1'
# os.chdir(dataset_path_1)
# print(os.listdir())


# def rename_and_convert_images(folder_path):
#     subfolders = os.listdir(folder_path)
#     print(subfolders)
#     num = 0
#     for folder in subfolders:
#         # print(num)
#         folder_full_path = os.path.join(folder_path, folder)
#         if os.path.isdir(folder_full_path):
#             image_files = os.listdir(folder_full_path)
#             image_count = 0
#
#             for image_file in image_files:
#                 num += 1
#                 # print(num)
#     print(num)
#
#
# rename_and_convert_images(r'/data/ay/pj/roadCLS/init/raw_data1')
# exit(0)



# if os.getcwd() == r'/data/ay/pj/roadCLS/init/pre':
#     print(f"--------now refactor the source directory--------")
#     utils.rename_and_convert_images(dataset_path_1)
#     print(f"--------successfully refactored!--------")

# df = utils.collect_image_info(dataset_path)
# print(type(df))
# print(df)


utils.display_img(r'/data/ay/pj/roadCLS/init/result', 42)


# import torch
# from torchvision.models import resnet18
# from torchcam.methods import SmoothGradCAMpp
# from torchvision import transforms
# from torchcam.utils import overlay_mask
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # print('device', device)
#
# model = resnet18(pretrained=True).eval().to(device)


# CAM GradCAM GradCAMpp ISCAM LayerCAM SSCAM ScoreCAM SmoothGradCAMpp XGradCAM

# cam_extractor = SmoothGradCAMpp(model)


#
# # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
# test_transform = transforms.Compose([transforms.Resize(256),
#                                      transforms.CenterCrop(224),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(
#                                          mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#                                     ])
#
#
# img_path = '/data/ay/pj/roadCLS/init/2/3.jpg'
#
# img_pil = Image.open(img_path)
# input_tensor = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
#
# pred_logits = model(input_tensor)
# pred_top1 = torch.topk(pred_logits, 1)
# pred_id = pred_top1[1].detach().cpu().numpy().squeeze().item()
#
#
# activation_map = cam_extractor(pred_id, pred_logits)
#
#
# activation_map = activation_map[0][0].detach().cpu().numpy()
#
#
#
# result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.7)
#
# result.save("/data/ay/pj/roadCLS/init/result3.jpg")


import torch
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
from torchvision import transforms
from torchcam.utils import overlay_mask
# import os
# from torchvision import transforms
# from torchcam.utils import overlay_mask
# import torch
# from torchvision.models import resnet18
# from torchcam.methods import SmoothGradCAMpp
# def process_folder(input_folder, output_folder):
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = resnet18(pretrained=True).eval().to(device)
#     cam_extractor = SmoothGradCAMpp(model)
#     # 创建输出文件夹
#     os.makedirs(output_folder, exist_ok=True)
#
#     # 定义图像预处理
#     test_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     # 获取输入文件夹中的所有图像文件
#     image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
#
#     for image_file in image_files:
#         # 构建输入图像的完整路径
#         input_path = os.path.join(input_folder, image_file)
#
#         # 读取输入图像
#         img_pil = Image.open(input_path)
#
#         # 预处理图像
#         input_tensor = test_transform(img_pil).unsqueeze(0).to(device)
#
#         # 预测和生成 CAM
#         pred_logits = model(input_tensor)
#         pred_top1 = torch.topk(pred_logits, 1)
#         pred_id = pred_top1[1].detach().cpu().numpy().squeeze().item()
#         activation_map = cam_extractor(pred_id, pred_logits)
#         activation_map = activation_map[0][0].detach().cpu().numpy()
#
#         # 叠加 CAM 到图像上
#         result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.7)
#
#         # 构建输出图像的完整路径
#         output_path = os.path.join(output_folder, image_file)
#
#         # 保存结果图像
#         result.save(output_path)
#
#     print("处理完成！")
#
#
# # 调用函数，传入输入文件夹和输出文件夹的路径
# input_folder = "/data/ay/pj/roadCLS/init/1/2"
# output_folder = "/data/ay/pj/roadCLS/init/result"
# process_folder(input_folder, output_folder)









