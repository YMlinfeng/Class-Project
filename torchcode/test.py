#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@author:  Zj Meng
@file:    test.py
@time:    2023-06-14 22:02
@contact: ymlfvlk@gmail.com 
@desc: "Welcome contact me if any questions"

"""
import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd
img = Image.open("/data/ay/pj/roadCLS/init/1/1/3.jpg")
img_array = np.array(img)
img_height, img_width, _ = img_array.shape
df = pd.DataFrame(columns=['类别', '文件名', '图像宽', '图像高'])
df = df._append({
    # '类别': folder,
    # '文件名': image_file,
    '图像宽': img_width,
    '图像高': img_height
}, ignore_index=True)
print(df)


def collect_image_info(folder_path):
    num = 0
    df = pd.DataFrame(columns=['类别', '文件名', '图像宽', '图像高'])

    subfolders = os.listdir(folder_path)
    for folder in subfolders:
        if num > 10:
            break
        folder_full_path = os.path.join(folder_path, folder)
        if os.path.isdir(folder_full_path):
            image_files = os.listdir(folder_full_path)

            for image_file in image_files:
                image_file_path = os.path.join(folder_full_path, image_file)
                try:
                    num += 1
                    if num > 10:
                        break
                    print(image_file_path)
                    img = Image.open(image_file_path)
                    img_array = np.array(img)
                    img_height, img_width, _ = img_array.shape

                    df = df._append({
                        '类别': folder,
                        '文件名': image_file,
                        '图像宽': img_width,
                        '图像高': img_height
                    }, ignore_index=True)
                except:
                    print(os.path.join(folder, image_file), '读取错误')

    return df
df = collect_image_info("/data/ay/pj/roadCLS/init/1")
print(df)