#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@author:  Zj Meng
@file:    utils.py
@time:    2023-06-10 15:35
@contact: ymlfvlk@gmail.com 
@desc: "Welcome contact me if any questions"

"""
import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import LogNorm

# 重命名数据
def rename_and_convert_images(folder_path):
    subfolders = os.listdir(folder_path)

    for folder in subfolders:
        folder_full_path = os.path.join(folder_path, folder)
        if os.path.isdir(folder_full_path):
            image_files = os.listdir(folder_full_path)
            image_count = 0

            for image_file in image_files:
                image_file_path = os.path.join(folder_full_path, image_file)

                # Check if the file is an image
                if os.path.isfile(image_file_path) and image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG')):
                    image_count += 1
                    new_file_name = f"{image_count}.jpg"
                    new_file_path = os.path.join(folder_full_path, new_file_name)

                    # Convert image format to JPEG
                    img = Image.open(image_file_path)
                    img.convert("RGB").save(new_file_path, "jpeg")

                    # Remove the original image file
                    os.remove(image_file_path)



# 统计原始数据个数并画分布图
def collect_image_info(folder_path):
    num = 0
    df = pd.DataFrame(columns=['类别', '文件名', '图像宽', '图像高'])

    subfolders = os.listdir(folder_path)
    for folder in subfolders:
        # if num > 10:
        #     break
        folder_full_path = os.path.join(folder_path, folder)
        if os.path.isdir(folder_full_path):
            image_files = os.listdir(folder_full_path)

            for image_file in image_files:
                image_file_path = os.path.join(folder_full_path, image_file)
                try:
                    # num += 1
                    # if num > 10:
                    #     break
                    # print(image_file_path)
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

    print("----已统计完，可视化中----")
    x = df['图像宽'].astype(float)
    y = df['图像高'].astype(float)

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    plt.figure(figsize=(10, 10))
    # plt.figure(figsize=(12,12))
    plt.scatter(x, y, c=z, s=5, cmap='Spectral_r')
    # plt.colorbar()
    # plt.xticks([])
    # plt.yticks([])

    plt.tick_params(labelsize=15)

    xy_max = max(max(df['图像宽']), max(df['图像高']))
    plt.xlim(xmin=0, xmax=xy_max)
    plt.ylim(ymin=0, ymax=xy_max)

    plt.ylabel('height', fontsize=25)
    plt.xlabel('width', fontsize=25)

    plt.savefig('图像尺寸分布.pdf', dpi=120, bbox_inches='tight')

    plt.show()

    return df


# 平铺方式列出图像
def display_img(folder_path, N):
    images = []
    for each_img in os.listdir(folder_path)[:N]:
        img_path = os.path.join(folder_path, each_img)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111,  # 类似绘制子图 subplot(111)
                     nrows_ncols=(7, 6),  # 创建 n 行 m 列的 axes 网格
                     axes_pad=0.02,  # 网格间距
                     share_all=True
                     )

    # 遍历每张图像
    for ax, im in zip(grid, images):
        ax.imshow(im)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
#
# n_classes = 31
# n_samples = 74
# accuracy = 0.09
#
# # 生成混淆矩阵
# conf_matrix = np.zeros((n_classes, n_classes))
# for c in range(n_classes):
#     # 计算本类被正确分类的样本数
#     n_correct = int(n_samples * accuracy)
#
#     # 随机选择被正确分类的样本
#     correct_ix = np.random.choice(n_samples, size=n_correct, replace=False)
#
#     # 填入对角线元素
#     conf_matrix[c, c] = len(correct_ix)
#
#     # 计算本类被误分类的样本数
#     n_wrong = n_samples - n_correct
#
#     # 随机选择被误分类的样本及其类别
#     wrong_ix = np.random.choice(n_samples, size=n_wrong, replace=False)
#     wrong_cat = np.random.choice(list(range(c)) + list(range(c + 1, n_classes)), size=n_wrong)
#
#     # 填入非对角线元素
#     for i in range(len(wrong_ix)):
#         conf_matrix[c, wrong_cat[i]] += 1
#
#     # 显示混淆矩阵
# plt.matshow(conf_matrix, cmap=plt.cm.Blues)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()
# plt.savefig('confusion_matrix_heatmap1.png')
