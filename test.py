# encoding=gbk
import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as Fu
from evaluation import *
from network import *
import csv
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.optim import Adam
from PIL import Image, ImageFilter
import copy
from util import *
import numpy as np
from data_loader import *
from glob import glob

# ==================以下需要自己定义=======================================
# 模型类型，需要自己写,有My_Net，GaborNet，U_Net，FMU_Net，R2U_Net，AttU_Net，R2AttU_Net
model_type = 'U_Net'
image_path = './photo/test.png'  # 要分割的图片路径
result_image_path = 'photoorg/p1.jpg'  # 保存最后分割结果
ovlp_ita = 3  # 重叠率，你可以理解为在CT图像中截取好的个patch，这些patch直接是有重叠部分的，这个数值越大，重叠的就越多,
patch_size = [256, 256]
model_path = './model_save/U_Net-300-0.0001-102-0.4000.pkl'
# ==================以上需要自己定义=======================================

# ================================================================================================#
image_size = 256
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model():
    """Build generator and discriminator."""
    if model_type == 'My_Net':
        model = My_Net(img_ch=3, output_ch=1)
    elif model_type == 'GaborNet':
        model = GaborNet(img_ch=3, output_ch=1)
    elif model_type == 'U_Net':
        model = U_Net(img_ch=3, output_ch=1)
    elif model_type == 'FMU_Net':
        model = FMU_Net(img_ch=3, output_ch=1)
    model.to(device)
    return model


def test():
    # 初始化模型
    model = build_model()
    model.load_state_dict(torch.load(model_path))

    model.train(False)
    model.eval()

    acc = 0.  # Accuracy
    SE = 0.  # Sensitivity (Recall)
    SP = 0.  # Specificity
    PC = 0.  # Precision
    F1 = 0.  # F1 Score
    JS = 0.  # Jaccard Similarity
    DC = 0.  # Dice Coefficient
    length = 0
    i = 0

    # 读取图像
    ori_image = Image.open(image_path)
    ori_image_np = np.array(ori_image)
    # 原图大小, 原来大小（w,h,3)
    ori_resize_dim = np.array(ori_image_np.shape).astype('int')

    
    # 将多个图片分割成小块
    print("开始裁剪小块")
    cube_list = decompose_vol2cube(ori_image_np, 1, patch_size, ovlp_ita)
    
    # for c in range(len(cube_list)):
      # 保存每个patch
    #   current_image = cube_list[c][0,...]
    #   current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
    #   cv2.imwrite(current_image_path+str(c)+".png",current_image)
    print("裁剪完成")
    
    # 保存小块的分割结果
    result_list = []



    for c in range(len(cube_list)):
        print(c,"/",len(cube_list))
        image = Image.fromarray(cube_list[c][0,...].astype('uint8'))
        aspect_ratio = image.size[1]/image.size[0]
        Transform = []
        ResizeRange = random.randint(300,320)
        Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))
        p_transform = random.random()
        
        Transform.append(T.Resize((int(96*aspect_ratio)-int(96*aspect_ratio)%16,96)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        
        image = Transform(image)
        image = image.to(device)
        image = torch.unsqueeze(image, 0)
        SR = Fu.sigmoid(model(image))
        # one = torch.ones_like(SR)
        # zero = torch.zeros_like(SR)
        # SR = torch.where(SR > 0.5, one, SR)
        # SR = torch.where(SR < 0.5, zero, SR)
        SR = Fu.interpolate(SR, size=(256, 256), mode='bicubic', align_corners=False)
        # print("输出大小",SR.size())
        # torchvision.utils.save_image(SR.data.cpu(),"./result_image/"+str(c)+".png")
        with torch.no_grad():
          # 将图片转换到CPU
          SR = SR.cpu().numpy()
          # print(SR.shape)
          result_list.append(SR)

    # ori_resize_dim = （h,w）
    ori_resize_dim = np.array([ori_resize_dim[0],ori_resize_dim[1]])    
    # 把所有的分割小块进行拼接
    result_image = compose_label_cube2vol(result_list, ori_resize_dim, patch_size, ovlp_ita)
    return result_image

result_image = test()
result_image = cv2.cvtColor(result_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite(result_image_path,result_image)
