# encoding=gbk
import base64
import concurrent
import io
import torch
import torch.nn.functional as Fu
from tqdm import tqdm

from network import *
from util import *
from data_loader import *
import grpc
from PIL import Image

import msg_pb2
import msg_pb2_grpc

# // 并发
from concurrent import futures
import time
import cv2
import numpy as np
import concurrent
import io
import os
# // 并发
from concurrent import futures
import base64
import cv2
import numpy as np
from PIL import Image

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
model_path = './model_save/U_Net-300-0.0001-102-0.4000.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_type = 'U_Net'
model = None
if model_type == 'My_Net':
    model = My_Net(img_ch=3, output_ch=1)
elif model_type == 'GaborNet':
    model = GaborNet(img_ch=3, output_ch=1)
elif model_type == 'U_Net':
    model = U_Net(img_ch=3, output_ch=1)
elif model_type == 'FMU_Net':
    model = FMU_Net(img_ch=3, output_ch=1)
model.to(device)
model.load_state_dict(torch.load(model_path))
model.train(False)
model.eval()

def process_test(img):
    ovlp_ita = 3  # 重叠率，你可以理解为在CT图像中截取好的个patch，这些patch直接是有重叠部分的，这个数值越大，重叠的就越多,
    patch_size = [256, 256]
    image_size = 256
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
    ori_image = img # 图片格式：PIL.Image.Image
    ori_image_np = np.array(ori_image) # 格式：np.array
    # 原图大小, 原来大小（w,h,3)
    ori_resize_dim = np.array(ori_image_np.shape).astype('int')

    # 将多个图片分割成小块
    cube_list = decompose_vol2cube(ori_image_np, 1, patch_size, ovlp_ita)

    # 保存小块的分割结果
    result_list = []
    for c in tqdm(range(len(cube_list))):
        image = Image.fromarray(cube_list[c][0, ...].astype('uint8'))
        aspect_ratio = image.size[1] / image.size[0]
        Transform = []
        ResizeRange = random.randint(300, 320)
        Transform.append(T.Resize((int(ResizeRange * aspect_ratio), ResizeRange)))
        p_transform = random.random()

        Transform.append(T.Resize((int(96 * aspect_ratio) - int(96 * aspect_ratio) % 16, 96)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)
        image = image.to(device)
        image = torch.unsqueeze(image, 0)
        SR = Fu.sigmoid(model(image))
        SR = Fu.interpolate(SR, size=(256, 256), mode='bicubic', align_corners=False)
        with torch.no_grad():
            # 将图片转换到CPU
            SR = SR.cpu().numpy()
            # print(SR.shape)
            result_list.append(SR)

    # ori_resize_dim = （h,w）
    ori_resize_dim = np.array([ori_resize_dim[0], ori_resize_dim[1]])
    # 把所有的分割小块进行拼接
    result_image = compose_label_cube2vol(result_list, ori_resize_dim, patch_size, ovlp_ita)
    result_image = cv2.cvtColor(result_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return result_image

def process_canny(img):
    img1 = img
    contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    canvas = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:] = (0, 0, 0)
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 5)
    # edges = cv2.Canny(gray,100,150)
    # cv2.imwrite("D:\\yycodes\\canny\\photocanny\\1101_canny.png", edges)
    return canvas


def process_fusion(img1, img2):
    rows, cols, chn = img1.shape
    # r1 = cv2.split(img1)[0]
    # b1 = cv2.split(img1)[0]
    g1 = cv2.split(img1)[0]
    # G, R通道设置为0,
    r1 = np.zeros((rows, cols), dtype=img1.dtype)
    b1 = np.zeros((rows, cols), dtype=img1.dtype)
    # g1 = np.zeros((rows, cols), dtype=img1.dtype)
    m1 = cv2.merge([r1, g1, b1])
    img = cv2.add(m1, img2)
    return img

# 启动一个线程来处理图像
def process_image_thread(input_image):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(process_test, input_image)
        img_res = future.result()
    return img_res  # 返回原始图像或处理后的图像

def GetMsg():
    try:
        # 2. 解码图像字符串
        # image_bytes = base64.b64decode(request.name)
        # 3. 创建图像对象
        img = Image.open('./photo/test.png')
        print("Received img")
        # img1 = process_image_thread(
        #     resize_image_to_max_size(np.array(img))
        # )
        img1 = process_image_thread(img)
        cv2.imencode(".jpg", img1)[1].tofile(
            os.path.join(os.getcwd(), "photo\\ori.jpg"))
        # img1 = process_image_thread(img)
        print("********")
        img2 = process_canny(cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY))
        cv2.imencode(".jpg", img2)[1].tofile(
            os.path.join(os.getcwd(), "photo\\canny.jpg"))
        print("*********")
        img3 = process_fusion(img2,
                              cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
        cv2.imencode(".jpg", img3)[1].tofile(
            os.path.join(os.getcwd(), "photo\\fusion.jpg"))
        print("**********")
        # 获取img1的byte数组，然后获取base64字符串
        print(img1.dtype)
        print(img2.dtype)
        print(img3.dtype)
        img_res1_str = base64.b64encode(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).tobytes()).decode()
        img_res2_str = base64.b64encode(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).tobytes()).decode()
        img_res3_str = base64.b64encode(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB).tobytes()).decode()
        print(img_res1_str[0:100])
        print(img_res2_str[0:100])
        print(img_res3_str[0:100])
    except Exception as e:
        print(e)

if __name__ == '__main__':
    GetMsg()

