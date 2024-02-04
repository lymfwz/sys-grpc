# encoding=gbk
import base64
import concurrent
import io
import torch.nn.functional as Fu
from tqdm import tqdm

from network import *
from util import *
from data_loader import *
import grpc
import torch
from PIL import Image

import msg_pb2
import msg_pb2_grpc

# // ����
from concurrent import futures
import time
import cv2
import numpy as np

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
    ovlp_ita = 3  # �ص��ʣ����������Ϊ��CTͼ���н�ȡ�õĸ�patch����Щpatchֱ�������ص����ֵģ������ֵԽ���ص��ľ�Խ��,
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
    # ��ȡͼ��
    ori_image = img  # ͼƬ��ʽ��PIL.Image.Image
    ori_image_np = np.array(ori_image)  # ��ʽ��np.array
    # ԭͼ��С, ԭ����С��w,h,3)
    ori_resize_dim = np.array(ori_image_np.shape).astype('int')

    # �����ͼƬ�ָ��С��
    cube_list = decompose_vol2cube(ori_image_np, 1, patch_size, ovlp_ita)

    # ����С��ķָ���
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
            # ��ͼƬת����CPU
            SR = SR.cpu().numpy()
            # print(SR.shape)
            result_list.append(SR)

    # ori_resize_dim = ��h,w��
    ori_resize_dim = np.array([ori_resize_dim[0], ori_resize_dim[1]])
    # �����еķָ�С�����ƴ��
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
    # G, Rͨ������Ϊ0,
    r1 = np.zeros((rows, cols), dtype=img1.dtype)
    b1 = np.zeros((rows, cols), dtype=img1.dtype)
    # g1 = np.zeros((rows, cols), dtype=img1.dtype)
    m1 = cv2.merge([r1, g1, b1])
    img = cv2.add(m1, img2)
    return img


# ����һ���߳�������ͼ��
def process_image_thread(input_image):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(process_test, input_image)
        img_res = future.result()
    return img_res  # ����ԭʼͼ��������ͼ��


def resize_image_to_max_size(image, max_size_bytes=2 * 1024 * 1024):
    # ���㵱ǰͼ����ֽڴ�С
    current_size_bytes = image.nbytes
    # ���ͼ���С�Ѿ�����ָ��������С�����������С
    if current_size_bytes <= max_size_bytes:
        return image
    # ������С������ʹͼ���С��������С
    resize_ratio = (max_size_bytes / current_size_bytes) ** 0.5
    new_width = int(image.shape[1] * resize_ratio)
    new_height = int(image.shape[0] * resize_ratio)
    # ����ͼ���С
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


# // service ʵ��GetMsg����
def img2BytesStr(img):
    img = Image.fromarray(img)
    # ��ͼ��ת��Ϊ�ֽ�����
    with io.BytesIO() as output:
        img.save(output, format="JPEG")  # ָ��ͼ���ʽ��������Ҫ����
        image_data = output.getvalue()
    img_res_str = base64.b64encode(image_data).decode()
    return img_res_str

class MsgServicer(msg_pb2_grpc.MsgServiceServicer):

    def GetMsg(self, request, context):
        try:
            # 2. ����ͼ���ַ���
            image_bytes = base64.b64decode(request.name)
            # 3. ����ͼ�����
            img = Image.open(io.BytesIO(image_bytes))
            print("Received img")
            # img1 = process_image_thread(
            #     resize_image_to_max_size(np.array(img))
            # )
            img1 = process_image_thread(img)
            cv2.imencode(".jpg", img1)[1].tofile(
                os.path.join(os.getcwd(), "photo\\ori.jpg"))
            # img1 = process_image_thread(img)
            img2 = process_canny(cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY))
            cv2.imencode(".jpg", img2)[1].tofile(
                os.path.join(os.getcwd(), "photo\\canny.jpg"))
            img3 = process_fusion(img2,
                                  cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
            cv2.imencode(".jpg", img3)[1].tofile(
                os.path.join(os.getcwd(), "photo\\fusion.jpg"))

            # ��ȡimg1��byte���飬Ȼ���ȡbase64�ַ���
            # ͼ��תΪ�ֽ�����
            img_res1_str = img2BytesStr(img1)
            img_res2_str = img2BytesStr(img2)
            img_res3_str = img2BytesStr(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))

            print("������ԣ�" + img_res1_str[0:10] + "*****" + img_res2_str[0:10] + "*****" + img_res3_str[0:10])
            # print("Received name: %s" % request.name)
            res_str = img_res1_str + "," + img_res2_str + "," + img_res3_str
            return msg_pb2.MsgResponse(msg='%s' % res_str)
            # return msg_pb2.MsgResponse(img_res_str)
        except Exception as e:
            print(e)
            return msg_pb2.MsgResponse(msg='error')


def serve():
    max_message_length = 1024 * 1024 * 1024  # ���������Ϣ��СΪ1GB

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
        ('grpc.max_receive_message_length', max_message_length)  # ������������Ϣ��С
    ])
    # server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    msg_pb2_grpc.add_MsgServiceServicer_to_server(MsgServicer(), server)
    server.add_insecure_port('[::]:50055')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()