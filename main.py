
import cv2
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
# 加载模型组件库
from components.hand_detect.yolo_v3_hand import yolo_v3_hand_model

from components.hand_keypoints.handpose_x import handpose_x_model
from components.hand_track.hand_track import Tracker
from components.hand_track.hand_track import pred_gesture
from lib.hand_lib.utils.utils import parse_data_cfg

from components.hand_gesture.resnet import resnet18

import torch
import argparse

def handpose_x_process(info_dict,config, is_videos, test_path):
    # 模型初始化
    print("load model component  ...")
    # yolo v3 手部检测模型初始化
    if config["detect_model_arch"] == 'yolo_v3':
        hand_detect_model = yolo_v3_hand_model(conf_thres=float(config["detect_conf_thres"]),nms_thres=float(config["detect_nms_thres"]),
            model_arch = config["detect_model_arch"],model_path = config["detect_model_path"])
    else:
        print('error : 无效检测模型输入')
        return None
    # handpose_x 21 关键点回归模型初始化
    handpose_model = handpose_x_model(model_arch = config["handpose_x_model_arch"],model_path = config["handpose_x_model_path"])

    # 识别手势
    gesture_model = resnet18()
    gesture_model = gesture_model.cuda()
    gesture_model.load_state_dict(torch.load(config["gesture_model_path"]))
    gesture_model.eval()

    print("start handpose process ~")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if is_videos:
        if args.video_path=='0':
            cap = cv2.VideoCapture(0) # 开启摄像机
        else:
            cap = cv2.VideoCapture(args.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        size = ( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter('outVideo.mp4', fourcc, fps, size)

        tracking = Tracker()
        while True:
            ret, img = cap.read()# 读取相机图像

            if ret:# 读取相机图像成功
                algo_img = img.copy()
                st_ = time.time()
                #------
                hand_bbox =hand_detect_model.predict(img,vis = True) # 检测手，获取手的边界框
                tracking.hand_tracking(hand_bbox)
                # 检测每个手的关键点及相关信息
                tracking.handpose_keypoints21(img, algo_img, handpose_model=handpose_model,gesture_model=gesture_model,vis=True)
                et_ = time.time()
                print('单帧耗时：', et_-st_)
                fps_ = 1. / (et_ - st_ + 1e-8)
                cv2.putText(img, 'fps :{} '.format(fps_),
                            (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

                out.write(img)

                cv2.imshow("image", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        out.release()
        cap.release()
    else:
        for img_name in os.listdir(test_path):
            print(img_name)
            img = cv2.imread(os.path.join(test_path, img_name))
            algo_img = img.copy()
            hand_bbox = hand_detect_model.predict(img, vis=True)  # 检测手，获取手的边界框
            for h_box in hand_bbox:
                x_min, y_min, x_max, y_max, score = h_box
                w_ = max(abs(x_max - x_min), abs(y_max - y_min))
                if w_ < 60:
                    continue
                w_ = w_ * 1.26

                x_mid = (x_max + x_min) / 2
                y_mid = (y_max + y_min) / 2

                x1, y1, x2, y2 = int(x_mid - w_ / 2), int(y_mid - w_ / 2), int(x_mid + w_ / 2), int(y_mid + w_ / 2)

                x1 = np.clip(x1, 0, img.shape[1] - 1)
                x2 = np.clip(x2, 0, img.shape[1] - 1)

                y1 = np.clip(y1, 0, img.shape[0] - 1)
                y2 = np.clip(y2, 0, img.shape[0] - 1)

                box = [x1,y1,x2,y2]

                pts_ = handpose_model.predict(algo_img[y1:y2, x1:x2, :])  # 预测手指关键点

                gesture_name = pred_gesture(box, pts_, algo_img, gesture_model)
                print(gesture_name)
                cv2.putText(img, 'LABEL {}'.format(gesture_name), (int(x_min + 2), int(y_min + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 3)
            cv2.imshow("image", img)
            cv2.waitKey(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Hand Pose Inference')
    parser.add_argument('--cfg_file', type=str, default = 'lib/hand_lib/cfg/handpose.cfg',
        help = 'model_path') # 模型路径
    parser.add_argument('--test_path', type=str, default = 'test_gesture/',
        help = 'test_path') # 测试图片路径 'weights/handpose_x_gesture_v1/handpose_x_gesture_v1/000-one' camera_id
    parser.add_argument('--is_video', type=bool, default=False,
                        help='if test_path is video')  # 是否视频
    parser.add_argument('--video_path', type=str, default='0',
                        help='0 for cam / path ')  # 是否视频

    print('\n/******************* {} ******************/\n'.format(parser.description))
    args = parser.parse_args()  # 解析添加参数

    config = parse_data_cfg(args.cfg_file)
    is_videos = args.is_video
    test_path = args.test_path
    handpose_x_process(None, config, is_videos, test_path)

