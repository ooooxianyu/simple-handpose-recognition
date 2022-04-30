import copy
import cv2
import numpy as np
import math
from components.hand_track.utils import *
from components.hand_keypoints.handpose_x import draw_bd_handpose_c, draw_mask_handpose

from PIL import Image
import torchvision.transforms as transforms
import torch

tags = {'0':'one', '1':'five', '2':'fist', '3':'ok', '4':'heartSingle',
       '5':'yearh', '6':'three', '7':'four', '8':'six',
       '9':'Iloveyou', '10':'gun', '11':'thunbUp', '12':'nine', '13':'pink'}

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

def pred_gesture(box, pts_, algo_img, gesture_model):
    x1, y1, x2, y2 = box
    pts_hand = {}
    for ptk in range(int(pts_.shape[0] / 2)):
        xh = (pts_[ptk * 2 + 0] * float(x2 - x1))
        yh = (pts_[ptk * 2 + 1] * float(y2 - y1))
        pts_hand[str(ptk)] = {
            "x": xh,
            "y": yh,
        }

    img_mask = np.ones(algo_img.shape, dtype=np.uint8)
    img_mask[:, :, 0] = 255
    img_mask[:, :, 1] = 255
    img_mask[:, :, 2] = 255

    draw_mask_handpose(img_mask, pts_hand, x1, y1, int(((x2 - x1) + (y2 - y1)) / 128))

    # 检测手势动作
    s_img_mask = img_mask[y1:y2, x1:x2, :]
    s_img_mask = cv2.resize(s_img_mask, (128, 128))

    s_img_mask = Image.fromarray(s_img_mask)

    if transform is not None:
        s_img_mask = transform(s_img_mask)
    s_img_mask = s_img_mask.unsqueeze(dim=0).cuda()
    output = gesture_model(s_img_mask)
    # print(output)

    pre_tag = torch.argmax(output, dim=1)[0].cpu().detach().tolist()
    gesture_name = tags[str(pre_tag)]
    return gesture_name

class hand_data:
    # 存放手掌数据
    def __init__(self, hand_dict, hand_id):
        self.title_boxes = []
        # self.gesture_lines_dict = {}  # 点击使能时的轨迹点
        self.index_finger_track = []  # 跟踪食指的位置
        self.hand_dict = hand_dict  # 手的信息 list
        self.hand_click_count = 0  # 手的按键信息计数 int
        self.gesture = None # 记录手势状态次数 （gesture,conut）
        # self.track_index = 0  # 跟踪的全局索引
        self.hand_id = hand_id # 手编号
        self.click_state = False

    def get_id(self):
        return self.hand_id

class Tracker:
    # 实现跟踪手掌及其数据
    def __init__(self):
        self.tracks = {} # 存储手掌信息 class hand_data
        self.none_count = 0 # 单帧图片没有检测到手的次数
        self.hand_num = 0 # 单帧图片手的数量
        self.track_index = 0
        self.track_hand_ids = [] # 记录单帧图片的手掌编号ids
        self.index_offset = (0, 0)  # 食指偏移量
        self.xy_index = None

    def hand_tracking(self, data):
        none_count = self.none_count
        track_index = self.track_index
        hand_num = self.hand_num
        # hands_dict = self.tracks
        if len(data) == 0 or len(data) != hand_num:
            none_count += 1
            if none_count > 20:
                # print('done')
                hand_num = len(data)
                if hand_num==0:
                    self.tracks = {}
                # hands_dict = {}
                # track_index = 0

        else:
            none_count = 0
            track_index = self.tracking_bbox(data, track_index)  # 目标跟踪

            # self.tracks = hand_dict
        self.track_index = track_index
        self.none_count = none_count
        self.hand_num = hand_num

        return 0

    def tracking_bbox(self,data, index, iou_thr=0.5):
        hands_dict = self.tracks
        track_index = index
        track_hand_ids = []
        reg_dict = {}
        Flag_ = True if hands_dict else False
        if Flag_ == False:
            # print("------------------->>. False")
            for bbox in data:
                x_min, y_min, x_max, y_max, score = bbox
                reg_dict[track_index] = (x_min, y_min, x_max, y_max, score, 0., 1, 1)
                track = hand_data(reg_dict[track_index], track_index)
                track_hand_ids.append(track_index)
                self.tracks[track_index] = track
                track_index += 1

                if track_index >= 65535:
                    track_index = 0
        else:
            # print("------------------->>. True ")
            for bbox in data:
                xa0, ya0, xa1, ya1, score = bbox
                is_track = False
                for k_ in hands_dict.keys():
                    xb0, yb0, xb1, yb1, _, _, cnt_, bbox_stanbel_cnt = hands_dict[k_].hand_dict

                    iou_ = compute_iou_tk((ya0, xa0, ya1, xa1), (yb0, xb0, yb1, xb1))
                    # print((ya0,xa0,ya1,xa1),(yb0,xb0,yb1,xb1))
                    # print("iou : ",iou_)
                    if iou_ > iou_thr:  # 跟踪成功目标
                        UI_CNT = 1
                        if iou_ > 0.888:
                            UI_CNT = bbox_stanbel_cnt + 1
                        reg_dict[k_] = (xa0, ya0, xa1, ya1, score, iou_, cnt_ + 1, UI_CNT)
                        track_hand_ids.append(k_)
                        self.up_boxdata(k_,reg_dict[k_] )

                        is_track = True
                        # print("is_track : " ,cnt_ + 1)
                if is_track == False:  # 新目标
                    reg_dict[track_index] = (xa0, ya0, xa1, ya1, score, 0., 1, 1)
                    track_hand_ids.append(track_index)
                    track = hand_data(reg_dict[track_index], track_index)
                    self.tracks[track_index] = track
                    track_index += 1
                    if track_index >= 65535:  # 索引越界归零
                        track_index = 0

                    if track_index >= 100:
                        track_index = 0

        # hand_dict = copy.deepcopy(reg_dict)
        # track = hand_data(reg_dict[track_index], track_index)
        # if len(track_hand_ids)>1:
        #     print("now:",track_hand_ids)
        self.track_hand_ids = track_hand_ids

        # ----------------- 获取跟踪到的手ID
        id_list = self.track_hand_ids
        print(id_list)
        # ----------------- 获取需要删除的手ID
        id_del_list = []
        for k_ in self.tracks.keys():
            if k_ not in id_list:  # 去除过往已经跟踪失败的目标手的相关轨迹
                id_del_list.append(k_)
        self.del_id(id_del_list)
        return track_index

    def up_boxdata(self, id_, hand_dict):
        self.tracks[id_].hand_dict = hand_dict

    def del_id(self, id_del_list):
        for k_ in id_del_list:
            del self.tracks[k_]

    def handpose_keypoints21(self, img, algo_img, handpose_model, gesture_model, vis):
        '''
        img:绘制图; algo_img:原图副本
        handpose_model: 关键点检测模型reXnet；gesture_model:手势识别模型 resnet18
        vis:是否可视化
        '''

        dst_thr = 35; angle_thr = 16

        hands_list = []
        hands_dict = self.tracks
        if algo_img is not None:

            for idx, id_ in enumerate(sorted(hands_dict.keys(), key=lambda x: x, reverse=False)):

                x_min, y_min, x_max, y_max, score, iou_, cnt_, ui_cnt = hands_dict[id_].hand_dict

                # x_min,y_min,x_max,y_max,score = bbox
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

                # bbox_ = x1, y1, x2, y2

                pts_ = handpose_model.predict(algo_img[y1:y2, x1:x2, :]) # 预测手指关键点

                img_mask = np.ones(algo_img.shape, dtype=np.uint8)
                img_mask[:, :, 0] = 255
                img_mask[:, :, 1] = 255
                img_mask[:, :, 2] = 255

                plam_list = []
                pts_hand = {}
                for ptk in range(int(pts_.shape[0] / 2)):
                    xh = (pts_[ptk * 2 + 0] * float(x2 - x1))
                    yh = (pts_[ptk * 2 + 1] * float(y2 - y1))
                    pts_hand[str(ptk)] = {
                        "x": xh,
                        "y": yh,
                    }
                    if ptk in [0, 1, 5, 9, 13, 17]:
                        plam_list.append((xh + x1, yh + y1))
                    if ptk == 0:  # 手掌根部
                        hand_root_ = int(xh + x1), int(yh + y1)
                    if ptk == 4:  # 大拇指
                        thumb_ = int(xh + x1), int(yh + y1)
                    if ptk == 8:  # 食指
                        index_ = int(xh + x1), int(yh + y1)
                    if vis:
                        if ptk == 0:  # 绘制腕关节点
                            cv2.circle(img, (int(xh + x1), int(yh + y1)), 9, (250, 60, 255), -1)
                            cv2.circle(img, (int(xh + x1), int(yh + y1)), 5, (20, 180, 255), -1)
                        cv2.circle(img, (int(xh + x1), int(yh + y1)), 4, (255, 50, 60), -1)
                        cv2.circle(img, (int(xh + x1), int(yh + y1)), 3, (25, 160, 255), -1)

                # 计算食指和大拇指中心坐标
                choose_pt = (int((index_[0] + thumb_[0]) / 2), int((index_[1] + thumb_[1]) / 2))
                # 计算掌心
                plam_list = np.array(plam_list)
                plam_center = (np.mean(plam_list[:, 0]), np.mean(plam_list[:, 1]))

                # 绘制掌心坐标圆
                if vis:
                    cv2.circle(img, (int(plam_center[0]), int(plam_center[1])), 12, (25, 160, 255), 9)
                    cv2.circle(img, (int(plam_center[0]), int(plam_center[1])), 12, (255, 190, 30), 2)

                # 计算食指大拇指的距离
                dst = np.sqrt(np.square(thumb_[0] - index_[0]) + np.square(thumb_[1] - index_[1]))
                # 计算大拇指和手指相对手掌根部的角度：
                angle_ = vector_2d_angle((thumb_[0] - hand_root_[0], thumb_[1] - hand_root_[1]),
                                         (index_[0] - hand_root_[0], index_[1] - hand_root_[1]))
                # 判断手的点击click状态，即大拇指和食指是否捏合
                click_state = False
                if dst < dst_thr and angle_ < angle_thr:  # 食指和大拇指的坐标欧氏距离，以及相对手掌根部的相对角度，两个约束关系判断是否点击
                    click_state = True
                    cv2.circle(img, choose_pt, 6, (0, 0, 255), -1)  # 绘制点击坐标，为轨迹的坐标
                    cv2.circle(img, choose_pt, 2, (255, 220, 30), -1)
                    cv2.putText(img, 'Click {:.1f} {:.1f}'.format(dst, angle_), (int(x_min + 2), y2 - 1),
                                cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 0, 0), 5)
                    cv2.putText(img, 'Click {:.1f} {:.1f}'.format(dst, angle_), (int(x_min + 2), y2 - 1),
                                cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255))
                else:
                    click_state = False
                    cv2.putText(img, 'NONE  {:.1f} {:.1f}'.format(dst, angle_), (int(x_min + 2), y2 - 1),
                                cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 0, 0), 5)
                    cv2.putText(img, 'NONE  {:.1f} {:.1f}'.format(dst, angle_), (int(x_min + 2), y2 - 1),
                                cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255))

                # ----------------------------------------------------
                # 记录手的点击（click）计数器，用于判断click稳定输出状态
                if id_ not in hands_dict.keys():  # 记录手的点击（click）计数器，用于稳定输出
                    hands_dict[id_].hand_click_count = 0
                if click_state == False:
                    hands_dict[id_].hand_click_count = 0
                elif click_state == True:
                    hands_dict[id_].hand_click_count += 1

                # --------------------- 绘制手的关键点连线
                draw_bd_handpose_c(img, pts_hand, x1, y1, 2)
                draw_mask_handpose(img_mask, pts_hand, x1, y1, int(((x2 - x1) + (y2 - y1)) / 128))

                # 检测手势动作
                s_img_mask = img_mask[y1:y2, x1:x2, :]
                s_img_mask = cv2.resize(s_img_mask, (128, 128))

                s_img_mask = Image.fromarray(s_img_mask)

                if transform is not None:
                    s_img_mask = transform(s_img_mask)
                s_img_mask = s_img_mask.unsqueeze(dim=0).cuda()
                output = gesture_model(s_img_mask)
                # print(output)

                pre_tag = torch.argmax(output, dim=1)[0].cpu().detach().tolist()
                gesture_name = tags[str(pre_tag)]
                if gesture_name == "gun":
                    gesture_name = "one"
                # print('label:', gesture_name)
                if id_ in hands_dict.keys() and hands_dict[id_].gesture is not None:
                    if hands_dict[id_].gesture[0] == gesture_name:
                        gesture_count = hands_dict[id_].gesture[1] + 1
                    else:
                        gesture_count = 0
                else:
                    gesture_count = 0
                # ----------------------------------------------------
                hands_list.append((pts_hand, (x1, y1), plam_center,
                                   {"id": id_, "click": click_state, "click_cnt": hands_dict[id_].hand_click_count,
                                    "gesture_name": gesture_name, "gesture_count": gesture_count,
                                    "choose_pt": choose_pt}))  # 局部21关键点坐标，全局bbox左上坐标，全局掌心坐标
                # 记录手势状态（gesture）计数器，用于手势稳定输出状态
                hands_dict[id_].gesture = (gesture_name, gesture_count)
                if gesture_name == 'one' and gesture_count >= 15:
                    if id_ in hands_dict and len(hands_dict[id_].index_finger_track)!=0:
                        hands_dict[id_].index_finger_track.append(index_)
                    else:
                        hands_dict[id_].index_finger_track = [index_]
                if id_ in hands_dict and vis:
                    if len(hands_dict[id_].index_finger_track) >= 2:
                        for point_c in range(len(hands_dict[id_].index_finger_track) - 1):
                            line_pointa = hands_dict[id_].index_finger_track[point_c]
                            line_pointb = hands_dict[id_].index_finger_track[point_c + 1]

                            cv2.line(img, line_pointa, line_pointb, (255, 0, 255), 5)  # 画轨迹

                # 清空轨迹
                if gesture_name == 'fist' and gesture_count >= 10:
                    hands_dict[id_].index_finger_track = []

                # 记录指尖偏移量
                if gesture_name == 'yearh':
                    x_offset, y_offset = 0,0
                    if gesture_count == 1:
                        self.xy_index = (index_[0],index_[1]) #记录第一次食指坐标
                    if gesture_count >= 30 and self.xy_index is not None: # 计算30帧指尖的偏移量
                        x_offset = index_[0]-self.xy_index[0]
                        y_offset = index_[1]-self.xy_index[1]
                    self.index_offset = (x_offset,y_offset)

                cv2.putText(img, 'offset x :{} y :{}'.format(self.index_offset[0], self.index_offset[1]), (10, 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                cv2.putText(img, 'hand-num :{} '.format(self.hand_num),
                            (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

                cv2.putText(img, 'LABEL {}'.format(tags[str(pre_tag)]), (int(x_min + 2), int(y_min + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 3)

            return hands_list

