import cv2
import math
'''
    求解二维向量的角度
'''
def vector_2d_angle(v1,v2):
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        angle_=math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ =65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_

def compute_iou_tk(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles

    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0.
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

'''
    判断各手的click状态是否稳定（点击稳定充电环），即click是否持续一定阈值
    注意：charge_cycle_step 充电步长越大，触发时间越短
'''
def judge_click_stabel(img,handpose_list,charge_cycle_step = 32):
    flag_click_stable = True
    for i in range(len(handpose_list)):
        _,_,_,dict_ = handpose_list[i]
        id_ = dict_["id"]
        click_cnt_ = dict_["click_cnt"]
        pt_ = dict_["choose_pt"]
        if click_cnt_ > 0:
            # print("double_en_pts --->>> id : {}, click_cnt : <{}> , pt : {}".format(id_,click_cnt_,pt_))
            # 绘制稳定充电环
            # 充电环时间控制
            charge_cycle_step = charge_cycle_step # 充电步长越大，触发时间越短
            fill_cnt = int(click_cnt_*charge_cycle_step)
            if fill_cnt < 360:
                cv2.ellipse(img,pt_,(16,16),0,0,fill_cnt,(255,255,0),2)
            else:
                cv2.ellipse(img,pt_,(16,16),0,0,fill_cnt,(0,150,255),4)
            # 充电环未充满，置为 False
            if fill_cnt<360:
                flag_click_stable = False
        else:
            flag_click_stable = False
    return flag_click_stable

'''
    绘制手click 状态时的大拇指和食指中心坐标点轨迹
'''
def draw_click_lines(img,gesture_lines_dict,vis = False):
    # 绘制点击使能后的轨迹
    if vis :
        for id_ in gesture_lines_dict.keys():
            if len(gesture_lines_dict[id_]["pts"]) >=2:
                for i in range(len(gesture_lines_dict[id_]["pts"])-1):
                    pt1 = gesture_lines_dict[id_]["pts"][i]
                    pt2 = gesture_lines_dict[id_]["pts"][i+1]
                    cv2.line(img,pt1,pt2,gesture_lines_dict[id_]["line_color"],2,cv2.LINE_AA)
