from dataclasses import dataclass
import numpy as np
import cv2
import paddleclas
from pedestrain_attr_dict import *
import sys
sys.path.append('./ByteTrack/')
from yolox.tracker.byte_tracker import BYTETracker, STrack


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


class Detection(object):
    def __init__(self, ltrb, track_id, person_attr):
        self.track_id = track_id
        self.ltrb = None
        self.sex = ''
        self.age = ''
        self.front = '未知'
        self.has_glasses = '否'
        self.has_hat = '否'
        self.bag = '未知'
        self.upper = ''
        self.lower = ''
        self.boots = '否'

        self.update(ltrb, person_attr)

    def update(self, ltrb, attr):
        self.ltrb = ltrb
        if attr is not None:
            self.sex = '女' if attr[0] == 'Female' else '男'
            self.age = age_dict[attr[1]]
            self.front = direct_list[attr[2]]
            self.has_glasses = '是' if attr[3] == 'Glasses: True' else '否'
            self.has_hat = '是' if attr[4] == 'Hat: True' else '否'
            self.bag = bag_dict[attr[6]]

            # 上半身
            self.upper = ' '.join([upper_dict[up] for up in attr[7].replace('Upper: ', '').split(' ')])
            # 下半身
            self.lower = ' '.join([lower_dict[lo] for lo in attr[8].replace('Lower:  ', '').split(' ')])
            self.boots = '是' if attr[9] == 'Boots' else '否'


class PedestrainTrack(object):
    def __init__(self):
        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        self.detection_dict = {}

        # 行人属性模型
        self.pedestrain_attr_model = paddleclas.PaddleClas(model_name="person_attribute")

    def update_track(self, boxes, frame):
        tracks = self.byte_tracker.update(
            output_results=boxes,
            img_info=frame.shape,
            img_size=frame.shape
        )

        new_detection_dict = {}
        for track in tracks:
            l, t, r, b = track.tlbr.astype(np.int32)
            track_id = track.track_id
            # 调用行人检测模型，识别行人属性
            track_box = frame[t:b, l:r]
            # print(track_box.shape)
            # cv2.imwrite('a.jpg', track_box)
            person_attr_res = self.pedestrain_attr_model.predict(track_box)
            attr = None
            try:
                for i in person_attr_res:
                    attr = i[0]['attributes']
            except:
                pass

            if track_id in self.detection_dict:
                detection = self.detection_dict[track_id]
                detection.update((l, t, r, b), attr)
            else:
                detection = Detection((l, t, r, b), track_id, attr)

            new_detection_dict[track_id] = detection

        self.detection_dict = new_detection_dict

        return self.detection_dict
