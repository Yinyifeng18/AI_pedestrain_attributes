import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from track import PedestrainTrack


def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), text_size=30):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("./fonts/simsun.ttc", text_size, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, text_color, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class PedestrainAttrDetection(object):
    def __init__(self):
        self.yolo_model = torch.hub.load('yolov5',
                                         'custom',
                                         path='./yolov5/weights/yolov5s.pt',
                                         source='local')
        self.yolo_model.conf = 0.6
        self.tracker = PedestrainTrack()

    def plot_detection(self, person_track_dict, frame):
        for track_id, detection in person_track_dict.items():
            l, t, r, b = detection.ltrb
            track_id = detection.track_id

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 1)
            cv2.putText(frame, f'id-{track_id}', (l + 2, t - 3), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            top_margin = 20
            font_size = 14
            font_color = (255, 0, 255)
            frame = cv2_add_chinese_text(frame, f'性别：{detection.sex}', (l+2, t+top_margin), font_color, font_size)
            frame = cv2_add_chinese_text(frame, f'年龄：{detection.age}', (l+2, t+top_margin*2), font_color, font_size)
            frame = cv2_add_chinese_text(frame, f'朝向：{detection.front}', (l+2, t+top_margin*3), font_color, font_size)
            frame = cv2_add_chinese_text(frame, f'戴眼镜：{detection.has_glasses}', (l+2, t+top_margin*4), font_color, font_size)
            frame = cv2_add_chinese_text(frame, f'戴帽子：{detection.has_hat}', (l+2, t+top_margin*5), font_color, font_size)
            frame = cv2_add_chinese_text(frame, f'背包：{detection.bag}', (l+2, t+top_margin*6), font_color, font_size)
            frame = cv2_add_chinese_text(frame, f'上半身：{detection.upper}', (l+2, t+top_margin*7), font_color, font_size)
            frame = cv2_add_chinese_text(frame, f'下半身：{detection.lower}', (l+2, t+top_margin*8), font_color, font_size)
            frame = cv2_add_chinese_text(frame, f'穿靴：{detection.boots}', (l+2, t+top_margin*9), font_color, font_size)
            # cv2.putText(frame, f'sex-{sex}', (l + 2, t + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        return frame

    @staticmethod
    def yolo_pd_to_numpy(yolo_pd):
        box_list = yolo_pd.to_numpy()
        detections = []
        for box in box_list:
            l, t = int(box[0]), int(box[1])
            r, b = int(box[2]), int(box[3])

            conf = box[4]

            detections.append([l, t, r, b, conf])
        return np.array(detections, dtype=float)

    def detect(self, video_file):
        cap = cv2.VideoCapture(video_file)
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        print(fps)
        # video_writer = cv2.VideoWriter('./video_p2.mp4', cv2.VideoWriter_fourcc(*'H264'), fps, (video_w, video_h))

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            yolo_det_results = self.yolo_model(frame[:, :, ::-1])
            pd = yolo_det_results.pandas().xyxy[0]
            person_pd = pd[pd['name'] == 'person']

            person_det_boxes = self.yolo_pd_to_numpy(person_pd)
            person_track_dict = self.tracker.update_track(person_det_boxes, frame)
            frame = self.plot_detection(person_track_dict, frame)

            cv2.imshow('pedestrain attributes detect', frame)
            
            # video_writer.write(frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                return


if __name__ == '__main__':
    PedestrainAttrDetection().detect('./video.mp4')