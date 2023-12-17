import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from .tracker import Tracker
import os

def v_count(vc=0):
    print(vc)
    return vc

def process_video_stream(video_url):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model = YOLO('yolov8s.pt')
    stream = CamGear(source=video_url, stream_mode=True, logging=True).start()

    def RGB(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            colorsBGR = [x, y]
            print(colorsBGR)

    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)

    coco_file_path = os.path.join(script_dir, 'coco.txt')
    print(f"File path: {coco_file_path}")

    with open(coco_file_path, "r") as my_file:
        data = my_file.read()
    class_list = data.split("\n")

    tracker = Tracker()
    area1 = [(752, 263), (414, 381), (437, 396), (772, 272)]
    area2 = [(777, 275), (445, 403), (455, 416), (796, 279)]

    downcar = {}
    downcarcounter = []
    upcar = {}
    upcarcounter = []

    with open(".\out.txt", "w") as out_file:
        while True:
            frame = stream.read()
            if frame is None:
                break

            frame = cv2.resize(frame, (1020, 500))
            results = model.predict(frame)
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")

            car_list = []
            for index, row in px.iterrows():
                x1, y1, x2, y2, _, d = map(int, row[:6])
                c = class_list[d]
                if 'car' in c:
                    car_list.append([x1, y1, x2, y2])

            bbox_idx = tracker.update(car_list)

            for bbox in bbox_idx:
                x3, y3, x4, y4, id1 = bbox
                cx = (x3 + x4) // 2
                cy = (y3 + y4) // 2

                result = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
                if result >= 0:
                    downcar[id1] = (cx, cy)
                    downcarcounter.append(id1)

                result2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
                if result2 >= 0:
                    upcar[id1] = (cx, cy)
                    upcarcounter.append(id1)

                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
                cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)

            cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 1)
            cvzone.putTextRect(frame, f'RightLane: {len(downcarcounter)}', (50, 60), 1, 1)
            cvzone.putTextRect(frame, f'LeftLane: {len(upcarcounter)}', (846, 59), 1, 1)
            cv2.imshow("RGB", frame)
     
            if (len(downcarcounter)) == 2 or cv2.waitKey(1) & 0xFF == 27:
                break

    stream.stop()
    cv2.destroyAllWindows()
