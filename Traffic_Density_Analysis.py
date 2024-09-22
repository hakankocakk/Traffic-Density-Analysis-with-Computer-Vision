import os
import time
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from ultralytics import YOLO
#from tensorflow.keras.models import load_model


class traffic_density_analysis():
    
    def __init__(self, model, all_classes, video):

        self.model = model
        self.all_classes = all_classes
        self.video = video

        self.detect_video()


    def process_image(self, img):
        image = cv2.resize(img, (1920, 1080),
                        interpolation=cv2.INTER_CUBIC)
        image = np.array(image, dtype='float32')
        return image
    
    
    def draw(self, image, boxes):
 
        countsag = 0
        countsol = 0
        countorta = 0
        Areasag = 1
        Areaorta = 1
        Areasol = 1

        cv2.circle(image, center=(965, 535), radius=255, color=(0, 0, 255), thickness=3)
        cv2.circle(image, center=(965, 535), radius=405, color=(0, 0, 255), thickness=3)
        cv2.rectangle(image, pt1=(0, 630), pt2=(415, 425), color=(0, 0, 255), thickness=3)
        cv2.rectangle(image, pt1=(1525, 635), pt2=(1920, 430), color=(0, 0, 255), thickness=3)
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            print(x1,y1,x2,y2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            side1 = x2 - x1
            side2 = y2 - y1


            #cv2.line(image, (415, 425), (855, 60), color=(0, 0, 255), thickness=3)
            #cv2.line(image, (1080, 1020), (1525, 635), color=(0, 0, 255), thickness=3)
            #cv2.line(image, (415, 630), (855, 1020), color=(0, 0, 255), thickness=3)
            #cv2.line(image, (1080, 60), (1525, 430), color=(0, 0, 255), thickness=3)

            if ((center_x >= 415) and (center_y <= ((-360/440) * (center_x - 415)) + 425)):
                color = (0, 255, 0)
            elif ((center_x >= 415) and (center_y >= ((390/440) * (center_x - 415)) + 630)):
                color = (0, 255, )
            elif ((center_x <= 1525) and (center_y >= ((-355/445) * (center_x - 1525)) + 635)):
                color = (0, 255, 0)
            elif ((center_x <= 1525) and (center_y <= ((370/445) * (center_x - 1525)) + 430)):
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            
    
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            scores = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            
            cv2.putText(image, '{0} {1:.2f}'.format(self.all_classes[cls], scores),
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 1,
                        cv2.LINE_AA)
            
            
            
            
            
            car_area = side1 * side2
            
            
            if ((center_x>=0 and center_x<=415) and (center_y<=630 and center_y>=425)):
                countsag += 1
                Areasag += car_area
                
            elif math.sqrt((center_x - 965)**2 + (center_y - 535)**2) <= 405:
                countorta+=1
                Areaorta += car_area
                
            elif ((center_x>=1525 and center_x<=1920) and (center_y<=635 and center_y>=440)):
                countsol += 1
                Areasol += car_area
                

            print('class: {0}, score: {1:.2f}'.format(self.all_classes[cls], scores))
            print('box coordinate x,y,w,h: {0}'.format(box))
            
        
        
        small_circle = 3.14 * 255 * 255
        big_circle = 3.14 * 405 * 405
        circle_area = big_circle - small_circle
        circle_density = 100 * Areaorta / circle_area
        cv2.rectangle(image, pt1=(820, 560), pt2=(1120, 500), color=(0, 0, 0), thickness=-1)
        cv2.putText(image, 'Number of Vehicles: {0}'.format(countorta),
                    (845, 520),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(image, 'Traffic density: %{0:.2f}'.format(circle_density),
                    (845, 550),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)
        
        
        rectangle_area_sag = 415 * 204
        rectangle_density_sag = 100 * Areasag / rectangle_area_sag
        cv2.rectangle(image, pt1=(50, 420), pt2=(350, 360), color=(0, 0, 0), thickness=-1)
        cv2.putText(image, 'Number of Vehicles: {0}'.format(countsag),
                    (75, 380),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(image, 'Traffic density: %{0:.2f}'.format(rectangle_density_sag),
                    (75, 410),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)
            
        
        rectangle_area_sol = 395 * 380
        rectangle_density_sol = 100 * Areasol / rectangle_area_sol
        cv2.rectangle(image, pt1=(1575, 425), pt2=(1875, 365), color=(0, 0, 0), thickness=-1)
        cv2.putText(image, 'Number of Vehicles: {0}'.format(countsol),
                    (1600, 385),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(image, 'Traffic density: %{0:.2f}'.format(rectangle_density_sol),
                    (1600, 415),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)
        
        #cv2.rectangle(image, pt1=(855, 1080), pt2=(1080, 1020), color=(0, 0, 255), thickness=3)
        #cv2.rectangle(image, pt1=(855, 0), pt2=(1080, 60), color=(0, 0, 255), thickness=3)
            
        print()


    def detect_image(self, image):
 
        pimage = self.process_image(image)

        start = time.time()
        results = self.model.predict(pimage, image.shape)
        end = time.time()

        print('time: {0:.2f}s'.format(end - start))
        

        for result in results:
            boxes = result.boxes
            
            filtered_boxes = [box for box in boxes if math.ceil((box.conf[0]*100))/100 >= 0.4]
            
            if filtered_boxes:
                self.draw(image, filtered_boxes)
                 
        return image
    

    def detect_video(self):

        video_path = os.path.join("videos", "test", self.video)
        camera = cv2.VideoCapture(video_path)
        cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

        sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mpeg')

        vout = cv2.VideoWriter()
        vout.open(os.path.join("videos", "res", self.video), fourcc, 20, sz, True)

        while True:
            res, frame = camera.read()

            if not res:
                break

            image = self.detect_image(frame)
            cv2.imshow("detection", image)

            vout.write(image)

            if cv2.waitKey(110) & 0xff == 27:
                    break

        vout.release()
        camera.release()
        cv2.destroyAllWindows()


def get_classes(file):
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names

def main():

    model = YOLO('runs/train/weights/best.pt')
    file = 'Data/voc_classes.txt'
    all_classes = get_classes(file)
    video = 'traffic8_new.mp4'

    traffic_density_analysis(model, all_classes, video)


if __name__ == '__main__':
    main()
        