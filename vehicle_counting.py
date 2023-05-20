
import cv2
import numpy as np

import math
import random
import time

import cardetect
from sort import *

def intersection_over_union(real,predict):
    x0=max(real[0],predict[0])
    y0=max(real[1],predict[1])
    x1=min(real[2],predict[2])
    y1=min(real[3],predict[3])
    
    interArea=max(0,x1-x0+1)*max(0,y1-y0+1)
    
    realArea=(real[2]-real[0]+1)*(real[3]-real[1]+1)
    predictArea=(predict[2]-predict[0]+1)*(predict[3]-predict[1]+1)
    
    iou=interArea/float(realArea+predictArea-interArea)
    
    return iou

colors=[]
for i in range(100):
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    colors.append([b,g,r])
    
mot_tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
local_tracks = []

video=cv2.VideoCapture('pexels-pixabay.mp4')

mask=np.zeros((200,935,3),dtype='uint8')
mask[:,:,1]=255

count=1
counted_ID=[]

prev_time = 0
new_time = 0

while video.isOpened():
    
    ret,frame=video.read()
    if not ret:
        break
    
    frame=cv2.resize(frame,(0,0),fx=.7,fy=.7)
    
    coordinates=cardetect.detect_car(frame,False) 
   
    track_bbs_ids=mot_tracker.update(np.asarray(coordinates))
      
    for track in reversed(track_bbs_ids):
        if local_tracks[int(track[4]) - 1:int(track[4])]:

            pr_x=local_tracks[int(track[4])-1][1][-1][0]
            pr_y=local_tracks[int(track[4])-1][1][-1][1]
            
            now_x=(track[0]+track[2])/2
            now_y=(track[1]+track[3])/2 
            d=math.sqrt((now_x - pr_x) ** 2 + (now_y - pr_y) ** 2)
            
            if d>300:
                continue
            
            local_tracks[int(track[4]) - 1][1].append((now_x, now_y))
            
        else:
            local_tracks.append([int(track[4]) - 1, [((track[0] + track[2]) / 2, (track[1] + track[3]) / 2)]])
        
        
            
        color_id=int(int(track[4])%100)
        color=colors[color_id]
            
        cv2.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), color, 2)

        roi=intersection_over_union([365,400,1300,600],[int(track[0]), int(track[1]), int(track[2]), int(track[3])])
        roi=int(roi*100)        
        
        if roi>3 and track[4] not in counted_ID:
            count=count+1
            counted_ID.append(track[4])
            
    cv2.putText(frame, f'car count:{str(count)}', (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    coupled_img=cv2.addWeighted(frame[400:600,365:1300,:],.6,mask,.4,0)
    frame[400:600,365:1300,:]=coupled_img
            
    new_time = time.time()
    fps = int(1/(new_time-prev_time))
    cv2.putText(frame, f'FPS:{str(fps)}', (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    prev_time = new_time
    
    cv2.imshow('frame',frame)
    key=cv2.waitKey(7)
    
    if key==ord('q'):
        break
    elif key==ord('w'):
        cv2.waitKey(0)

cv2.destroyAllWindows()
video.release()

