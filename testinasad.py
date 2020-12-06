import numpy as np
import cv2
import time
#import pafy
#cap = cv2.VideoCapture('test.mp4')
car=cv2.CascadeClassifier("cars.xml")


def ProcessedFrame(frame):
    grayed=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    average=np.mean(grayed)
    ret,thresh = cv2.threshold(grayed,average+30,255,cv2.THRESH_BINARY)
    return thresh

def DetectedVehicle(frame,classifier):
    grayed=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars=car.detectMultiScale(grayed,1.1,1)
    thresh=ProcessedFrame(frame)
    for (x,y,w,h) in cars:
        if thresh[y][x]== 255 and thresh[y+h][x+w]==255 and list(thresh[y][x:x+w])!=[255]*len(list(thresh[y][x:x+w])) and h>25:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame, 'Vehicle', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            continue
        if  len(thresh[y:y+h][x:x+w])==0:
            continue
        unique, counts=np.unique(thresh[y:y+h][x:x+w], return_counts=True)
        if len(counts)==1 and unique==0:
            continue
        elif len(counts)==1:
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, 'Vehicle', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            continue
        if counts[1]>counts[0] and w*h<11000:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
            cv2.putText(frame, 'Vehicle', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            
# url="https://www.youtube.com/watch?v=RfmPtePohho"
# video=pafy.new(url)
# best=video.getbest(preftype="mp4")            
# cap = cv2.VideoCapture(0)
# cap.open(best.url)
while(cap.isOpened()):
    ret, frame = cap.read()
    try:
        height,width,dimension=frame.shape
        cop=frame.copy()
    except:
        break
    thresh=ProcessedFrame(frame)
    DetectedVehicle(frame,car)
    grayed=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars=car.detectMultiScale(grayed,1.1,1)
    for (x,y,w,h) in cars:
            cv2.rectangle(cop,(x,y),(x+w,y+h),(255,255,0),2)
    cv2.imshow("cardet",frame)
    cv2.imshow("thesh",thresh)
    cv2.imshow("theshads",cop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.5)
cap.release()
cv2.destroyAllWindows()