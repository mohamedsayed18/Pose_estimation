import cv2
import time

cap = cv2.VideoCapture(0)
file_name = "Data_set/Straw_pose/"
county = 0
while(True):
    print("change pose")
    time.sleep(2)
    # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.imwrite(file_name +"img"+str(county)+".jpg", frame)

    county+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

