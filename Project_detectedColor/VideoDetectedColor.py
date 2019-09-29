# importing libraries 
import cv2 
import numpy as np
from imutils import contours
import imutils
   
# Create a VideoCapture object and read from input file
backsub = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture('video-1569690261.mp4')
sen = 0
minArea=1
lineCount = 500
counter = 0
   
# Check if camera opened successfully 
if (cap.isOpened()== False):  
  print("Error opening video  file") 
   
# Read until video is completed 
while(cap.isOpened()): 
      
  # Capture frame-by-frame 
  ret, frame = cap.read() 
  if ret == True:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape, dtype=np.uint8)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    # COLOR_MIN = np.array([0,50,50],np.uint8)      
    # COLOR_MAX = np.array([20,255,255],np.uint8)

    #กำหนดชิ้นงานที่จะนับตามสี
    # frame_threshed = cv2.inRange(image, COLOR_MIN, COLOR_MAX)
    # imgray = frame_threshed
    # cv2.imshow('Frame2', imgray)

    # lowwer led
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    #upper red
    lower_red2 = np.array([170,50,50])
    upper_red2 = np.array([180,255,255])
    mask = cv2.inRange(image, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    mask2 = cv2.inRange(image, lower_red2, upper_red2)
    res2 = cv2.bitwise_and(frame,frame, mask= mask2)
    mask = mask+mask2
    img3 = res+res2
    # cv2.imshow('img3', img3)

    ret,thresh = cv2.threshold(mask,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
          x,y,w,h = cv2.boundingRect(cnt)
          cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #นับจำนวนชิ้นงาน
    fgmask = backsub.apply(img3, None, 0.01)
    # cv2.imshow("blacksup",fgmask)
    erode=cv2.erode(fgmask,None,iterations=3)
    moments=cv2.moments(erode,True)         
    area=moments['m00']

    if moments['m01'] >=minArea:
            x=int(moments['m10']/moments['m00'])
            y=int (moments['m01']/moments['m00'])
            if y < lineCount:
                sen=sen << 1
            else:
                sen=(sen<<1)|1
            sen=sen&0x03
            if sen == 1:
                counter=counter+1
       

    cv2.line(frame,(0,lineCount),(500,lineCount),(0,100,0),2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Count Red = '+str(counter), (10,30),font,1, (255, 0, 0), 2)

    # Display the resulting frame 
    cv2.imshow('Frame', frame)
   
    # Press Q on keyboard to  exit 
    if cv2.waitKey(25) & 0xFF == ord('q'): 
      break
   
  # Break the loop 
  else:  
    break
   
# When everything done, release  
# the video capture object 
cap.release() 
   
# Closes all the frames 
cv2.destroyAllWindows() 
