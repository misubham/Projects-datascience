import  cv2
import numpy as np
print("subham mittal,project human face dectected and played video with vlc")
face_detection_model=cv2.CascadeClassifier("C:/Users/Subham/Desktop/Learn N Build/Image processing/Assignments/haarcascade_frontalface_default.xml")
v=cv2.VideoCapture(0)
while True:
    _,image=v.read()
    #print(image)
    image=cv2.resize(image,(600,600))
    D_face_edges=face_detection_model.detectMultiScale(image,1.3,7)
    print(len(D_face_edges))
    if len(D_face_edges)>0:
        for (x,y,w,h) in D_face_edges:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    image=np.flip(image,1)
    key=cv2.waitKey(1)
    if (len(D_face_edges)==1):
        video=cv2.VideoCapture("C:/Users/Subham/Desktop/Learn N Build/Image processing/Assignments/A. R.mp4")
        while(video.isOpened()):
            ret,frame=video.read()
            frame=cv2.resize(frame,(800,400))
            cv2.imshow("video",frame)
            ch=cv2.waitKey(1)
            if(ch==ord('q')):
                break
        cv2.destroyAllWindows()
        v.release()
        break
    cv2.imshow("Detected face",image)
