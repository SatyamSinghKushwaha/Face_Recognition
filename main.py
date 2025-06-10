import cv2

# webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640) #width
cap.set(4, 480) #height

#background img
imgBackgroud = cv2.imread('Resources/background.png')

while True:
    success, img = cap.read()

    #overlay camera on background img 162 , 55 are starting points (H ,W)
    imgBackgroud[162:162+480,55:55+640] = img

    cv2.imshow('Face Attendance', imgBackgroud)
    cv2.waitKey(1)