"""Created on Sun Nov 18 20:27:23 2018 @author: Omar.M.S.H
Smile detection """

#Importing the Libraries
import cv2

#creating a cascade object, using the XMl for the face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#creating a cascade object, using the XMl for the eyes
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#creating a cascade for the smile
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# defining the function that will do the detection
#input the image in [black and white] & original image. 
#start
def detectsmile(gray , frame):
    # We apply the detectMultiScale method from the face cascade 
    # to locate one or several faces in the image.
    # FACES ARE TUPLES OF 4 ELEMENTS
    #detectMultiScale method from the face cascade to locate faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #for to iterate throught the faces 
    for (x , y , w ,h ) in faces:
        
        #rectangle around the face.
        cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,1),5)
        
        # the region of interest in the black and white image.
        roi_gray = gray[y:y+h,x:x+w]
        
        # get the region of interest in the colored image.
        roi_color = frame[y:y+h,x:x+w] 
        
        # detectMultiScale method eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        # For each detected eye:
        for (ex , ey , ew ,eh ) in eyes:
            #Draw the rectangle around every eye
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0),2)
        
        #Try and error with the Parameters until you'll find that '1.7' is the best for the Scalling Factor
        #Try and error with the Parameters until you'll find that '20' is the best for number of neighbors
        smiles = smile_cascade.detectMultiScale(roi_gray,1.7,20)
        for (sx , sy , sw ,sh ) in smiles:
            #Draw the rectangle around every eye
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh), (0,0,255),2)
            
    return frame  # We return the image with the detector rectangles.
#end of function


#Face Recogition with the WebCam.
video_capture = cv2.VideoCapture(0)
while True:
    #Read returns two elements i only want one so use underscore
    _, frame = video_capture.read()

    #our detect method works with black and white
    #so GET THE AVG of BGR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #the last of the webcam and changed
    canvas = detectsmile(gray,frame)
    
    #Animation
    cv2.imshow('Video', canvas) #  display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.
        
#Turn off webcam
video_capture.release()

#destroy all
cv2.destroyAllWindows()

