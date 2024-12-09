import cv2
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
nameID = str(input("Ingresa el nombre: "))
path = 'valVGGface2/' + nameID

isExist = os.path.exists(path)
if isExist:
    print("Nombre ya existe")
    nameID=str(input("Ingresa otro nombre: "))
else:
    os.makedirs(path)

while True:
    ret, frame = video.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        count = count + 1
        name = './valVGGface2/' + nameID + '/' + str(count) + '.jpg'
        print("Creando imagenes..." + name)
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imshow("webcam", frame)
    cv2.waitKey(1)
    if count >= 400: #400 imagenes
        break
video.release()
cv2.destroyAllWindows()
