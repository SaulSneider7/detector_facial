import cv2
import os
from camera import getcamera

dataPath = './data'
imagePath = os.listdir(dataPath)
print ('imagePaths', imagePath)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modelo.xml')
faceClassif = cv2.CascadeClassifier("rostros.xml")

#Abrimos la camara
camera = getcamera()
cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

while True:
    # Toma una fotografia
    ret, frame = cap.read()

    if not ret:
        break

    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    for(x,y,w,h) in faces:
        # Obtener el recuadro del rostro
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150))
        #Utilizamos el modelo entrenado
        result = face_recognizer.predict(rostro)

        if result[1] < 75:
            cv2.putText(frame, '{}'.format(imagePath[result[0]]), (x,y - 25), 2, 1, (0, 255, 0))
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x,y - 20), 2, 1, (0, 0, 255))
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Mi Camara con Python', frame)

    if cv2.waitKey(1) & 0xFFF == ord('q'):
        break


cv2.destroyAllWindows()
cap.release()