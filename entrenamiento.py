import cv2
import os
import numpy as np

# carpeta con fotos para entrenar
dataPath = './data'
peopleList = os.listdir(dataPath)
print ('Lista de personas ', peopleList)

# inicio de variables
labels = []
facesData = []
label = 0

print (' Leyendo las imagenes ...')
for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))
    label = label + 1
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#Entrenar el modelo
print ('Entrenando modelo ...')
face_recognizer.train(facesData, np.array(labels))
face_recognizer.write('modelo.xml')
print("Modelo Entrenado")
