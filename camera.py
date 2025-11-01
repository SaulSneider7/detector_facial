import cv2


# Detecta qué camara esta disponible
def getcamera():
    for i in range(8):
        c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if c is None or not c.isOpened():
            print('Warning: unable to open video source: ', i)
        else:
            print('Camera found in video source: ', i)
            c.release()
            return i
