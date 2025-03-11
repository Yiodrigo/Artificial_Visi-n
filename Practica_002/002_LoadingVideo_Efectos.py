import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar detección de bordes estilo "cómic"
    #edges = cv2.Canny(gray, 100, 200)
    #cv2.imshow('Efecto Cómic', edges)
    #Sepia
    #sepia_filter = np.array([[0.272, 0.534, 0.131], 
    #                     [0.349, 0.686, 0.168], 
    #                     [0.393, 0.769, 0.189]])
    #sepia = cv2.transform(frame, sepia_filter)
    #cv2.imshow('Efecto Sepia', sepia)
    # Invierte los colores
    #inverted = cv2.bitwise_not(frame)  
    #cv2.imshow('Negativo', inverted)
    #Escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Escala de Grises', gray)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
