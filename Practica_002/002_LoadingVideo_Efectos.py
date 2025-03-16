import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Filtros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    edges = cv2.Canny(frame, 100, 200)
    inverted = cv2.bitwise_not(frame)
    mirror = cv2.flip(frame, 1)

    # Mostrar imágenes
    cv2.imshow('Original', frame)
    cv2.imshow('Escala de Grises', gray)
    cv2.imshow('Desenfoque', blur)
    cv2.imshow('Bordes', edges)
    cv2.imshow('Negativo', inverted)
    cv2.imshow('Espejo', mirror)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
