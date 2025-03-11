import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener el centro de la imagen
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)

    # Matriz de transformación para rotar 45°
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h))

    cv2.imshow('Rotado 45°', rotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
