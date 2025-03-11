import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Puntos originales (esquinas de la imagen)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # Puntos transformados (deformamos la perspectiva)
    pts2 = np.float32([[50, 50], [w-50, 0], [0, h-50], [w, h]])

    # Matriz de transformación
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(frame, M, (w, h))

    cv2.imshow('Perspectiva Inclinada', warped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
