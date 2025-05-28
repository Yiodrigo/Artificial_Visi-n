import cv2
import numpy as np

# Inicializar la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar para mejor rendimiento
    frame = cv2.resize(frame, (640, 480))

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------- Detección de Bordes --------------------

    # Laplaciano
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Sobel X
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobelx = cv2.convertScaleAbs(sobelx)

    # Sobel Y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobely = cv2.convertScaleAbs(sobely)

    # Canny
    canny = cv2.Canny(gray, 100, 200)

    # -------------------- Mostrar resultados --------------------

    cv2.imshow("Original", frame)
    cv2.imshow("Laplaciano", laplacian)
    cv2.imshow("Sobel X", sobelx)
    cv2.imshow("Sobel Y", sobely)
    cv2.imshow("Canny", canny)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
