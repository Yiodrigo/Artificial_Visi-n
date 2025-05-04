import cv2
import numpy as np

# ------------------- Captura de video -------------------
cap = cv2.VideoCapture(0)  # Captura de la cámara

if not cap.isOpened():
    print("Error: No se puede acceder a la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video.")
        break

    # Redimensionar para que sea más rápido el procesamiento
    frame = cv2.resize(frame, (640, 480))

    # ------------------- Filtros en espacio HSV -------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango de color rojo
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red = cv2.add(mask_red1, mask_red2)

    # Rango de color verde
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Rango de color azul
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Aplicar las máscaras para extraer los colores
    res_red = cv2.bitwise_and(frame, frame, mask=mask_red)
    res_green = cv2.bitwise_and(frame, frame, mask=mask_green)
    res_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)

    # ------------------- Filtro tipo Green Screen -------------------
    # Eliminamos el fondo verde
    mask_inv_green = cv2.bitwise_not(mask_green)
    green_removed = cv2.bitwise_and(frame, frame, mask=mask_inv_green)

    # ------------------- Filtros en espacio YUV (solo para visualización) -------------------
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # ------------------- Mostrar Resultados -------------------
    cv2.imshow('Video Original', frame)
    cv2.imshow('Filtro Rojo', res_red)
    cv2.imshow('Filtro Verde', res_green)
    cv2.imshow('Filtro Azul', res_blue)
    cv2.imshow('Green Screen (Verde Eliminado)', green_removed)
    cv2.imshow('Video en YUV', yuv)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
