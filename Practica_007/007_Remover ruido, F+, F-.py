import cv2
import numpy as np

# Inicializar cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se puede acceder a la cámara.")
    exit()

# Crear kernel morfológico
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))

# Aplicar filtros morfológicos + blur
def aplicar_morfologia(mask):
    blur = cv2.GaussianBlur(mask, (5, 5), 0)  # Filtro lineal
    return {
        'Blur': blur,
        'TopHat': cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel),
        'BlackHat': cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel),
        'Dilatacion': cv2.dilate(blur, kernel, iterations=1),
        'Erosion': cv2.erode(blur, kernel, iterations=1),
        'Opening': cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel),
        'Closing': cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    }

# Etiquetar cada imagen con su nombre y redimensionarla a tamaño estándar
def etiquetar(img, texto, size=(320, 240)):
    img_resized = cv2.resize(img, size)
    img_color = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    cv2.putText(img_color, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return img_color

# Simular ruido de F+ y F−
def agregar_ruido_falsos(mask, color_id):
    fplus = mask.copy()
    fminus = mask.copy()
    cv2.circle(fplus, (100 + color_id * 60, 100), 10, 255, -1)
    cv2.circle(fplus, (200 + color_id * 60, 150), 10, 255, -1)
    cv2.rectangle(fminus, (300, 200), (320, 220), 0, -1)
    cv2.rectangle(fminus, (400, 250), (420, 270), 0, -1)
    return fplus, fminus

# Construir mosaico con etiquetas
def mosaico_color(diccionario):
    etiquetas = ['Blur', 'TopHat', 'BlackHat', 'Dilatacion', 'Erosion', 'Opening', 'Closing']
    etiquetadas = [etiquetar(diccionario[nombre], nombre) for nombre in etiquetas]
    fila1 = np.hstack(etiquetadas[:4])  # 4 imágenes
    fila2 = np.hstack(etiquetadas[4:])  # 3 imágenes
    # Agregar imagen negra para emparejar
    negro = np.zeros_like(etiquetadas[0])
    fila2 = np.hstack([fila2, negro])  # completar con imagen vacía
    return np.vstack((fila1, fila2))

# Redimensionar ventanas
def escalar(img, ancho=1280, alto=480):
    return cv2.resize(img, (ancho, alto))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Máscaras HSV por color
    mask_red = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
               cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask_green = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([70, 255, 255]))
    mask_blue = cv2.inRange(hsv, np.array([100, 150, 0]), np.array([140, 255, 255]))

    # Simular errores F+ y F−
    red_fplus, _ = agregar_ruido_falsos(mask_red, 0)
    _, green_fminus = agregar_ruido_falsos(mask_green, 1)
    blue_fplus, _ = agregar_ruido_falsos(mask_blue, 2)

    # Aplicar filtros
    red_fx = aplicar_morfologia(red_fplus)
    green_fx = aplicar_morfologia(green_fminus)
    blue_fx = aplicar_morfologia(blue_fplus)

    # Mostrar resultados en 4 ventanas
    cv2.imshow("Original", cv2.resize(frame, (960, 540)))
    cv2.imshow("Rojo - F+ (con filtros)", escalar(mosaico_color(red_fx)))
    cv2.imshow("Verde - F− (con filtros)", escalar(mosaico_color(green_fx)))
    cv2.imshow("Azul - F+ (con filtros)", escalar(mosaico_color(blue_fx)))

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
