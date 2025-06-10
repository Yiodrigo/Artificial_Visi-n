import cv2
import numpy as np

# ---------------------- CONFIGURACIÓN ----------------------

# Cambia el nombre de la imagen por la que desees probar
nombre_imagen = "Mosaico.jpg"  # Puedes cambiar a: puerta.jpg, ventana.jpg, arquitectura.jpg

# ---------------------- CARGAR IMAGEN ----------------------

img = cv2.imread(nombre_imagen)
if img is None:
    print(f"Error: No se pudo cargar la imagen '{nombre_imagen}'")
    exit()

# Copia para trabajo
img_copia = img.copy()

# ---------------------- SELECCIÓN DE ROI ----------------------

print("Selecciona el ROI (objeto a conservar) con el mouse y presiona ENTER o ESPACIO.")
x, y, w, h = cv2.selectROI("Seleccionar ROI", img, False, False)
cv2.destroyWindow("Seleccionar ROI")

# Crear máscara negra del tamaño de la imagen
mascara = np.zeros_like(img)

# Copiar solo el ROI a la máscara (el resto queda negro)
mascara[y:y+h, x:x+w] = img_copia[y:y+h, x:x+w]

# Convertir ROI a escala de grises para detección de esquinas
roi_gris = cv2.cvtColor(mascara[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
roi_gris = np.float32(roi_gris)

# ---------------------- DETECCIÓN DE ESQUINAS ----------------------

esquinas = cv2.goodFeaturesToTrack(roi_gris, maxCorners=100, qualityLevel=0.01, minDistance=10)

# Dibujar esquinas sobre la imagen con fondo negro
if esquinas is not None:
    for esquina in esquinas:
        cx, cy = esquina.ravel()
        cv2.circle(mascara[y:y+h, x:x+w], (int(cx), int(cy)), 4, (0, 255, 0), -1)

# ---------------------- MOSTRAR RESULTADO ----------------------

cv2.imshow("Imagen con fondo removido y esquinas detectadas", mascara)
cv2.waitKey(0)
cv2.destroyAllWindows()
