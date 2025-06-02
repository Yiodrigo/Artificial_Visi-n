import cv2
import numpy as np

# Cargar imagen principal y template
img = cv2.imread('mesa.jpg')
template = cv2.imread('moneda.jpg')
h, w = template.shape[:2]

# Ejecutar template matching
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# Encontrar coincidencias con confianza >= 0.85
umbral = 0.85
loc = np.where(res >= umbral)

# Dibujar rectángulos donde hubo coincidencias
for pt in zip(*loc[::-1]):  # invertir coordenadas
    cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0,255,0), 2)

cv2.imshow("Coincidencias", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
