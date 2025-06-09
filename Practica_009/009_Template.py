import cv2
import numpy as np

# -------------------- CARGA DE IMÁGENES --------------------
img = cv2.imread("Ases.jpg")
template = cv2.imread("Corazon.jpg")

if img is None or template is None:
    print("❌ Error al cargar las imágenes.")
    exit()

h, w = template.shape[:2]

# -------------------- TEMPLATE MATCHING --------------------
# Convertimos a escala de grises para mejorar la precisión
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Aplicar template matching
resultado = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)

# Umbral mínimo de similitud
umbral = 0.85
loc = np.where(resultado >= umbral)

# -------------------- DIBUJAR RECTÁNGULOS DE DETECCIÓN --------------------
detectados = 0
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0, 255, 0), 3)
    detectados += 1

# -------------------- MOSTRAR RESULTADOS --------------------
print(f"✔ ROI detectados con umbral ≥ {umbral}: {detectados}")
cv2.imshow("Detección con Template Matching", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
