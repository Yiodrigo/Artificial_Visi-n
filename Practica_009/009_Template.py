import cv2
import numpy as np
import os

# -------------------- CARGA DE IMÁGENES --------------------
# Usamos nombres relativos (mismo directorio)
img_scene = cv2.imread("Ases.jpg")
img_template = cv2.imread("Corazon.jpg")

if img_scene is None or img_template is None:
    print("❌ Error al cargar las imágenes. Asegúrate de que 'Ases.jpg' y 'Corazon.jpg' estén en la misma carpeta.")
    exit()

# Convertir a escala de grises
gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

# -------------------- DETECCIÓN ORB --------------------
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(gray_template, None)
kp2, des2 = orb.detectAndCompute(gray_scene, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# -------------------- FILTRAR BUENAS COINCIDENCIAS --------------------
buenas = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        buenas.append(m)

# -------------------- HOMOGRAFÍA Y DIBUJO --------------------
if len(buenas) >= 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in buenas]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in buenas]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = gray_template.shape
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    destino = cv2.perspectiveTransform(pts, M)

    img_resultado = cv2.polylines(img_scene.copy(), [np.int32(destino)], True, (0, 255, 0), 3)
    mensaje = f"Coincidencias: {len(buenas)}"
else:
    img_resultado = img_scene.copy()
    mensaje = "No se encontro el Corazon"

# -------------------- MOSTRAR RESULTADO --------------------
cv2.putText(img_resultado, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Detección de Corazon con ORB", img_resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()
