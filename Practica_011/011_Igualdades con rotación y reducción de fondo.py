import cv2
import numpy as np

# Cargar imagen de escena (rotada o no) y template recortado
img_scene = cv2.imread("Ases.jpg")                  # Escena original
img_template = cv2.imread("Corazon.jpg")             # Template recortado

if img_scene is None or img_template is None:
    print("Error al cargar las imágenes.")
    exit()

# Convertir a escala de grises
gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

# ORB detector
orb = cv2.ORB_create(nfeatures=2000)

# Detectar keypoints y descriptores
kp1, des1 = orb.detectAndCompute(gray_template, None)
kp2, des2 = orb.detectAndCompute(gray_scene, None)

# Emparejamiento usando Hamming (ideal para ORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des1, des2, k=2)

# Filtro Lowe para buenas coincidencias
buenas = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        buenas.append(m)

# Si hay suficientes puntos clave, calcular homografía
if len(buenas) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in buenas]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in buenas]).reshape(-1, 1, 2)

    # Homografía usando RANSAC (permite tolerancia a errores)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0)

    # Obtener esquinas del template y transformarlas a la imagen original
    h, w = gray_template.shape
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    destino = cv2.perspectiveTransform(pts, M)

    # Dibujar contorno sobre la figura detectada
    img_detectada = cv2.polylines(img_scene.copy(), [np.int32(destino)], True, (0, 255, 0), 3)
    mensaje = f"Figura detectada con rotacion ({len(buenas)} coincidencias)"
else:
    img_detectada = img_scene.copy()
    mensaje = "No se detecto la figura (muy pocas coincidencias)"

# Mostrar mensaje y resultado
cv2.putText(img_detectada, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Resultado con soporte a rotacion", img_detectada)
cv2.waitKey(0)
cv2.destroyAllWindows()
