import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Cargar imagen ------------------
img = cv2.imread('watch.jpg', cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises
if img is None:
    print("Error: no se encontró la imagen 'watch.jpg'")
    exit()

# ------------------ Calcular histograma original ------------------
hist_orig = cv2.calcHist([img], [0], None, [256], [0, 256])

# ------------------ Ecualización del histograma ------------------
img_ecualizada = cv2.equalizeHist(img)

# ------------------ Calcular histograma de imagen ecualizada ------------------
hist_eq = cv2.calcHist([img_ecualizada], [0], None, [256], [0, 256])

# ------------------ Graficar todo en una sola ventana ------------------
plt.figure(figsize=(12, 8))

# Imagen original
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

# Histograma original
plt.subplot(2, 2, 2)
plt.plot(hist_orig, color='black')
plt.title('Histograma Original')
plt.xlabel('Intensidad de píxel')
plt.ylabel('Cantidad de píxeles')
plt.grid()

# Imagen ecualizada
plt.subplot(2, 2, 3)
plt.imshow(img_ecualizada, cmap='gray')
plt.title('Imagen Ecualizada')
plt.axis('off')

# Histograma ecualizado
plt.subplot(2, 2, 4)
plt.plot(hist_eq, color='black')
plt.title('Histograma Ecualizado')
plt.xlabel('Intensidad de píxel')
plt.ylabel('Cantidad de píxeles')
plt.grid()

plt.tight_layout()
plt.show()
