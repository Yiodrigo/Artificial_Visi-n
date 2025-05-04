import cv2

# Cargar la imagen a color
img = cv2.imread('watch.jpg')

# Verificar que la imagen se haya cargado correctamente
if img is None:
    raise IOError("No se pudo cargar la imagen 'watch.jpg'")

# Dibujar un rectángulo (marcando una región de interés visualmente)
cv2.rectangle(img, (100, 50), (200, 150), (0, 255, 0), 2)

# Escribir texto sobre la imagen
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Reloj', (105, 45), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

# Definir una Región de Interés (ROI)
roi = img[50:150, 100:200]  # Y: 50 a 150, X: 100 a 200

# Pegar esa ROI en otra parte de la imagen (arriba a la izquierda)
# Nota: verificar que las dimensiones sean válidas para evitar errores
if roi.shape[0] <= img.shape[0] and roi.shape[1] <= img.shape[1]:
    img[0:100, 0:100] = cv2.resize(roi, (100, 100))  # redimensionamos para encajar

# Mostrar la imagen final
cv2.imshow('Imagen con dibujo y ROI', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
