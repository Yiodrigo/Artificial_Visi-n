import cv2

# Captura desde cámara
cap = cv2.VideoCapture(0)
ret, frame_ref = cap.read()
if not ret:
    print("No se pudo acceder a la camara.")
    cap.release()
    exit()

# Convertir el primer frame en escala de grises (referencia)
frame_ref_gray = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
frame_ref_gray = cv2.GaussianBlur(frame_ref_gray, (21, 21), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Comparar con el fondo (primer frame)
    delta = cv2.absdiff(frame_ref_gray, gray)
    _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)

    # Mejorar la máscara
    thresh = cv2.dilate(thresh, None, iterations=2)
    mask_movimiento = cv2.bitwise_and(frame, frame, mask=thresh)

    # Mostrar resultados
    cv2.imshow("Original", frame)
    cv2.imshow("Movimiento Detectado", mask_movimiento)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
