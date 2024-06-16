#Se importan las librerias necesarias
import cv2
import numpy as np

def detect_tennis_ball(image_path):
    # Cargamos la imagen
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Error al cargar la imagen.")
        return

    # Convertimos la imagen a espacio de color HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definimos el rango de colores para la pelota de tenis
    lower_green = np.array([30, 100, 100])
    upper_green = np.array([70, 255, 255])

    # Creamos una máscara con los colores dentro del rango
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Aplicamos la máscara a la imagen
    res = cv2.bitwise_and(img, img, mask=mask)

    # Convertimos la imagen a escala de grises
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Aplicamos un desenfoque gaussiano para reducir el ruido
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # Aplicamos el algoritmo de Canny para la detección de bordes
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Usamos la transformada de Hough para detectar círculos
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=300, param2=10, minRadius=10, maxRadius=59)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Filtramos los círculos detectados para encontrar el más grande
        max_radius = 0
        best_circle = None
        for i in circles[0, :]:
            if i[2] > max_radius:
                max_radius = i[2]
                best_circle = i

        # Dibujamos solo el mejor círculo
        if best_circle is not None:
            cv2.circle(img, (best_circle[0], best_circle[1]), best_circle[2], (0, 255, 0), 2)
            cv2.circle(img, (best_circle[0], best_circle[1]), 2, (0, 0, 255), 3)

    # Mostramos la imagen con los círculos detectados
    cv2.imshow('Detected Circles', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Llamamos a la función con la ruta de la imagen
detect_tennis_ball('tenis-ladrillo.jpg')