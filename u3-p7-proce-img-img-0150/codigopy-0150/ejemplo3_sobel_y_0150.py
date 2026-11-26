# =====================================================
# EJEMPLO 3: SOBEL Y EN EL DOMINIO DE LA FRECUENCIA
# Zavala Medina | NC= 0150
# =====================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from funciones_frecuencia_0150 import (
    obtener_espectro,
    aplicar_filtro_frecuencia,
    normalizar_visualizacion,
    guardar_grafica
)

# 1. Cargar imagen local en escala de grises
imagen = cv2.imread("image copy 2.png", cv2.IMREAD_GRAYSCALE)

if imagen is None:
    raise FileNotFoundError("No se encontró 'image copy 2.png'. Asegúrate de estar en la carpeta 'u3-p7-proce-img-0150'.")

# 2. Crear kernel Sobel Y (Detecta cambios verticales -> resalta bordes horizontales)
sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# 3. Aplicar Sobel Y en el dominio de la frecuencia
resultado_y = aplicar_filtro_frecuencia(imagen, sobel_y)

# 4. Calcular el espectro de la respuesta filtrada
espectro_y = obtener_espectro(resultado_y)

# 5. Normalizar la respuesta para visualizarla
imagen_y_normalizada = normalizar_visualizacion(resultado_y)

# 6. Mostrar resultados
figura, ejes = plt.subplots(1, 3, figsize=(15, 5))

ejes[0].imshow(imagen, cmap="gray")
ejes[0].set_title("Imagen Original 0150")
ejes[0].axis("off")

ejes[1].imshow(espectro_y, cmap="gray")
ejes[1].set_title("Espectro después de Sobel Y 0150")
ejes[1].axis("off")

ejes[2].imshow(imagen_y_normalizada, cmap="gray")
ejes[2].set_title("Respuesta Sobel Y (Bordes Horizontales) 0150")
ejes[2].axis("off")

plt.tight_layout()
guardar_grafica(figura, "03_image_copy_2_sobel_y.png")
plt.show()