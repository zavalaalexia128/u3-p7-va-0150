# =====================================================
# EJEMPLO 4: MAGNITUD DEL GRADIENTE (image copy 3.png)
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
imagen = cv2.imread("image copy 3.png", cv2.IMREAD_GRAYSCALE)

if imagen is None:
    raise FileNotFoundError("No se encontró 'image copy 3.png'. Asegúrate de estar en la carpeta 'u3-p7-proce-img-0150'.")

# 2. Definir kernels
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

# 3. Aplicar ambos filtros en frecuencia
resultado_x = aplicar_filtro_frecuencia(imagen, sobel_x)
resultado_y = aplicar_filtro_frecuencia(imagen, sobel_y)

# 4. Calcular la magnitud del gradiente: G = sqrt(Gx^2 + Gy^2)
# Nota: Usamos los resultados crudos (antes de normalizar) para la operación matemática correcta
magnitud_gradiente = np.sqrt(resultado_x**2 + resultado_y**2)

# 5. Normalizar todas las respuestas para visualización
img_x_norm = normalizar_visualizacion(resultado_x)
img_y_norm = normalizar_visualizacion(resultado_y)
img_mag_norm = normalizar_visualizacion(magnitud_gradiente)

# 6. Calcular espectros para visualización comparativa
espectro_x = obtener_espectro(resultado_x)
espectro_y = obtener_espectro(resultado_y)

# 7. Mostrar los 6 resultados en una cuadrícula de 2x3
figura, ejes = plt.subplots(2, 3, figsize=(18, 10))

# Fila 1: Imagen original y Espectros
ejes[0, 0].imshow(imagen, cmap="gray")
ejes[0, 0].set_title("1. Imagen Original 0150")
ejes[0, 0].axis("off")

ejes[0, 1].imshow(espectro_x, cmap="gray")
ejes[0, 1].set_title("2. Espectro Sobel X 0150")
ejes[0, 1].axis("off")

ejes[0, 2].imshow(espectro_y, cmap="gray")
ejes[0, 2].set_title("3. Espectro Sobel Y 0150")
ejes[0, 2].axis("off")

# Fila 2: Respuestas espaciales y Magnitud
ejes[1, 0].imshow(img_x_norm, cmap="gray")
ejes[1, 0].set_title("4. Respuesta Sobel X (Verticales) 0150")
ejes[1, 0].axis("off")

ejes[1, 1].imshow(img_y_norm, cmap="gray")
ejes[1, 1].set_title("5. Respuesta Sobel Y (Horizontales) 0150")
ejes[1, 1].axis("off")

ejes[1, 2].imshow(img_mag_norm, cmap="gray")
ejes[1, 2].set_title("6. Magnitud del Gradiente (Todos los bordes) 0150")
ejes[1, 2].axis("off")

plt.tight_layout()
guardar_grafica(figura, "04_image_copy_3_magnitud_gradiente.png")
plt.show()
print("Zavala Medina | NC= 0150")