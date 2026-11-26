# =====================================================
# EJEMPLO 1: IMAGEN ORIGINAL (image.png) Y ESPECTRO DE FRECUENCIA
# Zavala Medina | NC= 0150
# =====================================================
import cv2
import matplotlib.pyplot as plt
from funciones_frecuencia_0150 import (
    obtener_espectro,
    guardar_grafica
)

# 1. Cargar tu imagen local en escala de grises
imagen = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Verificación por si la ruta no coincide
if imagen is None:
    raise FileNotFoundError("No se encontró 'image.png'. Asegúrate de ejecutar el comando desde la carpeta 'u3-p7-proce-img-0150'.")

# 2. Calcular el espectro de frecuencia
espectro = obtener_espectro(imagen)

# 3. Mostrar dimensiones en consola
print("--- EJEMPLO 1 ---")
print("Imagen local cargada correctamente.")
print(f"Dimensiones: Alto={imagen.shape[0]}, Ancho={imagen.shape[1]}")

# 4. Crear la figura con 2 subgráficas (1 fila, 2 columnas)
figura, ejes = plt.subplots(1, 2, figsize=(12, 6))

# 5. Mostrar imagen original
ejes[0].imshow(imagen, cmap="gray")
ejes[0].set_title("Imagen Original (Dominio Espacial) 0150")
ejes[0].axis("off")

# 6. Mostrar espectro de frecuencia
ejes[1].imshow(espectro, cmap="gray")
ejes[1].set_title("Espectro de Frecuencia (Dominio de Frecuencia) 0150")
ejes[1].axis("off")

plt.tight_layout()

# 7. Guardar y mostrar
guardar_grafica(figura, "01_image_png_y_espectro.png")
plt.show()
print("Zavala Medina | NC = 0150")