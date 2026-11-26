# =====================================================
# ARCHIVO: funciones_frecuencia_0150.py
# PROPÓSITO: Funciones compartidas para procesamiento en frecuencia
# =====================================================
# Zavala Medina | NC= 0150
import numpy as np
import cv2
import os
from skimage import data
import matplotlib.pyplot as plt

def cargar_imagen():
    """Obtiene la imagen del camarógrafo en escala de grises."""
    return data.camera()

def obtener_espectro(imagen):
    """Calcula la Transformada de Fourier y devuelve el espectro visualizable."""
    # 1. Transformada de Fourier 2D
    f_transform = np.fft.fft2(imagen)
    # 2. Desplazamiento del origen al centro
    f_shift = np.fft.fftshift(f_transform)
    # 3. Magnitud y escala logarítmica para visualización (evitamos log(0) sumando 1)
    espectro = 20 * np.log(np.abs(f_shift) + 1)
    return espectro

def aplicar_filtro_frecuencia(imagen, kernel):
    """
    Aplica un kernel en el dominio de la frecuencia.
    Se añade un borde de relleno (padding) para evitar artefactos de convolución circular.
    """
    pad_size = 1
    # Rellenar la imagen con un borde reflejado
    imagen_padded = np.pad(imagen, pad_size, mode='reflect')
    rows, cols = imagen_padded.shape
    
    # Crear un kernel del mismo tamaño que la imagen, centrado en ceros
    kernel_padded = np.zeros((rows, cols), dtype=np.float32)
    k_rows, k_cols = kernel.shape
    start_r = (rows - k_rows) // 2
    start_c = (cols - k_cols) // 2
    kernel_padded[start_r:start_r+k_rows, start_c:start_c+k_cols] = kernel
    
    # Mover el centro del kernel al origen (esquina superior izquierda) para la FFT
    kernel_padded_shifted = np.fft.ifftshift(kernel_padded)
    
    # Transformada de Fourier de la imagen y del kernel
    fft_img = np.fft.fft2(imagen_padded)
    fft_ker = np.fft.fft2(kernel_padded_shifted)
    
    # Multiplicación en el dominio de la frecuencia (Teorema de la Convolución)
    resultado_fft = fft_img * fft_ker
    
    # Regresar al dominio espacial con la Transformada Inversa
    resultado_complejo = np.fft.ifft2(resultado_fft)
    resultado_real = np.real(resultado_complejo)
    
    # Quitar el píxel de relleno añadido al principio
    return resultado_real[pad_size:-pad_size, pad_size:-pad_size]

def normalizar_visualizacion(imagen):
    """Convierte una respuesta de filtro (con valores negativos/positivos) a 0-255."""
    imagen_abs = np.abs(imagen)
    imagen_normalizada = cv2.normalize(
        imagen_abs, None, 0, 255, cv2.NORM_MINMAX
    )
    return imagen_normalizada.astype(np.uint8)

def guardar_grafica(figura, nombre_archivo):
    """Guarda una figura de matplotlib dentro de la carpeta resultados-img-0150."""
    carpeta = "resultados-img-0150"
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, nombre_archivo)
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"Imagen guardada correctamente en: {ruta}")
    print("Zavala Medina NC= 0150")