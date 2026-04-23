# =============================================================================
#  PIA - VISIÓN COMPUTACIONAL
#  Módulo: Generación de Dataset Sintético (Fase 1)
# -----------------------------------------------------------------------------
#  CONTEXTO DE LA CELDA ROBÓTICA
#  -----------------------------
#  - Mesa de trabajo: 320 x 180 mm
#  - Origen de la mesa respecto al robot: W = (200, -50, 400) mm
#  - Orientación de la mesa respecto al robot: R = (0, -30, 0) grados (ϴ, φ, ψ)
#  - Cámara cenital (vista perpendicular a la mesa, ya rectificada por
#    calibración eye-to-hand). Por eso la imagen representa directamente
#    el plano de la mesa sin deformación.
#
#  OBJETIVO DEL SCRIPT
#  -------------------
#  Generar 20 imágenes sintéticas con perfiles mecánicos (rectángulos
#  elongados, L, T, cruz) rotados a distintos ángulos. Guardar un CSV
#  con los valores REALES (ground truth) que usaremos después para
#  medir la precisión del algoritmo de Canny + momentos (> 98 %).
#
#  El ground truth incluye:
#       - Nombre del archivo
#       - Tipo de figura
#       - Centroide real en píxeles (cx_px, cy_px)
#       - Centroide real en mm del sistema de la mesa (cx_mm, cy_mm)
#       - Ángulo de orientación real en grados
# =============================================================================

import os           # manejo de carpetas y rutas
import csv          # para escribir el archivo de ground truth
import random       # rotaciones, posiciones y tamaños aleatorios
import numpy as np  # arreglos numéricos (imágenes son arreglos 2D/3D)
import cv2          # OpenCV: dibujo y manipulación de imágenes


# -----------------------------------------------------------------------------
# PARÁMETROS DE LA MESA Y DE LA IMAGEN
# -----------------------------------------------------------------------------
# Dimensiones físicas de la mesa en milímetros.
MESA_ANCHO_MM = 320          # dimensión en X (largo de la mesa)
MESA_ALTO_MM  = 180          # dimensión en Y (ancho de la mesa)

# Escala de digitalización: 5 píxeles por cada milímetro real.
# A mayor escala, mayor resolución y mayor precisión de los momentos.
PX_POR_MM = 5                # 1 mm = 5 px

# Tamaño de la imagen derivado de lo anterior.
ANCHO = MESA_ANCHO_MM * PX_POR_MM   # 320 * 5 = 1600 px
ALTO  = MESA_ALTO_MM  * PX_POR_MM   # 180 * 5 =  900 px

COLOR_FONDO  = (255, 255, 255)      # fondo blanco (B, G, R)
COLOR_FIGURA = (0, 0, 0)            # figura negra -> máximo contraste

# -----------------------------------------------------------------------------
# PARÁMETROS DEL DATASET
# -----------------------------------------------------------------------------
NUM_IMAGENES   = 20
CARPETA_SALIDA = "dataset/imagenes"
CSV_SALIDA     = "dataset/ground_truth.csv"

# Semilla fija -> resultados reproducibles entre corridas.
random.seed(42)
np.random.seed(42)


# -----------------------------------------------------------------------------
# UTILIDAD: CONVERSIÓN PÍXELES <-> MILÍMETROS DE LA MESA
# -----------------------------------------------------------------------------
# Convención de ejes de la mesa:
#   - Origen en la esquina INFERIOR-IZQUIERDA de la imagen.
#   - Eje X de la mesa apunta hacia la derecha.
#   - Eje Y de la mesa apunta hacia ARRIBA (opuesto al eje Y de la imagen,
#     que en OpenCV apunta hacia abajo).
#
# De esta forma, el sistema de coordenadas de la mesa coincide con la
# convención habitual de RoboDK / ABB.
# -----------------------------------------------------------------------------
def px_a_mm(x_px, y_px):
    """Convierte coordenadas (x, y) en píxeles a (x, y) en mm de la mesa."""
    x_mm = x_px / PX_POR_MM
    y_mm = (ALTO - y_px) / PX_POR_MM   # se invierte Y
    return round(x_mm, 2), round(y_mm, 2)


# -----------------------------------------------------------------------------
# FUNCIONES PARA DIBUJAR CADA TIPO DE FIGURA
# -----------------------------------------------------------------------------
# Cada función recibe las dimensiones en MILÍMETROS y las convierte a
# píxeles internamente. Así los tamaños son realistas para la mesa.
# Todas devuelven los vértices CENTRADOS en (0, 0) para que rotarlos
# luego sea trivial.
# -----------------------------------------------------------------------------

def figura_rectangulo(largo_mm, ancho_mm):
    """Rectángulo elongado (p.ej. placa o barra plana)."""
    largo = largo_mm * PX_POR_MM
    ancho = ancho_mm * PX_POR_MM
    hl, ha = largo / 2, ancho / 2
    return np.array([
        [-hl, -ha],
        [ hl, -ha],
        [ hl,  ha],
        [-hl,  ha],
    ], dtype=np.float32)


def figura_L(largo_mm, ancho_mm, grosor_mm):
    """Perfil en L (típico ángulo estructural)."""
    largo  = largo_mm  * PX_POR_MM
    ancho  = ancho_mm  * PX_POR_MM
    grosor = grosor_mm * PX_POR_MM
    # Se construye el contorno de la L y se centra en (0,0).
    pts = np.array([
        [0,        0],
        [largo,    0],
        [largo,    grosor],
        [grosor,   grosor],
        [grosor,   ancho],
        [0,        ancho],
    ], dtype=np.float32)
    return pts - np.array([largo / 2, ancho / 2])


def figura_T(largo_mm, ancho_mm, grosor_mm):
    """Perfil en T."""
    largo  = largo_mm  * PX_POR_MM
    ancho  = ancho_mm  * PX_POR_MM
    grosor = grosor_mm * PX_POR_MM
    pts = np.array([
        [0,                   0],
        [largo,               0],
        [largo,               grosor],
        [largo/2 + grosor/2,  grosor],
        [largo/2 + grosor/2,  ancho],
        [largo/2 - grosor/2,  ancho],
        [largo/2 - grosor/2,  grosor],
        [0,                   grosor],
    ], dtype=np.float32)
    return pts - np.array([largo / 2, ancho / 2])


def figura_cruz(largo_mm, grosor_mm):
    """Perfil en cruz (+), totalmente simétrico."""
    largo  = largo_mm  * PX_POR_MM
    grosor = grosor_mm * PX_POR_MM
    hl, hg = largo / 2, grosor / 2
    return np.array([
        [-hl, -hg], [-hg, -hg], [-hg, -hl], [ hg, -hl],
        [ hg, -hg], [ hl, -hg], [ hl,  hg], [ hg,  hg],
        [ hg,  hl], [-hg,  hl], [-hg,  hg], [-hl,  hg],
    ], dtype=np.float32)


# -----------------------------------------------------------------------------
# ROTACIÓN DE UN POLÍGONO ALREDEDOR DEL ORIGEN
# -----------------------------------------------------------------------------
def rotar_puntos(puntos, angulo_grados):
    """
    Aplica la matriz de rotación 2D estándar:
        R(θ) = [ cos(θ)  -sin(θ) ]
               [ sin(θ)   cos(θ) ]
    Esta misma θ es la que después se intentará recuperar con los
    momentos centrales de segundo orden. Por eso se guarda en el CSV.
    """
    theta = np.deg2rad(angulo_grados)
    c, s  = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)
    return puntos @ R.T


# -----------------------------------------------------------------------------
# GENERACIÓN DE UNA IMAGEN INDIVIDUAL
# -----------------------------------------------------------------------------
def generar_imagen(idx):
    """
    Genera una imagen con UNA figura aleatoria y devuelve la imagen y
    el diccionario con el ground truth.
    """
    # 1) Lienzo completamente blanco con el tamaño de la mesa digitalizada.
    img = np.full((ALTO, ANCHO, 3), COLOR_FONDO, dtype=np.uint8)

    # 2) Tipo de figura al azar.
    tipo = random.choice(["rectangulo", "L", "T", "cruz"])

    # 3) Dimensiones realistas en MILÍMETROS para una mesa de 320x180 mm.
    if tipo == "rectangulo":
        largo_mm = random.randint(40, 80)    # lado largo
        ancho_mm = random.randint(8,  18)    # lado corto (elongado)
        puntos   = figura_rectangulo(largo_mm, ancho_mm)

    elif tipo == "L":
        largo_mm  = random.randint(35, 60)
        ancho_mm  = random.randint(35, 60)
        grosor_mm = random.randint(8,  14)
        puntos    = figura_L(largo_mm, ancho_mm, grosor_mm)

    elif tipo == "T":
        largo_mm  = random.randint(40, 65)
        ancho_mm  = random.randint(30, 50)
        grosor_mm = random.randint(8,  14)
        puntos    = figura_T(largo_mm, ancho_mm, grosor_mm)

    else:  # "cruz"
        largo_mm  = random.randint(40, 65)
        grosor_mm = random.randint(8,  14)
        puntos    = figura_cruz(largo_mm, grosor_mm)

    # 4) Ángulo de rotación aleatorio en [-90°, 90°].
    angulo = round(random.uniform(-90, 90), 2)
    puntos_rot = rotar_puntos(puntos, angulo)

    # 5) Posición aleatoria del centroide dentro de la mesa. Se reserva
    #    un margen igual al radio máximo de la figura para evitar que
    #    se corte contra los bordes de la imagen.
    radio_max = int(np.max(np.linalg.norm(puntos_rot, axis=1))) + 2
    cx_px = random.randint(radio_max, ANCHO - radio_max)
    cy_px = random.randint(radio_max, ALTO  - radio_max)

    # 6) Trasladar la figura al centro elegido y rellenarla.
    poligono = (puntos_rot + np.array([cx_px, cy_px])).astype(np.int32)
    cv2.fillPoly(img, [poligono], COLOR_FIGURA)

    # 7) Convertir el centroide a milímetros del sistema de la mesa.
    cx_mm, cy_mm = px_a_mm(cx_px, cy_px)

    # 8) Empaquetar todo el ground truth.
    gt = {
        "archivo":         f"figura_{idx:02d}.png",
        "tipo":            tipo,
        "cx_px":           cx_px,
        "cy_px":           cy_px,
        "cx_mm":           cx_mm,
        "cy_mm":           cy_mm,
        "angulo_real_deg": angulo,
    }
    return img, gt


# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    registros = []
    for i in range(1, NUM_IMAGENES + 1):
        img, gt = generar_imagen(i)
        ruta = os.path.join(CARPETA_SALIDA, gt["archivo"])
        cv2.imwrite(ruta, img)
        registros.append(gt)

        print(f"[{i:02d}/{NUM_IMAGENES}] {gt['archivo']}  "
              f"tipo={gt['tipo']:<10}  "
              f"centro_px=({gt['cx_px']:4d},{gt['cy_px']:4d})  "
              f"centro_mm=({gt['cx_mm']:6.2f},{gt['cy_mm']:6.2f})  "
              f"ángulo={gt['angulo_real_deg']:+6.2f}°")

    with open(CSV_SALIDA, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=registros[0].keys())
        writer.writeheader()
        writer.writerows(registros)

    print("\nDataset generado correctamente.")
    print(f"  Resolución:   {ANCHO} x {ALTO} px  ({MESA_ANCHO_MM} x {MESA_ALTO_MM} mm)")
    print(f"  Escala:       {PX_POR_MM} px/mm")
    print(f"  Imágenes:     {CARPETA_SALIDA}")
    print(f"  Ground truth: {CSV_SALIDA}")


if __name__ == "__main__":
    main()
