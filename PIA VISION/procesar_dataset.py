# =============================================================================
#  PIA - VISIÓN COMPUTACIONAL | Fases 2, 3 y 4
#  Script que CARGA las 20 imágenes generadas desde CAD, les aplica el
#  pipeline clásico (gris -> Canny -> contornos -> momentos) y compara
#  el resultado contra el 'ground_truth_cad.csv'.
#
#  Salidas:
#     resultados/resultados.csv    -> tabla comparativa CAD vs OpenCV
#     resultados/anotadas/*.png    -> imágenes con centroide y eje dibujados
#
#  Uso:
#     python procesar_dataset.py
# =============================================================================

import os
import csv
import math
import cv2
import numpy as np


# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
IMG_DIR    = "dataset_cad/imagenes"
CSV_CAD    = "dataset_cad/ground_truth_cad.csv"
OUT_DIR    = "resultados"
OUT_ANOT   = os.path.join(OUT_DIR, "anotadas")
OUT_CSV    = os.path.join(OUT_DIR, "resultados.csv")

# Mesa y escala (DEBEN coincidir con las del script de FreeCAD).
MESA_X_MM = 320.0
MESA_Y_MM = 180.0
IMG_W_PX  = 1600
IMG_H_PX  =  900
PX_POR_MM = IMG_W_PX / MESA_X_MM  # = 5 px/mm

# Ángulo de orientación "base" de cada perfil (teórico, sin rotar).
# OpenCV detectará angulo_aplicado + este base.
ANG_BASE = {
    "rectangulo":  0.0,
    "L":          45.0,
    "T":          90.0,
    "cruz":        0.0,  # degenerado, se excluye de comparación
}


# -----------------------------------------------------------------------------
# UTILIDADES GEOMÉTRICAS
# -----------------------------------------------------------------------------
def px_a_mm(x_px, y_px):
    """Convierte (x_px, y_px) de imagen al sistema de la mesa en mm.
    Origen de la mesa = esquina inferior-izquierda de la imagen;
    eje Y invertido respecto a la imagen."""
    x_mm = x_px / PX_POR_MM
    y_mm = (IMG_H_PX - y_px) / PX_POR_MM
    return x_mm, y_mm


def diff_angulos(a_deg, b_deg):
    """
    Diferencia entre dos ángulos tomando en cuenta que el eje principal
    tiene periodicidad de 180° (no 360°). El resultado queda en (-90, 90].
    """
    d = (a_deg - b_deg + 90) % 180 - 90
    return d


# -----------------------------------------------------------------------------
# PIPELINE DE VISIÓN
# -----------------------------------------------------------------------------
def procesar_imagen(ruta_img):
    """
    Aplica el pipeline completo a UNA imagen y devuelve:
        cx_px, cy_px, angulo_deg, contorno (para visualizar)
    """
    # 1) Cargar en escala de grises (fase 2 del PIA).
    img_gris = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
    if img_gris is None:
        raise FileNotFoundError(ruta_img)

    # 2) Detección de bordes con Canny. Umbrales 50/150 estándar para
    #    bordes de alto contraste como los nuestros (figura en gris
    #    sobre fondo blanco). Las imágenes CAD tienen sombreado suave
    #    así que un blur previo ayuda a que Canny no detecte ruido.
    img_blur = cv2.GaussianBlur(img_gris, (5, 5), 0)
    edges = cv2.Canny(img_blur, 50, 150)

    # 3) Contornos externos (ignora los huecos internos).
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError(f"No se detectaron contornos en {ruta_img}")

    # Nos quedamos con el contorno de MAYOR área (la pieza principal).
    cnt = max(contours, key=cv2.contourArea)

    # 4) Momentos espaciales del contorno.
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        raise RuntimeError(f"Contorno degenerado en {ruta_img}")

    # 5) Centroide (momentos M10/M00, M01/M00).
    cx_px = M["m10"] / M["m00"]
    cy_px = M["m01"] / M["m00"]

    # 6) Ángulo con momentos CENTRALES de segundo orden:
    #    θ = 0.5 * atan2(2·μ11, μ20 - μ02)
    mu20 = M["mu20"]
    mu02 = M["mu02"]
    mu11 = M["mu11"]
    angulo_rad = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    angulo_deg = math.degrees(angulo_rad)

    return cx_px, cy_px, angulo_deg, cnt


# -----------------------------------------------------------------------------
# DIBUJO DEL REPORTE VISUAL (fase 4)
# -----------------------------------------------------------------------------
def dibujar_resultado(ruta_img, cx_px, cy_px, angulo_deg, cnt, out_path):
    """Superpone sobre la imagen original: contorno, centroide y eje principal."""
    img = cv2.imread(ruta_img, cv2.IMREAD_COLOR)

    # Contorno en verde.
    cv2.drawContours(img, [cnt], -1, (0, 200, 0), 2)

    # Eje principal: línea del largo del objeto en la dirección del ángulo.
    largo = 120  # px, arbitrario para visualización
    dx = largo * math.cos(math.radians(angulo_deg))
    dy = largo * math.sin(math.radians(angulo_deg))
    p1 = (int(cx_px - dx), int(cy_px - dy))
    p2 = (int(cx_px + dx), int(cy_px + dy))
    cv2.line(img, p1, p2, (0, 0, 255), 3)   # rojo

    # Centroide en azul.
    cv2.circle(img, (int(cx_px), int(cy_px)), 8, (255, 0, 0), -1)

    # Texto con coords mm y ángulo.
    x_mm, y_mm = px_a_mm(cx_px, cy_px)
    txt = f"({x_mm:.1f},{y_mm:.1f}) mm  {angulo_deg:+.1f} deg"
    cv2.putText(img, txt, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    cv2.imwrite(out_path, img)


# -----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_ANOT, exist_ok=True)

    # Cargar ground truth del CAD.
    gt = {}
    with open(CSV_CAD, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["archivo"]] = row

    resultados = []
    for nombre in sorted(gt.keys()):
        ruta = os.path.join(IMG_DIR, nombre)
        cx_px, cy_px, ang_px, cnt = procesar_imagen(ruta)

        # Convertir centroide a mm de la mesa.
        cx_mm, cy_mm = px_a_mm(cx_px, cy_px)

        # Datos del CAD.
        g = gt[nombre]
        tipo          = g["tipo"]
        cx_mm_cad     = float(g["cx_mm_cad"])
        cy_mm_cad     = float(g["cy_mm_cad"])
        ang_cad_deg   = float(g["angulo_cad_deg"])

        # Ángulo esperado que OpenCV DEBERÍA detectar.
        # OJO al signo: el eje Y en la imagen crece hacia abajo, mientras
        # en la mesa crece hacia arriba. Por eso una rotación CCW en CAD
        # se observa como CW en la imagen -> el signo de α se invierte.
        ang_esperado = -ang_cad_deg + ANG_BASE[tipo]

        # Errores.
        err_cx_mm    = cx_mm - cx_mm_cad
        err_cy_mm    = cy_mm - cy_mm_cad
        err_centro   = math.sqrt(err_cx_mm**2 + err_cy_mm**2)

        if tipo == "cruz":
            err_ang = None  # la cruz es degenerada
        else:
            err_ang = diff_angulos(ang_px, ang_esperado)

        # Precisión del centroide respecto al tamaño de la mesa.
        # Usamos la diagonal como referencia de "tamaño del espacio".
        diag_mesa = math.sqrt(MESA_X_MM**2 + MESA_Y_MM**2)
        precision_centro = 100.0 * (1 - err_centro / diag_mesa)

        resultados.append({
            "archivo":          nombre,
            "tipo":             tipo,
            "cx_mm_cad":        round(cx_mm_cad, 3),
            "cy_mm_cad":        round(cy_mm_cad, 3),
            "cx_mm_cv":         round(cx_mm, 3),
            "cy_mm_cv":         round(cy_mm, 3),
            "err_cx_mm":        round(err_cx_mm, 3),
            "err_cy_mm":        round(err_cy_mm, 3),
            "err_centro_mm":    round(err_centro, 3),
            "precision_%":      round(precision_centro, 3),
            "angulo_cad_deg":   round(ang_cad_deg, 2),
            "angulo_esperado":  round(ang_esperado, 2),
            "angulo_cv_deg":    round(ang_px, 2),
            "err_angulo_deg":   None if err_ang is None else round(err_ang, 2),
        })

        # Imagen con anotaciones.
        out_anot = os.path.join(OUT_ANOT, nombre)
        dibujar_resultado(ruta, cx_px, cy_px, ang_px, cnt, out_anot)

        print(f"{nombre}  tipo={tipo:<10}  "
              f"err_centro={err_centro:6.3f} mm  "
              f"precision={precision_centro:6.3f}%  "
              f"err_ang={'N/A' if err_ang is None else f'{err_ang:+6.2f}°'}")

    # Guardar CSV de resultados.
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=resultados[0].keys())
        w.writeheader()
        w.writerows(resultados)

    # Resumen final.
    errs = [r["err_centro_mm"] for r in resultados]
    precs = [r["precision_%"] for r in resultados]
    angs = [abs(r["err_angulo_deg"]) for r in resultados
            if r["err_angulo_deg"] is not None]

    print("\n" + "=" * 55)
    print("  RESUMEN DE PRECISIÓN")
    print("=" * 55)
    print(f"  Imágenes procesadas:       {len(resultados)}")
    print(f"  Error centroide promedio:  {np.mean(errs):.3f} mm")
    print(f"  Error centroide máximo:    {np.max(errs):.3f} mm")
    print(f"  Precisión promedio:        {np.mean(precs):.3f} %")
    print(f"  Precisión mínima:          {np.min(precs):.3f} %")
    if angs:
        print(f"  Error ángulo promedio:     {np.mean(angs):.3f}°  "
              f"(N={len(angs)}, excluye cruz)")
        print(f"  Error ángulo máximo:       {np.max(angs):.3f}°")
    print(f"  Salida:  {OUT_CSV}")
    print(f"           {OUT_ANOT}/*.png")
    print("=" * 55)


if __name__ == "__main__":
    main()
