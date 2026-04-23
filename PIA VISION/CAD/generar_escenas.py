# =============================================================================
#  PIA - VISIÓN COMPUTACIONAL | Fase 1 (versión CAD)
#  Macro de FreeCAD que genera 20 escenas sintéticas a partir de los
#  4 perfiles diseñados en CAD (rectangulo, L, T, cruz).
#
#  Cada escena:
#     - Coloca UNA pieza (aleatoria) sobre la mesa de 320 x 180 mm
#       con una traslación (tx, ty) y una rotación (ángulo) conocidas.
#     - Exporta la vista cenital como PNG de 1600 x 900 px
#       (escala fija = 5 px/mm, el mismo criterio que nuestro dataset sintético).
#     - Lee el centroide real desde el CAD (Shape.CenterOfMass)
#       como "ground truth" para comparar luego con OpenCV.
#
#  Salidas:
#     dataset_cad/imagenes/figura_01.png ... figura_20.png
#     dataset_cad/ground_truth_cad.csv
#
#  CÓMO EJECUTAR DESDE FREECAD
#  ---------------------------
#     1) Abre FreeCAD.
#     2) Menú  View -> Panels -> Python console
#     3) En la consola, escribe:
#           exec(open(r"D:\REPOSITORIOS\PIA VISION\cad\generar_escenas.py").read())
#     4) Espera: verás 20 mensajes "[i/20] figura_XX.png ..."
# =============================================================================

import FreeCAD as App
import FreeCADGui as Gui
import os
import csv
import random
from FreeCAD import Placement, Vector, Rotation
from pivy import coin


# -----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------------------------------------------------------
BASE_DIR = r"D:\REPOSITORIOS\PIA VISION\cad"                 # donde están los .FCStd
OUT_DIR  = r"D:\REPOSITORIOS\PIA VISION\dataset_cad"         # salida
IMG_DIR  = os.path.join(OUT_DIR, "imagenes")
CSV_PATH = os.path.join(OUT_DIR, "ground_truth_cad.csv")

# Mesa de trabajo (del enunciado).
MESA_X_MM = 320.0
MESA_Y_MM = 180.0

# Resolución de la imagen exportada (mismo 5 px/mm que el dataset sintético).
IMG_W = 1600  # 320 mm * 5 px/mm
IMG_H =  900  # 180 mm * 5 px/mm

# Número de escenas a generar.
NUM_ESCENAS = 20

# Margen en mm para que la pieza no se salga de la mesa al rotar.
MARGEN_MM = 50

# Semilla -> reproducibilidad.
random.seed(123)

# Mapeo: nombre de pieza -> archivo .FCStd. El objeto principal se
# detecta automáticamente (ver 'detectar_objeto_principal').
PIEZAS = {
    "rectangulo": "rectangulo.FCStd",
    "L":          "L.FCStd",
    "T":          "T.FCStd",
    "cruz":       "cruz.FCStd",
}


# -----------------------------------------------------------------------------
# UTILIDADES
# -----------------------------------------------------------------------------
def detectar_objeto_principal(doc):
    """
    Busca en el documento el objeto que representa la pieza final.
    Prioriza:
        1) Una fusión booleana (Part::MultiFuse / Part::Fuse).
        2) Un cubo (Part::Box) si no hay fusión (caso rectángulo).
        3) Cualquier objeto con Shape no vacío.
    Devuelve None si no encuentra ninguno.
    """
    # Primero: fusiones
    for o in doc.Objects:
        if o.TypeId in ("Part::MultiFuse", "Part::Fuse"):
            return o
    # Luego: cubos (para el rectángulo)
    for o in doc.Objects:
        if o.TypeId == "Part::Box":
            return o
    # Por último: cualquier objeto con Shape
    for o in doc.Objects:
        if hasattr(o, "Shape") and o.Shape is not None and not o.Shape.isNull():
            return o
    return None


def obtener_solido(obj):
    """
    Devuelve el sólido de la pieza aunque el Shape venga envuelto como
    Compound (caso de la fusión booleana de FreeCAD con 2 cubos).
    """
    shp = obj.Shape
    if shp.ShapeType == "Compound" and len(shp.Solids) > 0:
        return shp.Solids[0]
    return shp


def configurar_camara_ortografica(view):
    """
    Coloca la cámara EXACTAMENTE encima del centro de la mesa, mirando
    hacia abajo, con una ventana ortográfica que encuadra justo la
    mesa (320 x 180 mm). Así la relación px/mm es constante entre
    imágenes: 1600 px / 320 mm = 5 px/mm.
    """
    view.setCameraType("Orthographic")
    cam = view.getCameraNode()

    # Posición: centro de la mesa, alto en Z (para no cortar la pieza
    # de 2 mm de espesor).
    cam.position.setValue(MESA_X_MM / 2, MESA_Y_MM / 2, 100.0)

    # Orientación identidad = mirar hacia -Z con "arriba" en +Y.
    cam.orientation.setValue(coin.SbRotation())

    # "height" en cámara ortográfica = extensión VERTICAL de la ventana
    # en las mismas unidades de la escena (mm). El ancho se ajusta
    # automáticamente según el aspect ratio de la imagen exportada.
    cam.height.setValue(MESA_Y_MM)

    # Planos de corte cerca/lejos: cualquier rango amplio va bien.
    cam.nearDistance.setValue(1.0)
    cam.farDistance.setValue(500.0)


def aplicar_pose(obj, tx, ty, angulo_deg):
    """
    Compone la pose deseada (rotación alrededor de Z + traslación)
    ENCIMA del Placement que ya trae la pieza (el Placement que
    compensa para que el centroide base esté en (0,0,0)).

    De este modo, tras aplicar la nueva pose:
        - El centroide queda en (tx, ty, ~1).
        - La pieza gira alrededor de su propio centroide.
    """
    pose_nueva = Placement(
        Vector(tx, ty, 0),
        Rotation(Vector(0, 0, 1), angulo_deg),  # eje Z, ángulo en grados
    )
    # placement_final = pose_nueva ∘ placement_actual
    # (aplica primero la compensación de la pieza, luego la pose nueva)
    obj.Placement = pose_nueva.multiply(obj.Placement)


# -----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    registros = []

    # IMPORTANTE: cerrar cualquier documento que pudiera haber quedado
    # abierto de una corrida anterior. Si no, al volver a 'abrir' el
    # archivo FreeCAD reutiliza el doc ya cargado (con la pose previa
    # aplicada) y la pose se aplicaría DOBLE.
    for nombre_doc in list(App.listDocuments().keys()):
        App.closeDocument(nombre_doc)

    tipos = list(PIEZAS.keys())

    for i in range(1, NUM_ESCENAS + 1):
        # 1) Tipo de pieza aleatorio.
        tipo = random.choice(tipos)
        archivo = PIEZAS[tipo]

        # 2) Abrir el archivo correspondiente.
        ruta_fcstd = os.path.join(BASE_DIR, archivo)
        doc = App.openDocument(ruta_fcstd)
        Gui.ActiveDocument = Gui.getDocument(doc.Name)

        # 3) Detectar automáticamente el objeto principal (sin depender
        #    del nombre "Cube" o "Fusion" porque puede variar).
        obj = detectar_objeto_principal(doc)
        if obj is None:
            nombres = [o.Name for o in doc.Objects]
            raise RuntimeError(
                f"No se encontró objeto principal en {archivo}. "
                f"Objetos presentes: {nombres}"
            )

        # 4) Elegir pose aleatoria.
        angulo_deg = round(random.uniform(-90, 90), 2)
        tx = round(random.uniform(MARGEN_MM, MESA_X_MM - MARGEN_MM), 2)
        ty = round(random.uniform(MARGEN_MM, MESA_Y_MM - MARGEN_MM), 2)

        # 5) Aplicar pose.
        aplicar_pose(obj, tx, ty, angulo_deg)
        doc.recompute()

        # 6) Configurar vista cenital ortográfica.
        #    El fondo blanco lo forza saveImage con el parámetro "White".
        view = Gui.ActiveDocument.ActiveView
        view.viewTop()
        configurar_camara_ortografica(view)

        # 7) Guardar PNG.
        nombre = f"figura_{i:02d}.png"
        ruta_png = os.path.join(IMG_DIR, nombre)
        view.saveImage(ruta_png, IMG_W, IMG_H, "White")

        # 8) Leer centroide REAL desde el CAD (verdad de referencia).
        solido = obtener_solido(obj)
        cm = solido.CenterOfMass  # Vector (x, y, z) en mm

        registros.append({
            "archivo":        nombre,
            "tipo":           tipo,
            "cx_mm_cad":      round(cm.x, 3),
            "cy_mm_cad":      round(cm.y, 3),
            "angulo_cad_deg": angulo_deg,
            "tx_mm":          tx,
            "ty_mm":          ty,
        })

        print(f"[{i:02d}/{NUM_ESCENAS}] {nombre}  tipo={tipo:<10}  "
              f"centro_cad=({cm.x:7.2f},{cm.y:7.2f})  "
              f"ángulo={angulo_deg:+6.2f}°")

        # 9) Cerrar el documento SIN guardar (para no alterar el base).
        App.closeDocument(doc.Name)

    # 10) Escribir CSV con el ground truth de CAD.
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=registros[0].keys())
        w.writeheader()
        w.writerows(registros)

    print("\n=================================================")
    print("  Dataset CAD generado correctamente.")
    print(f"  Imágenes:     {IMG_DIR}")
    print(f"  Ground truth: {CSV_PATH}")
    print(f"  Resolución:   {IMG_W} x {IMG_H} px  ({MESA_X_MM} x {MESA_Y_MM} mm)")
    print(f"  Escala:       {IMG_W / MESA_X_MM} px/mm")
    print("=================================================")


# Ejecutar al cargar el script.
main()
