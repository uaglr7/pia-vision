# FASE 1, 2 Y 3
## 1. Diseño de perfiles en CAD (FreeCAD)
Se modelaron 4 perfiles mecánicos como cuerpos 3D con su centroide colocado en el origen (0,0,0), verificado con Shape.CenterOfMass:

- cad/rectangulo.FCStd → placa 60 × 12 × 2 mm.
- cad/L.FCStd → perfil en L 50 × 50 mm, grosor 12, espesor 2.
- cad/T.FCStd → perfil en T (cabecera 60 × 12, tallo 12 × 40), espesor 2.
- cad/cruz.FCStd → cruz 60 × 60, grosor 12, espesor 2.

## 2. Generador automático de escenas CAD
cad/generar_escenas.py — macro de FreeCAD que:

1) Abre cada .FCStd y aplica a la pieza una pose aleatoria (rotación + traslación) sobre la mesa de 320 × 180 mm.
2) Exporta la vista cenital como PNG 1600 × 900 px (escala fija 5 px/mm).
3) Lee la verdad desde CAD (Shape.CenterOfMass) y la guarda en CSV.
4) Reproducible (random.seed(123)).
Salida: dataset_cad/imagenes/figura_01..20.png + dataset_cad/ground_truth_cad.csv.

## 3. Dataset sintético complementario
generar_dataset.py — genera 20 imágenes 100 % en OpenCV (sin CAD). Está en dataset/. Útil como caso ideal para el reporte. USO LIMITADO A PRUEBAS.

## 4. Pipeline de visión con OpenCV
procesar_dataset.py — para cada imagen:

- Escala de grises + Gaussian blur.
- Canny para bordes.
- Contornos externos → el de mayor área.
- Centroide por momentos espaciales (M10/M00, M01/M00).
- Ángulo por momentos centrales de segundo orden (0.5·atan2(2·μ11, μ20−μ02)).
- Conversión píxeles → mm de la mesa (Y invertido).
- Comparación contra CSV de CAD (considera el offset base de cada perfil: rect 0°, L 45°, T 90°, cruz excluida).
- Genera imágenes anotadas con centroide y eje principal.
Salida: resultados/resultados.csv + resultados/anotadas/figura_*.png.

## Precisión actual:

- Precisión promedio centroide: 99.959 %
- Precisión mínima centroide: 99.947 %
- Error máximo centroide: 0.195 mm
- Error máximo ángulo: 0.010°

## Cómo ejecutar todo el pipeline desde cero

### 1) Dataset sintético (opcional, complementario)
python generar_dataset.py

### 2) Dataset CAD: 

- Descargar FreeCAD: https://www.freecad.org/

- DENTRO de FreeCAD abrir Python console:

exec(open(r"tu-dirección\PIA VISION\cad\generar_escenas.py").read())

### 3) Procesar y comparar en IDE:

python procesar_dataset.py
