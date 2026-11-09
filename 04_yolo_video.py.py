import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

RUTA_VIDEO = "video_aeropuerto.mp4"
MODELO = "yolov8n.pt"
CONFIANZA = 0.25

DISPOSITIVO = 0 if torch.cuda.is_available() else "cpu"

print("Cargando modelo YOLOv8...")
model = YOLO(MODELO)

print("Dispositivo:", DISPOSITIVO)

# ==========================================
# 2. ABRIR VIDEO
# ==========================================

cap = cv2.VideoCapture(RUTA_VIDEO)

if not cap.isOpened():
    raise Exception("No se pudo abrir el video.")

ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_original = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Video de salida
salida = cv2.VideoWriter(
    "resultado_yolo.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_original,
    (ancho, alto)
)

# ==========================================
# 3. VARIABLES DE EVALUACIÓN
# ==========================================

frame_actual = 0
total_detecciones = 0
tiempos = []

print("\nIniciando detección...\n")

# ==========================================
# 4. PROCESAMIENTO DEL VIDEO
# ==========================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_actual += 1

    # Medir tiempo de inferencia
    if DISPOSITIVO != "cpu":
        torch.cuda.synchronize()

    inicio = time.perf_counter()

    resultados = model.predict(
        source=frame,
        conf=CONFIANZA,
        device=DISPOSITIVO,
        verbose=False
    )

    if DISPOSITIVO != "cpu":
        torch.cuda.synchronize()

    fin = time.perf_counter()

    tiempo_ms = (fin - inicio) * 1000
    tiempos.append(tiempo_ms)

    resultado = resultados[0]
    cajas = resultado.boxes

    cantidad = len(cajas)
    total_detecciones += cantidad

    # ======================================
    # 5. DIBUJAR DETECCIONES
    # ======================================

    for caja in cajas:

        # Coordenadas
        x1, y1, x2, y2 = map(
            int, caja.xyxy[0].tolist()
        )

        # Clase y confianza
        clase_id = int(caja.cls[0].item())
        confianza = float(caja.conf[0].item())

        nombre_clase = model.names[clase_id]
        porcentaje = confianza * 100

        # Dibujar rectángulo
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Etiqueta
        etiqueta = f"{nombre_clase}: {porcentaje:.1f}%"

        # Fondo de la etiqueta
        (tw, th), _ = cv2.getTextSize(
            etiqueta,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        cv2.rectangle(
            frame,
            (x1, max(0, y1 - 25)),
            (x1 + tw + 5, y1),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            etiqueta,
            (x1 + 2, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    # ======================================
    # 6. PANEL DE INFORMACIÓN
    # ======================================

    progreso = (frame_actual / total_frames) * 100

    fps_inferencia = 1000 / tiempo_ms if tiempo_ms > 0 else 0

    # Fondo del panel
    cv2.rectangle(
        frame,
        (10, 10),
        (390, 160),
        (30, 30, 30),
        -1
    )

    informacion = [
        "DETECCION DE OBJETOS - YOLOv8",
        f"Fotograma: {frame_actual}/{total_frames}",
        f"Progreso: {progreso:.1f}%",
        f"Objetos detectados: {cantidad}",
        f"Inferencia: {tiempo_ms:.2f} ms",
        f"FPS inferencia: {fps_inferencia:.2f}"
    ]

    for i, texto in enumerate(informacion):

        cv2.putText(
            frame,
            texto,
            (20, 35 + i * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

    # ======================================
    # 7. MOSTRAR Y GUARDAR
    # ======================================

    salida.write(frame)

    cv2.imshow(
        "Sistema de deteccion YOLOv8",
        frame
    )

    # Mostrar progreso en terminal
    if frame_actual % 25 == 0:

        print(
            f"Progreso: {progreso:.1f}% | "
            f"Objetos: {cantidad} | "
            f"Tiempo: {tiempo_ms:.2f} ms | "
            f"FPS: {fps_inferencia:.2f}"
        )

    # Presionar Q para finalizar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Procesamiento detenido por el usuario.")
        break


# ==========================================
# 8. FINALIZAR Y MOSTRAR RESULTADOS
# ==========================================

cap.release()
salida.release()
cv2.destroyAllWindows()

if tiempos:

    tiempo_promedio = np.mean(tiempos)
    fps_promedio = 1000 / tiempo_promedio

    print("\n========== RESULTADOS ==========")
    print("Modelo: YOLOv8n")
    print("Fotogramas procesados:", frame_actual)
    print("Total de detecciones:", total_detecciones)
    print(f"Tiempo promedio: {tiempo_promedio:.2f} ms")
    print(f"FPS promedio: {fps_promedio:.2f}")
    print("Video guardado: resultado_yolo.mp4")
    print("================================")