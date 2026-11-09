"""
Actividad 06 (Segunda unidad) - Apartado C
Deteccion de objetos en VIDEO con Faster R-CNN (familia RCNN).

Interfaz moderna (customtkinter) que:
  * Lee un video guardado en la computadora.
  * Detecta objetos cuadro a cuadro con Faster R-CNN (torchvision).
  * Muestra en vivo el video con las cajas, clases y confianza.
  * Calcula metricas de desempeno: latencia promedio por frame,
    tiempo promedio por objeto, FPS de inferencia, percentil 95,
    objetos totales, confianza media y conteo por clase.
  * Grafica la latencia en tiempo real y permite exportar un CSV.
  * Puede GUARDAR el video procesado con las cajas delimitadoras (.mp4).
  * Monitorea el consumo de CPU (sistema y proceso) y de RAM con psutil.

Ejecutar:  python main.py
"""

import csv
import hashlib
import os
import queue
import threading
import time
from collections import Counter
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import matplotlib
import numpy as np
import psutil
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
from torchvision.models import detection as tv_det
from torchvision.transforms import functional as TF

matplotlib.use("TkAgg")

# --------------------------------------------------------------------------
# Tema / paleta (estilo web moderno, modo oscuro)
# --------------------------------------------------------------------------
BG = "#0b0f1a"
SURFACE = "#121829"
CARD = "#182036"
BORDER = "#26304d"
TEXT = "#e8ecf6"
MUTED = "#8b95b3"
ACCENT = "#6366f1"      # indigo
ACCENT_2 = "#22d3ee"    # cyan
GOOD = "#34d399"
WARN = "#fbbf24"
DANGER = "#f87171"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

MODELS = {
    "Faster R-CNN · MobileNetV3 (rapido)": "mobilenet",
    "Faster R-CNN · ResNet50-FPN (preciso)": "resnet50",
}

PALETTE = [
    (99, 102, 241), (34, 211, 238), (52, 211, 153), (251, 191, 36),
    (248, 113, 113), (244, 114, 182), (167, 139, 250), (96, 165, 250),
    (74, 222, 128), (251, 146, 60), (45, 212, 191), (232, 121, 249),
]


def color_for(label: str):
    """Color estable por clase (RGB)."""
    h = int(hashlib.md5(label.encode()).hexdigest(), 16)
    return PALETTE[h % len(PALETTE)]


# --------------------------------------------------------------------------
# Hilo de deteccion
# --------------------------------------------------------------------------
class DetectorWorker(threading.Thread):
    """Carga el modelo, lee el video y procesa los frames en segundo plano."""

    def __init__(self, video_path, model_key, conf_thr, frame_step, max_width, out_q,
                 out_path=None):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.out_path = out_path  # ruta del video procesado (None = no guardar)
        self.model_key = model_key
        self.conf_thr = conf_thr
        self.frame_step = max(1, frame_step)
        self.max_width = max_width
        self.out_q = out_q
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # set = corriendo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- utilidades ---
    def _load_model(self):
        if self.model_key == "mobilenet":
            weights = tv_det.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            model = tv_det.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        else:
            weights = tv_det.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = tv_det.fasterrcnn_resnet50_fpn_v2(weights=weights)
        model.eval().to(self.device)
        return model, weights.meta["categories"]

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def set_conf(self, value):
        self.conf_thr = value

    # --- bucle principal ---
    def run(self):
        cap = None
        writer = None
        saved_frames = 0
        try:
            self.out_q.put(("status", "Cargando modelo (la primera vez descarga los pesos)..."))
            model, categories = self._load_model()

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("No se pudo abrir el video seleccionado.")

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 0
            self.out_q.put(("info", {"total": total, "fps_src": fps_src,
                                     "device": self.device.type.upper()}))

            # Calentamiento: la 1ra inferencia es lenta (inicializacion) y se descarta.
            self.out_q.put(("status", "Calentando el modelo..."))
            dummy = torch.rand(3, 360, 640, device=self.device)
            with torch.no_grad():
                model([dummy])
            self._sync()

            self.out_q.put(("status", "Procesando video..."))
            frame_idx = 0
            processed = 0
            while not self.stop_event.is_set():
                self.pause_event.wait()
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if (frame_idx - 1) % self.frame_step != 0:
                    continue

                # Reducir resolucion para acelerar (y mostrar el mismo frame)
                h, w = frame.shape[:2]
                if w > self.max_width:
                    scale = self.max_width / w
                    frame = cv2.resize(frame, (self.max_width, int(h * scale)),
                                       interpolation=cv2.INTER_AREA)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = TF.to_tensor(rgb).to(self.device)

                # --- inferencia cronometrada (solo forward del modelo) ---
                self._sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model([tensor])[0]
                self._sync()
                dt_ms = (time.perf_counter() - t0) * 1000.0

                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()

                dets = []
                for b, s, l in zip(boxes, scores, labels):
                    if s >= self.conf_thr:
                        dets.append((b.astype(int), float(s), categories[int(l)]))

                annotated = self._draw(rgb, dets)
                processed += 1

                # --- guardar el frame con las cajas (fuera del tiempo medido) ---
                if self.out_path:
                    if writer is None:
                        hh, ww = annotated.shape[:2]
                        # Si se saltan frames, se baja el FPS para conservar la duracion original
                        out_fps = max((fps_src or 30.0) / self.frame_step, 1.0)
                        writer = cv2.VideoWriter(self.out_path,
                                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                                 out_fps, (ww, hh))
                        if not writer.isOpened():
                            writer = None
                            self.out_path = None
                            self.out_q.put(("status", "No se pudo crear el video de salida."))
                    if writer is not None:
                        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                        saved_frames += 1

                self.out_q.put(("frame", {
                    "image": annotated,
                    "time_ms": dt_ms,
                    "dets": [(d[2], d[1]) for d in dets],
                    "frame_idx": frame_idx,
                    "processed": processed,
                    "total": total,
                }))

                # Evita saturar la UI si el modelo es muy rapido (GPU)
                while self.out_q.qsize() > 3 and not self.stop_event.is_set():
                    time.sleep(0.005)

            self.out_q.put(("done", None))
        except Exception as exc:  # noqa
            self.out_q.put(("error", str(exc)))
        finally:
            # Siempre se liberan el video de entrada y el archivo de salida,
            # incluso si el usuario detiene el proceso a la mitad.
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
                if saved_frames:
                    self.out_q.put(("saved", {"path": self.out_path, "frames": saved_frames}))

    @staticmethod
    def _draw(rgb, dets):
        img = rgb.copy()
        for (x1, y1, x2, y2), score, name in dets:
            col = color_for(name)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
            text = f"{name} {score * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_top = max(y1 - th - 8, 0)
            cv2.rectangle(img, (x1, y_top), (x1 + tw + 8, y_top + th + 8), col, -1)
            cv2.putText(img, text, (x1 + 4, y_top + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img


# --------------------------------------------------------------------------
# Componentes de UI
# --------------------------------------------------------------------------
class MetricCard(ctk.CTkFrame):
    def __init__(self, master, title, unit="", accent=ACCENT):
        super().__init__(master, fg_color=CARD, corner_radius=16,
                         border_width=1, border_color=BORDER)
        self.unit = unit
        bar = ctk.CTkFrame(self, width=36, height=4, fg_color=accent, corner_radius=2)
        bar.pack(anchor="w", padx=16, pady=(14, 0))
        ctk.CTkLabel(self, text=title.upper(), text_color=MUTED,
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=16, pady=(8, 0))
        self.value = ctk.CTkLabel(self, text="—", text_color=TEXT,
                                  font=ctk.CTkFont(size=28, weight="bold"))
        self.value.pack(anchor="w", padx=16, pady=(0, 0))
        self.sub = ctk.CTkLabel(self, text=unit, text_color=MUTED, font=ctk.CTkFont(size=11))
        self.sub.pack(anchor="w", padx=16, pady=(0, 14))

    def set(self, value, sub=None):
        self.value.configure(text=value)
        if sub is not None:
            self.sub.configure(text=sub)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Detección de objetos en video · Faster R-CNN")
        self.geometry("1440x920")
        self.minsize(1200, 820)
        self.configure(fg_color=BG)

        self.video_path = None
        self.worker = None
        self.q = queue.Queue()
        self.times = []          # latencia por frame (ms)
        self.det_counts = []     # nº de objetos por frame
        self.confs = []          # todas las confianzas
        self.class_counter = Counter()
        self.frame_log = []      # para CSV
        self.total_frames = 0
        self.t_start = None
        self._photo = None

        # --- monitoreo de recursos (psutil) ---
        self.proc = psutil.Process(os.getpid())
        self.ncpu = psutil.cpu_count(logical=True) or 1
        self.cpu_sys = []        # % de CPU de todo el sistema
        self.cpu_proc = []       # % de CPU de este programa (sobre el total de la CPU)
        self.ram_mb = []         # RAM usada por este programa (MB)
        self._mon_job = None
        self.saved_info = None

        self._build_sidebar()
        self._build_main()
        self.after(15, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=320, fg_color=SURFACE, corner_radius=0)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        ctk.CTkLabel(sb, text="◆ VisionLab", text_color=ACCENT_2,
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=24, pady=(26, 0))
        ctk.CTkLabel(sb, text="Faster R-CNN\nen video", text_color=TEXT, justify="left",
                     font=ctk.CTkFont(size=26, weight="bold")).pack(anchor="w", padx=24, pady=(4, 0))
        ctk.CTkLabel(sb, text="Actividad 06 · Apartado C", text_color=MUTED,
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=24, pady=(2, 18))

        self._section(sb, "VIDEO")
        self.btn_open = ctk.CTkButton(sb, text="📂  Seleccionar video", height=40, corner_radius=12,
                                      fg_color=CARD, hover_color=BORDER, border_width=1,
                                      border_color=BORDER, command=self.choose_video)
        self.btn_open.pack(fill="x", padx=24, pady=(6, 4))
        self.lbl_file = ctk.CTkLabel(sb, text="Ningún archivo seleccionado", text_color=MUTED,
                                     font=ctk.CTkFont(size=11), wraplength=270, justify="left")
        self.lbl_file.pack(anchor="w", padx=24, pady=(0, 10))

        self._section(sb, "MODELO")
        self.model_var = ctk.StringVar(value=list(MODELS.keys())[0])
        ctk.CTkOptionMenu(sb, values=list(MODELS.keys()), variable=self.model_var, height=36,
                          corner_radius=10, fg_color=CARD, button_color=BORDER,
                          button_hover_color=ACCENT, dropdown_fg_color=CARD).pack(fill="x", padx=24, pady=(6, 10))

        self._section(sb, "PARÁMETROS")
        self.conf_lbl = self._slider_label(sb, "Umbral de confianza", "0.50")
        self.conf_slider = ctk.CTkSlider(sb, from_=0.1, to=0.95, number_of_steps=85,
                                         progress_color=ACCENT, button_color=ACCENT_2,
                                         command=self._on_conf)
        self.conf_slider.set(0.5)
        self.conf_slider.pack(fill="x", padx=24, pady=(0, 8))

        self.step_lbl = self._slider_label(sb, "Procesar 1 de cada N frames", "1")
        self.step_slider = ctk.CTkSlider(sb, from_=1, to=10, number_of_steps=9,
                                         progress_color=ACCENT, button_color=ACCENT_2,
                                         command=lambda v: self.step_lbl.configure(text=str(int(v))))
        self.step_slider.set(1)
        self.step_slider.pack(fill="x", padx=24, pady=(0, 8))

        self.width_lbl = self._slider_label(sb, "Ancho máx. de inferencia (px)", "640")
        self.width_slider = ctk.CTkSlider(sb, from_=320, to=1280, number_of_steps=16,
                                          progress_color=ACCENT, button_color=ACCENT_2,
                                          command=lambda v: self.width_lbl.configure(text=str(int(v))))
        self.width_slider.set(640)
        self.width_slider.pack(fill="x", padx=24, pady=(0, 10))

        self.save_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sb, text="Guardar video procesado (.mp4)", variable=self.save_var,
                      progress_color=ACCENT, text_color=TEXT,
                      font=ctk.CTkFont(size=12)).pack(anchor="w", padx=24, pady=(0, 12))

        self.btn_start = ctk.CTkButton(sb, text="▶  Iniciar detección", height=44, corner_radius=12,
                                       fg_color=ACCENT, hover_color="#4f46e5",
                                       font=ctk.CTkFont(size=14, weight="bold"), command=self.start)
        self.btn_start.pack(fill="x", padx=24, pady=(4, 6))

        row = ctk.CTkFrame(sb, fg_color="transparent")
        row.pack(fill="x", padx=24)
        self.btn_pause = ctk.CTkButton(row, text="⏸ Pausa", height=36, corner_radius=10, state="disabled",
                                       fg_color=CARD, hover_color=BORDER, command=self.toggle_pause)
        self.btn_pause.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.btn_stop = ctk.CTkButton(row, text="⏹ Detener", height=36, corner_radius=10, state="disabled",
                                      fg_color=CARD, hover_color=DANGER, command=self.stop)
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=(4, 0))

        self.btn_export = ctk.CTkButton(sb, text="⬇  Exportar métricas (CSV)", height=36, corner_radius=10,
                                        fg_color="transparent", border_width=1, border_color=BORDER,
                                        hover_color=CARD, state="disabled", command=self.export_csv)
        self.btn_export.pack(fill="x", padx=24, pady=(10, 0))

        dev = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        ctk.CTkLabel(sb, text=f"Dispositivo: {dev}", text_color=GOOD if torch.cuda.is_available() else WARN,
                     font=ctk.CTkFont(size=11, weight="bold")).pack(side="bottom", anchor="w", padx=24, pady=18)

    def _section(self, parent, text):
        ctk.CTkLabel(parent, text=text, text_color=MUTED,
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=24, pady=(6, 0))

    def _slider_label(self, parent, text, value):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=(6, 2))
        ctk.CTkLabel(row, text=text, text_color=TEXT, font=ctk.CTkFont(size=12)).pack(side="left")
        lbl = ctk.CTkLabel(row, text=value, text_color=ACCENT_2, font=ctk.CTkFont(size=12, weight="bold"))
        lbl.pack(side="right")
        return lbl

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        main.pack(side="left", fill="both", expand=True, padx=(18, 18), pady=18)

        # --- tarjetas de metricas ---
        cards = ctk.CTkFrame(main, fg_color="transparent")
        cards.pack(fill="x")
        for i in range(5):
            cards.grid_columnconfigure(i, weight=1, uniform="c")
        self.c_lat = MetricCard(cards, "Latencia promedio", "ms por frame", ACCENT)
        self.c_obj = MetricCard(cards, "Tiempo por objeto", "ms por detección", ACCENT_2)
        self.c_fps = MetricCard(cards, "FPS de inferencia", "frames / segundo", GOOD)
        self.c_cnt = MetricCard(cards, "Objetos detectados", "total acumulado", WARN)
        self.c_conf = MetricCard(cards, "Confianza media", "de las detecciones", "#f472b6")
        for i, c in enumerate([self.c_lat, self.c_obj, self.c_fps, self.c_cnt, self.c_conf]):
            c.grid(row=0, column=i, sticky="nsew", padx=5 if 0 < i < 4 else (0, 5) if i == 0 else (5, 0))

        # --- video + clases ---
        mid = ctk.CTkFrame(main, fg_color="transparent")
        mid.pack(fill="both", expand=True, pady=(12, 12))
        mid.grid_columnconfigure(0, weight=3)
        mid.grid_columnconfigure(1, weight=1, minsize=250)
        mid.grid_rowconfigure(0, weight=1)

        vcard = ctk.CTkFrame(mid, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER)
        vcard.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        head = ctk.CTkFrame(vcard, fg_color="transparent")
        head.pack(fill="x", padx=16, pady=(12, 0))
        ctk.CTkLabel(head, text="Vista en vivo", text_color=TEXT,
                     font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        self.badge = ctk.CTkLabel(head, text="  EN ESPERA  ", text_color=MUTED, fg_color=SURFACE,
                                  corner_radius=8, font=ctk.CTkFont(size=10, weight="bold"))
        self.badge.pack(side="right")
        self.video_label = ctk.CTkLabel(vcard, text="Selecciona un video para comenzar",
                                        text_color=MUTED, font=ctk.CTkFont(size=14))
        self.video_label.pack(fill="both", expand=True, padx=12, pady=12)

        ccard = ctk.CTkFrame(mid, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER)
        ccard.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ctk.CTkLabel(ccard, text="Clases detectadas", text_color=TEXT,
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=16, pady=(12, 6))
        self.class_rows = []
        for _ in range(8):
            r = ctk.CTkFrame(ccard, fg_color="transparent")
            r.pack(fill="x", padx=16, pady=3)
            name = ctk.CTkLabel(r, text="", text_color=TEXT, font=ctk.CTkFont(size=12))
            name.pack(anchor="w")
            bar = ctk.CTkProgressBar(r, height=6, progress_color=ACCENT, fg_color=SURFACE)
            bar.set(0)
            bar.pack(fill="x", pady=(2, 0))
            self.class_rows.append((name, bar))
        self.extra_stats = ctk.CTkLabel(ccard, text="", text_color=MUTED, justify="left",
                                        font=ctk.CTkFont(size=11))
        self.extra_stats.pack(side="bottom", anchor="w", padx=16, pady=14)

        # --- graficos: latencia (izquierda) y uso de CPU (derecha) ---
        gcard = ctk.CTkFrame(main, fg_color=CARD, corner_radius=16, border_width=1,
                             border_color=BORDER, height=230)
        gcard.pack(fill="x")
        gcard.pack_propagate(False)
        gcard.grid_columnconfigure(0, weight=3, uniform="g")
        gcard.grid_columnconfigure(1, weight=2, uniform="g")
        gcard.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(gcard, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(left, text="Latencia por frame (ms)", text_color=TEXT,
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=16, pady=(10, 0))
        self.fig = Figure(figsize=(6, 2), dpi=100, facecolor=CARD)
        self.ax = self.fig.add_subplot(111)
        self._style_axes(self.ax, self.fig)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().configure(bg=CARD, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 6))

        right = ctk.CTkFrame(gcard, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        head2 = ctk.CTkFrame(right, fg_color="transparent")
        head2.pack(fill="x", padx=16, pady=(10, 0))
        ctk.CTkLabel(head2, text="Uso de CPU (%)", text_color=TEXT,
                     font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        self.res_lbl = ctk.CTkLabel(head2, text="", text_color=MUTED, font=ctk.CTkFont(size=11))
        self.res_lbl.pack(side="right")
        self.fig2 = Figure(figsize=(4, 2), dpi=100, facecolor=CARD)
        self.ax2 = self.fig2.add_subplot(111)
        self._style_axes(self.ax2, self.fig2)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=right)
        self.canvas2.get_tk_widget().configure(bg=CARD, highlightthickness=0)
        self.canvas2.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 6))

        # --- barra de estado ---
        foot = ctk.CTkFrame(main, fg_color="transparent")
        foot.pack(fill="x", pady=(10, 0))
        self.status = ctk.CTkLabel(foot, text="Listo.", text_color=MUTED, font=ctk.CTkFont(size=12))
        self.status.pack(side="left")
        self.progress = ctk.CTkProgressBar(foot, width=360, height=8, progress_color=ACCENT_2, fg_color=CARD)
        self.progress.set(0)
        self.progress.pack(side="right")

    @staticmethod
    def _style_axes(ax, fig):
        ax.set_facecolor(CARD)
        for s in ax.spines.values():
            s.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=BORDER, alpha=0.5, linewidth=0.6)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.95, bottom=0.18)

    # ------------------------------------------------------------- acciones
    def choose_video(self):
        path = filedialog.askopenfilename(
            title="Selecciona un video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"), ("Todos", "*.*")])
        if path:
            self.video_path = path
            self.lbl_file.configure(text=path.replace("\\", "/").split("/")[-1], text_color=TEXT)
            self._preview_first_frame(path)

    def _preview_first_frame(self, path):
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            self._show_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _on_conf(self, v):
        self.conf_lbl.configure(text=f"{v:.2f}")
        if self.worker:
            self.worker.set_conf(float(v))

    def start(self):
        if not self.video_path:
            messagebox.showinfo("Falta el video", "Primero selecciona un video.")
            return
        out_path = None
        if self.save_var.get():
            stem = os.path.splitext(os.path.basename(self.video_path))[0]
            out_path = filedialog.asksaveasfilename(
                title="Guardar video procesado como...",
                defaultextension=".mp4", initialfile=f"{stem}_faster_rcnn.mp4",
                filetypes=[("Video MP4", "*.mp4")])
            if not out_path:  # el usuario cancelo el dialogo
                return
        self.stop()
        self._reset_metrics()
        self.saved_info = None
        self.q = queue.Queue()
        self.worker = DetectorWorker(
            self.video_path, MODELS[self.model_var.get()], float(self.conf_slider.get()),
            int(self.step_slider.get()), int(self.width_slider.get()), self.q,
            out_path=out_path)
        self.worker.start()
        self._start_monitor()
        self.t_start = time.perf_counter()
        self.btn_start.configure(state="disabled")
        self.btn_pause.configure(state="normal", text="⏸ Pausa")
        self.btn_stop.configure(state="normal")
        self.btn_export.configure(state="disabled")
        self._set_badge("CARGANDO", WARN)

    def toggle_pause(self):
        if not self.worker:
            return
        if self.worker.pause_event.is_set():
            self.worker.pause_event.clear()
            self.btn_pause.configure(text="▶ Reanudar")
            self._set_badge("PAUSADO", WARN)
        else:
            self.worker.pause_event.set()
            self.btn_pause.configure(text="⏸ Pausa")
            self._set_badge("PROCESANDO", GOOD)

    def stop(self):
        if self.worker and self.worker.is_alive():
            self.worker.stop_event.set()
            self.worker.pause_event.set()
        self.worker = None
        self._stop_monitor()
        self.btn_start.configure(state="normal")
        self.btn_pause.configure(state="disabled")
        self.btn_stop.configure(state="disabled")
        if self.times:
            self.btn_export.configure(state="normal")
        self._set_badge("EN ESPERA", MUTED)

    # ------------------------------------------------- monitoreo de CPU / RAM
    def _start_monitor(self):
        """Inicia el muestreo de CPU y RAM cada 500 ms (psutil)."""
        self._stop_monitor()
        psutil.cpu_percent(interval=None)   # la 1ra lectura siempre es 0: se descarta
        self.proc.cpu_percent(interval=None)
        self._mon_job = self.after(500, self._sample_resources)

    def _stop_monitor(self):
        if self._mon_job is not None:
            self.after_cancel(self._mon_job)
            self._mon_job = None

    def _sample_resources(self):
        sys_cpu = psutil.cpu_percent(interval=None)
        # cpu_percent() de un proceso suma todos los nucleos (puede pasar de 100):
        # se divide entre los nucleos logicos para tener un % sobre el total de la CPU.
        proc_cpu = min(self.proc.cpu_percent(interval=None) / self.ncpu, 100.0)
        ram = self.proc.memory_info().rss / (1024 ** 2)
        self.cpu_sys.append(sys_cpu)
        self.cpu_proc.append(proc_cpu)
        self.ram_mb.append(ram)
        self.res_lbl.configure(
            text=f"Proceso {proc_cpu:.0f} % · Sistema {sys_cpu:.0f} % · RAM {ram / 1024:.2f} GB")
        self._update_cpu_chart()
        self._mon_job = self.after(500, self._sample_resources)

    def _update_cpu_chart(self):
        ax = self.ax2
        ax.clear()
        self._style_axes(ax, self.fig2)
        sys_d, proc_d = self.cpu_sys[-120:], self.cpu_proc[-120:]
        xs = [i * 0.5 for i in range(len(self.cpu_sys) - len(sys_d), len(self.cpu_sys))]
        ax.plot(xs, sys_d, color=MUTED, linewidth=1.2, label="Sistema")
        ax.plot(xs, proc_d, color=ACCENT, linewidth=1.6, label="Este programa")
        ax.fill_between(xs, proc_d, color=ACCENT, alpha=0.15)
        ax.set_ylim(0, 100)
        ax.set_xlabel("segundos", color=MUTED, fontsize=8)
        ax.legend(loc="upper left", fontsize=7, frameon=True, facecolor=CARD,
                  edgecolor=BORDER, labelcolor=MUTED)
        self.canvas2.draw_idle()

    def _reset_metrics(self):
        self.times.clear()
        self.det_counts.clear()
        self.confs.clear()
        self.class_counter.clear()
        self.frame_log.clear()
        self.cpu_sys.clear()
        self.cpu_proc.clear()
        self.ram_mb.clear()
        self.res_lbl.configure(text="")
        self.ax.clear()
        self._style_axes(self.ax, self.fig)
        self.canvas.draw_idle()
        self.ax2.clear()
        self._style_axes(self.ax2, self.fig2)
        self.canvas2.draw_idle()
        for c in (self.c_lat, self.c_obj, self.c_fps, self.c_cnt, self.c_conf):
            c.set("—")
        for name, bar in self.class_rows:
            name.configure(text="")
            bar.set(0)
        self.extra_stats.configure(text="")
        self.progress.set(0)

    def _set_badge(self, text, color):
        self.badge.configure(text=f"  {text}  ", text_color=color)

    def _on_close(self):
        if self.worker:
            self.worker.stop_event.set()
            self.worker.pause_event.set()
        self.destroy()

    # --------------------------------------------------------- bucle de UI
    def _poll(self):
        updated = False
        try:
            for _ in range(6):  # procesa varios mensajes por tick
                kind, data = self.q.get_nowait()
                if kind == "frame":
                    self._on_frame(data)
                    updated = True
                elif kind == "status":
                    self.status.configure(text=data)
                elif kind == "info":
                    self.total_frames = data["total"]
                    self.status.configure(
                        text=f"Video: {data['total']} frames · {data['fps_src']:.1f} FPS · {data['device']}")
                    self._set_badge("PROCESANDO", GOOD)
                elif kind == "done":
                    self._on_done()
                elif kind == "saved":
                    self.saved_info = data
                    name = os.path.basename(data["path"])
                    self.status.configure(
                        text=f"Video procesado guardado: {name} ({data['frames']} frames)")
                    messagebox.showinfo("Video guardado",
                                        f"Video con cajas delimitadoras guardado en:\n{data['path']}")
                elif kind == "error":
                    self.stop()
                    messagebox.showerror("Error", data)
        except queue.Empty:
            pass
        self.after(10 if updated else 25, self._poll)

    def _on_frame(self, d):
        self._show_image(d["image"])
        self.times.append(d["time_ms"])
        self.det_counts.append(len(d["dets"]))
        for name, conf in d["dets"]:
            self.class_counter[name] += 1
            self.confs.append(conf)
        mean_conf = (sum(c for _, c in d["dets"]) / len(d["dets"])) if d["dets"] else 0.0
        # ultimo consumo de CPU medido (vacio si aun no hay muestra)
        cpu_p = round(self.cpu_proc[-1], 1) if self.cpu_proc else ""
        cpu_s = round(self.cpu_sys[-1], 1) if self.cpu_sys else ""
        self.frame_log.append((d["frame_idx"], round(d["time_ms"], 3), len(d["dets"]),
                               round(mean_conf, 4), cpu_p, cpu_s))

        n = len(self.times)
        avg = sum(self.times) / n
        total_objs = sum(self.det_counts)
        self.c_lat.set(f"{avg:.1f}", f"ms por frame · último {d['time_ms']:.0f} ms")
        self.c_fps.set(f"{1000.0 / avg:.2f}", "frames / segundo")
        self.c_cnt.set(f"{total_objs}", f"en {n} frames procesados")
        if total_objs:
            per_obj = sum(self.times) / total_objs
            self.c_obj.set(f"{per_obj:.1f}", "ms por detección")
            self.c_conf.set(f"{np.mean(self.confs) * 100:.1f}%", "de las detecciones")

        if d["total"]:
            self.progress.set(min(d["frame_idx"] / d["total"], 1.0))

        if n % 5 == 0 or n < 5:
            self._update_chart()
            self._update_classes()

    def _update_chart(self):
        ax = self.ax
        ax.clear()
        self._style_axes(ax, self.fig)
        data = self.times[-200:]
        xs = range(len(self.times) - len(data) + 1, len(self.times) + 1)
        ax.plot(xs, data, color=ACCENT_2, linewidth=1.4)
        ax.fill_between(xs, data, color=ACCENT_2, alpha=0.12)
        ax.axhline(sum(self.times) / len(self.times), color=WARN, linestyle="--", linewidth=1)
        ax.set_xlabel("frame procesado", color=MUTED, fontsize=8)
        self.canvas.draw_idle()

    def _update_classes(self):
        top = self.class_counter.most_common(len(self.class_rows))
        mx = top[0][1] if top else 1
        for i, (name_lbl, bar) in enumerate(self.class_rows):
            if i < len(top):
                nm, cnt = top[i]
                r, g, b = color_for(nm)
                name_lbl.configure(text=f"{nm}  ·  {cnt}")
                bar.configure(progress_color=f"#{r:02x}{g:02x}{b:02x}")
                bar.set(cnt / mx)
            else:
                name_lbl.configure(text="")
                bar.set(0)
        if self.times:
            arr = np.array(self.times)
            self.extra_stats.configure(
                text=(f"Mín: {arr.min():.1f} ms\nMáx: {arr.max():.1f} ms\n"
                      f"Desv. est.: {arr.std():.1f} ms\nP95: {np.percentile(arr, 95):.1f} ms\n"
                      f"Obj. por frame: {np.mean(self.det_counts):.2f}"))

    def _on_done(self):
        self._update_chart()
        self._update_classes()
        self.progress.set(1)
        elapsed = time.perf_counter() - (self.t_start or time.perf_counter())
        cpu_txt = ""
        if self.cpu_proc:
            cpu_txt = (f" · CPU promedio {np.mean(self.cpu_proc):.0f} % "
                       f"· RAM pico {np.max(self.ram_mb) / 1024:.2f} GB")
        self.status.configure(
            text=f"Completado: {len(self.times)} frames en {elapsed:.1f} s · "
                 f"latencia media {sum(self.times) / max(len(self.times), 1):.1f} ms{cpu_txt}")
        self.stop()
        self._set_badge("FINALIZADO", ACCENT_2)

    def _show_image(self, rgb):
        self.update_idletasks()
        w = max(self.video_label.winfo_width() - 8, 320)
        h = max(self.video_label.winfo_height() - 8, 240)
        img = Image.fromarray(rgb)
        img.thumbnail((w, h), Image.BILINEAR)
        self._photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.video_label.configure(image=self._photo, text="")

    def export_csv(self):
        if not self.frame_log:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="metricas_faster_rcnn.csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        arr = np.array(self.times)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "tiempo_inferencia_ms", "n_objetos", "confianza_media",
                        "cpu_programa_pct", "cpu_sistema_pct"])
            w.writerows(self.frame_log)
            w.writerow([])
            w.writerow(["RESUMEN"])
            w.writerow(["modelo", self.model_var.get()])
            w.writerow(["frames_procesados", len(arr)])
            w.writerow(["latencia_promedio_ms", f"{arr.mean():.3f}"])
            w.writerow(["latencia_p95_ms", f"{np.percentile(arr, 95):.3f}"])
            w.writerow(["fps_inferencia", f"{1000.0 / arr.mean():.3f}"])
            tot = sum(self.det_counts)
            w.writerow(["objetos_totales", tot])
            w.writerow(["tiempo_por_objeto_ms", f"{(arr.sum() / tot):.3f}" if tot else "n/a"])
            w.writerow(["conteo_por_clase", dict(self.class_counter)])
            if self.cpu_proc:
                w.writerow(["cpu_programa_promedio_pct", f"{np.mean(self.cpu_proc):.1f}"])
                w.writerow(["cpu_programa_pico_pct", f"{np.max(self.cpu_proc):.1f}"])
                w.writerow(["cpu_sistema_promedio_pct", f"{np.mean(self.cpu_sys):.1f}"])
                w.writerow(["ram_programa_pico_mb", f"{np.max(self.ram_mb):.0f}"])
                w.writerow(["nucleos_logicos_cpu", self.ncpu])
            if self.saved_info:
                w.writerow(["video_procesado", self.saved_info["path"]])
        messagebox.showinfo("Exportado", f"Métricas guardadas en:\n{path}")


if __name__ == "__main__":
    App().mainloop()
