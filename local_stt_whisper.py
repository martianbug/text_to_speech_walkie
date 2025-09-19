# local_stt_whisper.py
import threading
import queue
import tempfile
import time
import traceback

import numpy as np
import soundfile as sf
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import sounddevice as sd

from faster_whisper import WhisperModel
from vad_segmenter import VADAudio
import os
# ---------------- Config ----------------
MODEL_SIZE = "small"        # "tiny", "base", "small", ...
DEVICE = "cpu"             # "cuda" si tienes GPU
LANG = "en"                
BEAM_SIZE = 5              
SEGMENT_QUEUE_MAX = 8      

class TranscriptionWorker(threading.Thread):
    """
    Worker que carga el modelo en run() (no en __init__) y consume items de la cola.
    Cada item esperado: (sid, audio_np, start_ts, end_ts)
    """
    def __init__(self, segment_queue, ui_append_func, model_size=MODEL_SIZE, device=DEVICE, lang=LANG, beam_size=BEAM_SIZE):
        super().__init__(daemon=True)
        self.segment_queue = segment_queue
        self.ui_append_func = ui_append_func
        self._stop_event = threading.Event()
        self.model = None
        self.model_ready_event = threading.Event()
        self.model_size = model_size
        self.device = device
        self.lang = lang
        self.beam_size = beam_size

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            self.ui_append_func("[Worker] Cargando modelo...\n")
            compute_type = None
            if self.device != "cpu":
                compute_type = "int8_float16"
            self.model = WhisperModel(self.model_size, device=self.device)
            self.ui_append_func("[Worker] Modelo cargado. Habla ahora\n\n")
            self.model_ready_event.set()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.ui_append_func(f"[ERROR] No se pudo cargar el modelo: {e}\n{tb}\n")
            return

        while not self._stop_event.is_set():
            try:
                item = self.segment_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # ---------------- Normalizar item ----------------
                sid = "(no-id)"
                audio_np = None
                start_ts = None
                end_ts = None

                if isinstance(item, (tuple, list)):
                    if len(item) >= 2 and isinstance(item[0], (str, int)):
                        sid = item[0]
                        audio_np = item[1]
                        if len(item) >= 3:
                            start_ts = item[2]
                        if len(item) >= 4:
                            end_ts = item[3]
                    else:
                        audio_np = item[0]
                        if len(item) >= 2:
                            start_ts = item[1]
                        if len(item) >= 3:
                            end_ts = item[2]
                        sid = "(no-id)"
                else:
                    audio_np = item
                    sid = "(no-id)"

                try:
                    audio_np = np.asarray(audio_np)
                    if audio_np.dtype == np.int16:
                        audio_np = audio_np.astype(np.float32) / 32768.0
                    else:
                        audio_np = audio_np.astype(np.float32)
                    if audio_np.ndim > 1:
                        audio_np = audio_np.reshape(-1)

                    duration_s = audio_np.shape[0] / 16000.0
                    self.ui_append_func(f"[Worker] Transcribiendo segmento {sid}: dur={duration_s:.2f}s\n")

                    # --- transcripción usando numpy directo ---
                    segments, info = self.model.transcribe(audio_np, language=self.lang, beam_size=self.beam_size)
                    text = ""
                    for r in segments:
                        if hasattr(r, "text"):
                            text += r.text
                        elif isinstance(r, dict) and "text" in r:
                            text += r["text"]

                    start_s = f"{start_ts:.2f}s" if start_ts is not None else "?"
                    end_s = f"{end_ts:.2f}s" if end_ts is not None else "?"
                    out = f"[{start_s} - {end_s} | seg {sid}] {text.strip() or '(no text)'}\n"
                    # self.ui_append_func(out)
                    self.ui_append_func("\n"+text.strip())
                    
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    self.ui_append_func(f"[ERROR] worker procesando segmento {sid}: {e}\n{tb}\n")
                finally:
                    try:
                        self.segment_queue.task_done()
                    except Exception:
                        pass


                # seg_time = time.time() - t0
                # # ---------------- Mostrar resultado en UI ----------------
                # start_s = f"{start_ts:.2f}s" if (start_ts is not None) else "?"
                # end_s = f"{end_ts:.2f}s" if (end_ts is not None) else "?"
                # out = f"[{start_s} - {end_s} | seg {sid} | {seg_time:.2f}s] {text.strip() or '(no text)'}\n"
                # self.ui_append_func(out)

            except Exception as worker_exc:
                import traceback
                tb = traceback.format_exc()
                self.ui_append_func(f"[ERROR worker procesando item] {worker_exc}\n{tb}\n")
            finally:
                try:
                    self.segment_queue.task_done()
                except Exception:
                    pass

        # fin loop
        self.ui_append_func("[Worker] Terminando.\n")

class App:
    def __init__(self, root):
        self.root = root
        root.title("Live STT local — faster-whisper + VAD (es)")
        self.text = ScrolledText(root, wrap=tk.WORD, height=20, width=90, font=("Helvetica", 13))
        self.text.pack(padx=10, pady=10)
        btn_frame = tk.Frame(root)
        btn_frame.pack(padx=10, pady=(0,10))
        self.start_btn = tk.Button(btn_frame, text="Iniciar (Local)", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(btn_frame, text="Detener", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.segment_queue = queue.Queue(maxsize=SEGMENT_QUEUE_MAX)
        self.vad = None
        self.worker = None
        self._segment_counter = 0

    def append_text(self, txt):
        def _append():
            self.text.insert(tk.END, txt)
            self.text.see(tk.END)
        self.root.after(0, _append)

    def start(self):
        # Remove all widgets from the window (including any Exit button)
        for widget in self.root.winfo_children():
            widget.pack_forget()

        # Recreate UI elements
        self.text = ScrolledText(self.root, wrap=tk.WORD, height=20, width=90, font=("Helvetica", 13))
        self.text.pack(padx=10, pady=10)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(padx=10, pady=(0,10))
        self.start_btn = tk.Button(btn_frame, text="Iniciar (Local)", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(btn_frame, text="Detener", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        exit_button = tk.Button(self.root, text="Exit", command=lambda: [exit_button.pack_forget(), self.root.destroy()])
        exit_button.pack(pady=20)
        
        self.worker = TranscriptionWorker(self.segment_queue, self.append_text)
        self.worker.start()
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.text.delete("1.0", tk.END)

        # iniciar VAD
        self.vad = VADAudio(
            aggressiveness=3,
            input_sample_rate=48000,
            vad_sample_rate=16000,
            frame_duration_ms=30,
            padding_duration_ms=300,
        )
        self.vad.start_stream(samplerate=48000, channels=1)

        # lanzar hilo productor que pujee segmentos en la cola
        self._producer_thread = threading.Thread(target=self._segment_producer, daemon=True)
        self._producer_thread.start()
        self.append_text("Starting Program\n")

    def _segment_producer(self):
        """
        Producer de debug: imprime exactamente lo que sale de vad.vad_collector()
        y encola TODO tal cual (sin merge). Úsalo para comprobar la forma/tipos.
        """
        import numpy as np
        import time

        for seg in self.vad.vad_collector():
            # Normalizar al formato (sid, audio_np, start, end)
            sid = self._segment_counter
            self._segment_counter += 1
            audio_np = None
            start_ts = None
            end_ts = None

            if isinstance(seg, (tuple, list)):
                if len(seg) >= 1:
                    audio_np = seg[0]
                if len(seg) >= 2:
                    start_ts = seg[1]
                if len(seg) >= 3:
                    end_ts = seg[2]
            else:
                audio_np = seg


            # encolar tal cual (worker hará conversiones finales)
            try:
                self.segment_queue.put_nowait((sid, audio_np, start_ts, end_ts))
            except queue.Full:
                try:
                    _ = self.segment_queue.get_nowait()
                    self.segment_queue.task_done()
                except Exception:
                    pass
                try:
                    self.segment_queue.put_nowait((sid, audio_np, start_ts, end_ts))
                except Exception:
                    self.append_text(f"[WARN] cola llena, descartando segmento {sid}\n")

            # pequeña pausa
            time.sleep(0.01)

        self.append_text("Producer terminado (VAD detenido).\n")

    def stop(self):
            self.append_text("Deteniendo...\n")
            if self.vad is not None:
                self.vad.stop_stream()
            if self.worker is not None:
                self.worker.stop()
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.append_text("Detenido.\n")



if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [app.stop(), root.destroy()])
    root.mainloop()
    
