import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import math

# ─── Theme ────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG        = "#0F1117"  
CARD_BG   = "#1A1D27"
ACCENT    = "#8173FD"
TEXT_PRI  = "#EAEAEA"
TEXT_SEC  = "#7A7F8E"
BORDER    = "#313A4B"

EMOTION_CONFIG = {
    "Happy":     {"emoji": "😊", "color": "#FFD166"},
    "Sad":       {"emoji": "😢", "color": "#74B0FF"},
    "Angry":     {"emoji": "😠", "color": "#FF6B6B"},
    "Surprised": {"emoji": "😲", "color": "#A78BFA"},
    "Fearful":   {"emoji": "😨", "color": "#94D2BD"},
    "Disgusted": {"emoji": "🤢", "color": "#6DBF67"},
    "Neutral":   {"emoji": "😐", "color": "#A0A0A0"},
}

TAGLINE   = "Your face tells a story.\nWe just need to read it."
EMOJIS    = ["😊", "😢", "😠", "😲", "😨", "🤢", "😐"]

#  HOME PAGE
class HomePage(ctk.CTkFrame):
    def __init__(self, master, on_launch):
        super().__init__(master, fg_color=BG, corner_radius=0)
        self.on_launch      = on_launch
        self._anim_step     = 0
        self._float_step    = 0
        self._typing_index  = 0
        self._typed_text    = ""
        self._full_text     = TAGLINE
        self._anim_running  = True

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build()
        self._start_animations()

    # ── Build ──────────────────────────────────────────────────────────────────
    def _build(self):
        center = ctk.CTkFrame(self, fg_color="transparent")
        center.grid(row=0, column=0)

        # Floating emoji ring (canvas)
        self.canvas = ctk.CTkCanvas(
            center, width=320, height=320,
            bg=BG, highlightthickness=0,
        )
        self.canvas.pack(pady=(0, 0))

        # logo
        self.center_emoji_id = self.canvas.create_text(
            160, 155, text="🎭",
            font=("Segoe UI", 72), fill=TEXT_PRI,
        )
        # Title
        self.title_label = ctk.CTkLabel(
            center,
            text="EmoSync",
            font=ctk.CTkFont("Segoe UI", 38, weight="bold"),
            text_color=TEXT_PRI,
        )
        self.title_label.pack(pady=(0, 6))

        # underline strip
        strip = ctk.CTkFrame(center, fg_color=ACCENT, height=3, width=200, corner_radius=2)
        strip.pack(pady=(0, 18))

        # Tagline (typing effect)
        self.tagline_label = ctk.CTkLabel(
            center,
            text="",
            font=ctk.CTkFont("Segoe UI", 16),
            text_color=TEXT_SEC,
            justify="center",
        )
        self.tagline_label.pack(pady=(0, 36))

        # Launch button
        self.launch_btn = ctk.CTkButton(
            center,
            text="  Launch Detector  ▶",
            font=ctk.CTkFont("Segoe UI", 15, weight="bold"),
            fg_color=ACCENT,
            hover_color="#6056D6",
            corner_radius=30,
            height=48,
            width=220,
            command=self._launch,
        )
        self.launch_btn.pack(pady=(0, 10))

        ctk.CTkLabel(
            center,
            text="Detects emotions in real time through your webcam",
            font=ctk.CTkFont("Segoe UI", 11),
            text_color=TEXT_SEC,
        ).pack()

    # ── Animations ───
    def _start_animations(self):
        self._animate_orbit()
        self._animate_typing()

    def _animate_orbit(self):
        if not self._anim_running:
            return
        self.canvas.delete("orbit")

        cx, cy = 160, 155
        radius = 118
        count  = len(EMOJIS)

        for i, emoji in enumerate(EMOJIS):
            angle = (2 * math.pi / count) * i + self._float_step
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            # Size pulse: emoji nearest top is slightly bigger
            scale = 1.0 + 0.18 * math.sin(angle - math.pi / 2 + self._float_step)
            size  = int(22 + 10 * scale)
            self.canvas.create_text(
                x, y, text=emoji,
                font=("Segoe UI", size),
                fill="#8173FD",
                tags="orbit",
            )

        self._float_step += 0.025
        self.after(40, self._animate_orbit)

    def _animate_typing(self):
        if not self._anim_running:
            return
        if self._typing_index <= len(self._full_text):
            self._typed_text = self._full_text[:self._typing_index]
            # blinking cursor
            cursor = "▌" if int(time.time() * 2) % 2 == 0 else ""
            self.tagline_label.configure(text=self._typed_text + cursor)
            self._typing_index += 1
            self.after(42, self._animate_typing)
        else:
            # just blink cursor at end
            cursor = "▌" if int(time.time() * 2) % 2 == 0 else ""
            self.tagline_label.configure(text=self._full_text + cursor)
            self.after(500, self._animate_typing)

    def _launch(self):
        self._anim_running = False
        self.on_launch()

    def destroy(self):
        self._anim_running = False
        super().destroy()

#  DETECTOR PAGE

class DetectorPage(ctk.CTkFrame):
    def __init__(self, master, on_back):
        super().__init__(master, fg_color=BG, corner_radius=0)
        self.on_back = on_back

        self.is_running      = False
        self.is_muted        = False
        self.cap             = None
        self.camera_thread   = None
        self.emotion_history = []

        self._pending_emotion    = None
        self._pending_confidence = None
        self._emotion_lock       = threading.Lock()

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        self._build_header()
        self._build_camera_panel()
        self._build_info_panel()
        self._build_controls()
        self._poll_emotion_updates()

    # ── Header ──
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=0, height=56)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        # Back button
        ctk.CTkButton(
            header,
            text="← Home",
            font=ctk.CTkFont("Segoe UI", 13),
            fg_color="transparent",
            hover_color=BORDER,
            text_color=TEXT_SEC,
            corner_radius=8,
            height=32, width=80,
            command=self._go_back,
        ).grid(row=0, column=0, padx=(12, 0), pady=12, sticky="w")

        ctk.CTkLabel(
            header,
            text="🎭  EmoSync",
            font=ctk.CTkFont("Segoe UI", 20, weight="bold"),
            text_color=TEXT_PRI,
        ).grid(row=0, column=1, pady=14)

        self.status_dot = ctk.CTkLabel(
            header, text="⬤  Idle",
            font=ctk.CTkFont("Segoe UI", 13),
            text_color=TEXT_SEC,
        )
        self.status_dot.grid(row=0, column=2, padx=24, pady=14, sticky="e")

    # ── Camera Panel ──
    def _build_camera_panel(self):
        cam_frame = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=12)
        cam_frame.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=12)
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        self.cam_label = ctk.CTkLabel(
            cam_frame,
            text="📷\n\nPress  Start Detection\nto begin",
            font=ctk.CTkFont("Segoe UI", 15),
            text_color=TEXT_SEC,
            fg_color="#12141C",
            corner_radius=8,
        )
        self.cam_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

    # ── Info Panel ──
    def _build_info_panel(self):
        panel = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=12)
        panel.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=12)
        panel.grid_columnconfigure(0, weight=1)

        # Emotion card
        emo_card = ctk.CTkFrame(panel, fg_color="#12141C", corner_radius=10)
        emo_card.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
        emo_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            emo_card, text="CURRENT EMOTION",
            font=ctk.CTkFont("Segoe UI", 10, weight="bold"),
            text_color=TEXT_SEC,
        ).grid(row=0, column=0, pady=(12, 2))

        self.emoji_label = ctk.CTkLabel(
            emo_card, text="😐",
            font=ctk.CTkFont("Segoe UI", 52),
        )
        self.emoji_label.grid(row=1, column=0, pady=(4, 0))

        self.emotion_label = ctk.CTkLabel(
            emo_card, text="Neutral",
            font=ctk.CTkFont("Segoe UI", 22, weight="bold"),
            text_color=EMOTION_CONFIG["Neutral"]["color"],
        )
        self.emotion_label.grid(row=2, column=0, pady=(2, 12))

        # Confidence
        conf_frame = ctk.CTkFrame(panel, fg_color="transparent")
        conf_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=4)
        conf_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            conf_frame, text="CONFIDENCE",
            font=ctk.CTkFont("Segoe UI", 10, weight="bold"),
            text_color=TEXT_SEC,
        ).grid(row=0, column=0, sticky="w")

        self.conf_pct_label = ctk.CTkLabel(
            conf_frame, text="0%",
            font=ctk.CTkFont("Segoe UI", 10, weight="bold"),
            text_color=ACCENT,
        )
        self.conf_pct_label.grid(row=0, column=1, sticky="e")

        self.conf_bar = ctk.CTkProgressBar(
            conf_frame, height=8,
            fg_color=BORDER, progress_color=ACCENT,
        )
        self.conf_bar.set(0)
        self.conf_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        # Divider
        ctk.CTkFrame(panel, fg_color=BORDER, height=1).grid(
            row=2, column=0, sticky="ew", padx=14, pady=10
        )

        # Volume
        vol_frame = ctk.CTkFrame(panel, fg_color="transparent")
        vol_frame.grid(row=3, column=0, sticky="ew", padx=14, pady=2)
        vol_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            vol_frame, text="🔊",
            font=ctk.CTkFont("Segoe UI", 16),
        ).grid(row=0, column=0, padx=(0, 8))

        self.volume_slider = ctk.CTkSlider(
            vol_frame, from_=0, to=100,
            fg_color=BORDER, progress_color=ACCENT,
            button_color=ACCENT, button_hover_color="#9D96FF",
            command=self._on_volume_change,
        )
        self.volume_slider.set(70)
        self.volume_slider.grid(row=0, column=1, sticky="ew")

        self.vol_label = ctk.CTkLabel(
            vol_frame, text="70",
            font=ctk.CTkFont("Segoe UI", 11),
            text_color=TEXT_SEC, width=28,
        )
        self.vol_label.grid(row=0, column=2, padx=(8, 0))

        # Mute
        self.mute_btn = ctk.CTkButton(
            panel, text="🔇  Mute Sound",
            font=ctk.CTkFont("Segoe UI", 13),
            fg_color=BORDER, hover_color="#2F3244",
            text_color=TEXT_PRI, corner_radius=8, height=34,
            command=self._toggle_mute,
        )
        self.mute_btn.grid(row=4, column=0, sticky="ew", padx=14, pady=(8, 4))

        # Divider
        ctk.CTkFrame(panel, fg_color=BORDER, height=1).grid(
            row=5, column=0, sticky="ew", padx=14, pady=10
        )

        # History
        ctk.CTkLabel(
            panel, text="RECENT HISTORY",
            font=ctk.CTkFont("Segoe UI", 10, weight="bold"),
            text_color=TEXT_SEC,
        ).grid(row=6, column=0, sticky="w", padx=14)

        self.history_frame = ctk.CTkScrollableFrame(
            panel, fg_color="#12141C", corner_radius=8, height=130,
        )
        self.history_frame.grid(row=7, column=0, sticky="nsew", padx=14, pady=(6, 14))
        self.history_frame.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(7, weight=1)

        ctk.CTkLabel(
            self.history_frame,
            text="No emotions detected yet",
            font=ctk.CTkFont("Segoe UI", 12),
            text_color=TEXT_SEC,
        ).pack(pady=20)

    # ── Controls bar ──
    def _build_controls(self):
        bar = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=0, height=64)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure((0, 1, 2), weight=1)

        self.start_btn = ctk.CTkButton(
            bar, text="▶   Start Detection",
            font=ctk.CTkFont("Segoe UI", 14, weight="bold"),
            fg_color=ACCENT, hover_color="#6056D6",
            corner_radius=8, height=38, width=200,
            command=self._start_detection,
        )
        self.start_btn.grid(row=0, column=1, pady=13, padx=8)

        self.stop_btn = ctk.CTkButton(
            bar, text="■   Stop",
            font=ctk.CTkFont("Segoe UI", 14),
            fg_color="#3A1E1E", hover_color="#5A2A2A",
            text_color="#FF7E7E", corner_radius=8, height=38, width=120,
            command=self._stop_detection,
            state="disabled",
        )
        self.stop_btn.grid(row=0, column=2, pady=13, padx=(0, 24), sticky="e")

    # ── Camera loop ──
    def _camera_loop(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.after(0, self._show_cam_error)
            return
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame     = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img       = Image.fromarray(frame_rgb)
            lw = self.cam_label.winfo_width()  or 580
            lh = self.cam_label.winfo_height() or 400
            img.thumbnail((lw - 8, lh - 8), Image.LANCZOS)
            ctk_img = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=ctk_img, text="")
            self.cam_label.image = ctk_img
            time.sleep(0.03)
        if self.cap:
            self.cap.release()
        self.cap = None

    def _show_cam_error(self):
        self.cam_label.configure(
            image="", text="⚠️  Camera not found.\nCheck your webcam connection.",
            text_color="#FF6B6B",
        )
        self._set_idle_state()

    # ── Emotion API ──
    def update_emotion(self, emotion: str, confidence: float):
        with self._emotion_lock:
            self._pending_emotion    = emotion
            self._pending_confidence = confidence

    def _poll_emotion_updates(self):
        with self._emotion_lock:
            emotion    = self._pending_emotion
            confidence = self._pending_confidence
            self._pending_emotion    = None
            self._pending_confidence = None
        if emotion is not None and self.is_running:
            self._apply_emotion(emotion, confidence)
        self.after(100, self._poll_emotion_updates)

    def _apply_emotion(self, emotion: str, confidence: float):
        cfg   = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["Neutral"])
        color = cfg["color"]
        self.emoji_label.configure(text=cfg["emoji"])
        self.emotion_label.configure(text=emotion, text_color=color)
        self.conf_bar.configure(progress_color=color)
        self.conf_bar.set(confidence)
        self.conf_pct_label.configure(text=f"{int(confidence * 100)}%")
        ts = datetime.now().strftime("%H:%M:%S")
        self.emotion_history.insert(0, (ts, emotion, cfg["emoji"], color))
        if len(self.emotion_history) > 20:
            self.emotion_history.pop()
        self._refresh_history()

    def _refresh_history(self):
        for w in self.history_frame.winfo_children():
            w.destroy()
        if not self.emotion_history:
            ctk.CTkLabel(
                self.history_frame,
                text="No emotions detected yet",
                font=ctk.CTkFont("Segoe UI", 12),
                text_color=TEXT_SEC,
            ).pack(pady=20)
            return
        for ts, emotion, emoji, color in self.emotion_history:
            row = ctk.CTkFrame(self.history_frame, fg_color="transparent", height=28)
            row.pack(fill="x", padx=6, pady=1)
            row.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(row, text=ts, font=ctk.CTkFont("Segoe UI", 10),
                         text_color=TEXT_SEC, width=60).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(row, text=f"{emoji}  {emotion}",
                         font=ctk.CTkFont("Segoe UI", 12, weight="bold"),
                         text_color=color).grid(row=0, column=1, sticky="w", padx=(6, 0))

    # ── Controls ──
    def _start_detection(self):
        if self.is_running:
            return
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_dot.configure(text="⬤  Running", text_color="#6DBF67")
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

    def _stop_detection(self):
        self._set_idle_state()
        self.after(400, lambda: self.cam_label.configure(
            image="",
            text="📷\n\nPress  Start Detection\nto begin",
            text_color=TEXT_SEC,
        ))

    def _set_idle_state(self):
        self.is_running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_dot.configure(text="⬤  Idle", text_color=TEXT_SEC)

    def _toggle_mute(self):
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.mute_btn.configure(text="🔊  Unmute Sound",
                                    fg_color="#1E2A1E", text_color="#6DBF67")
        else:
            self.mute_btn.configure(text="🔇  Mute Sound",
                                    fg_color=BORDER, text_color=TEXT_PRI)

    def _on_volume_change(self, value):
        self.vol_label.configure(text=str(int(value)))

    def _go_back(self):
        if self.is_running:
            self._stop_detection()
        self.on_back()

    # ── Public API for teammates ──────────────────────────
    def get_volume(self) -> int:
        return int(self.volume_slider.get())

    def get_muted(self) -> bool:
        return self.is_muted


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP  (router between pages)
# ══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EmoSync AI")
        self.geometry("960x620")
        self.minsize(860, 560)
        self.configure(fg_color=BG)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._home_page     = None
        self._detector_page = None

        self._show_home()

    # ── Navigation ─────────────────────────────────────────────────────────────
    def _show_home(self):
        if self._detector_page:
            self._detector_page.destroy()
            self._detector_page = None

        self._home_page = HomePage(self, on_launch=self._show_detector)
        self._home_page.grid(row=0, column=0, sticky="nsew")

    def _show_detector(self):
        if self._home_page:
            self._home_page.destroy()
            self._home_page = None

        self._detector_page = DetectorPage(self, on_back=self._show_home)
        self._detector_page.grid(row=0, column=0, sticky="nsew")

    # ── Clean shutdown ─────────────────────────────────────────────────────────
    def on_closing(self):
        if self._detector_page and self._detector_page.is_running:
            self._detector_page.is_running = False
            time.sleep(0.12)
        self.destroy()


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()