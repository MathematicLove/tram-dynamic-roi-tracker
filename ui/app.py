"""Minimal launcher UI: choose Camera Mode or Video Mode, then run the tracker.

Run with:  python ui/app.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import (
    DISABLED,
    END,
    NORMAL,
    Button,
    Entry,
    Frame,
    Label,
    StringVar,
    Tk,
    filedialog,
    messagebox,
)
from tkinter.scrolledtext import ScrolledText

from validation import check_input_file, parse_camera_index, parse_speed

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    _HAS_DND = True
except ImportError:
    _HAS_DND = False

_ROOT = Path(__file__).resolve().parent.parent
DETECTION_SCRIPT = _ROOT / "algorithm" / "detection.py"
IMAGE_SCRIPT = _ROOT / "algorithm" / "image_infer.py"

BG = "white"
FG = "black"
BORDER = "black"


class App:
    def __init__(self) -> None:
        base_cls = TkinterDnD.Tk if _HAS_DND else Tk
        self.root = base_cls()
        self.root.title("Tram Dynamic ROI Tracker")
        self.root.geometry("640x420")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.mode: str | None = None
        self.input_path: str | None = None
        self.speed_var = StringVar(value="40")
        self.camera_index_var = StringVar(value="0")
        self.proc: subprocess.Popen | None = None

        self._build_mode_screen()

    def _clear(self) -> None:
        for w in self.root.winfo_children():
            w.destroy()

    def _label(self, parent, text, size=10, bold=False) -> Label:
        return Label(
            parent, text=text, bg=BG, fg=FG,
            font=("Helvetica", size, "bold" if bold else "normal"),
        )

    def _button(self, parent, text, command, state=NORMAL) -> Button:
        return Button(
            parent, text=text, command=command, bg=BG, fg=FG,
            activebackground=BG, activeforeground=FG,
            highlightbackground=BORDER, highlightthickness=1,
            relief="flat", borderwidth=1, padx=10, pady=4, state=state,
        )

    def _build_mode_screen(self) -> None:
        self._clear()
        self._label(self.root, "Tram Dynamic ROI Tracker", 16, bold=True).pack(pady=(30, 6))
        self._label(self.root, "Select mode").pack(pady=(0, 24))

        row = Frame(self.root, bg=BG)
        row.pack()

        self._mode_card(row, "Camera Mode", "Use a connected camera",
                         lambda: self._select_mode("camera")).pack(side="left", padx=10)
        self._mode_card(row, "Video Mode", "Process a video file",
                         lambda: self._select_mode("video")).pack(side="left", padx=10)
        self._mode_card(row, "Image Mode", "Process a single image",
                         lambda: self._select_mode("image")).pack(side="left", padx=10)

    def _mode_card(self, parent, title, subtitle, command) -> Frame:
        card = Frame(
            parent, bg=BG, width=200, height=140,
            highlightthickness=1, highlightbackground=BORDER,
        )
        card.pack_propagate(False)
        self._label(card, title, 12, bold=True).pack(pady=(20, 6))
        Label(
            card, text=subtitle, bg=BG, fg=FG, font=("Helvetica", 9),
            wraplength=170, justify="center",
        ).pack()
        self._button(card, "Select", command).pack(pady=16)
        return card

    def _select_mode(self, mode: str) -> None:
        self.mode = mode
        self.input_path = None
        if mode == "camera":
            self._build_camera_screen()
        elif mode == "video":
            self._build_video_screen()
        else:
            self._build_image_screen()

    def _back_button(self) -> None:
        self._button(self.root, "Back", self._build_mode_screen).pack(
            anchor="w", padx=16, pady=(12, 0)
        )

    def _speed_row(self) -> None:
        row = Frame(self.root, bg=BG)
        row.pack(pady=(10, 0))
        self._label(row, "Tram speed (km/h):").pack(side="left", padx=(0, 8))
        Entry(
            row, textvariable=self.speed_var, width=6, justify="center",
            bg=BG, fg=FG, highlightthickness=1, highlightbackground=BORDER,
            relief="flat",
        ).pack(side="left")

    def _build_camera_screen(self) -> None:
        self._clear()
        self._back_button()
        self._label(self.root, "Camera Mode", 14, bold=True).pack(pady=(10, 4))
        self._label(self.root, "The connected camera will be used").pack()

        row = Frame(self.root, bg=BG)
        row.pack(pady=(16, 0))
        self._label(row, "Camera index:").pack(side="left", padx=(0, 8))
        Entry(
            row, textvariable=self.camera_index_var, width=4, justify="center",
            bg=BG, fg=FG, highlightthickness=1, highlightbackground=BORDER,
            relief="flat",
        ).pack(side="left")

        self._speed_row()

        self._button(self.root, "Start", self._start).pack(pady=20)

        self._build_log_panel()

    def _build_video_screen(self) -> None:
        self._clear()
        self._back_button()
        self._label(self.root, "Video Mode", 14, bold=True).pack(pady=(10, 4))
        self._build_drop_area("Drop a video file here\nor click Browse",
                               "Click Browse to select a video file",
                               [("Video", "*.mp4 *.avi *.mov *.mkv *.m4v")])
        self._speed_row()
        self.start_btn = self._button(self.root, "Start", self._start, state=DISABLED)
        self.start_btn.pack(pady=16)
        self._build_log_panel()

    def _build_image_screen(self) -> None:
        self._clear()
        self._back_button()
        self._label(self.root, "Image Mode", 14, bold=True).pack(pady=(10, 4))
        self._build_drop_area("Drop an image file here\nor click Browse",
                               "Click Browse to select an image file",
                               [("Image", "*.jpg *.jpeg *.png *.bmp")])
        self.start_btn = self._button(self.root, "Start", self._start, state=DISABLED)
        self.start_btn.pack(pady=20)
        self._build_log_panel()

    def _build_drop_area(self, drop_text: str, browse_text: str, filetypes) -> None:
        text = drop_text if _HAS_DND else browse_text
        self.drop_label = Label(
            self.root, text=text, bg=BG, fg=FG, font=("Helvetica", 10),
            width=44, height=4, relief="flat",
            highlightthickness=1, highlightbackground=BORDER,
        )
        self.drop_label.pack(pady=(16, 8), padx=16)

        if _HAS_DND:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self._on_drop)

        self._button(
            self.root, "Browse", lambda: self._browse_file(filetypes)
        ).pack()

    def _build_log_panel(self) -> None:
        self.log = ScrolledText(
            self.root, height=7, bg=BG, fg=FG, insertbackground=FG,
            font=("Menlo", 9), relief="flat",
            highlightthickness=1, highlightbackground=BORDER,
        )
        self.log.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        self.log.configure(state=DISABLED)

        self.stop_btn = self._button(self.root, "Stop", self._stop, state=DISABLED)
        self.stop_btn.pack(pady=(0, 12))

    def _set_input_file(self, path: str) -> None:
        self.input_path = path
        self.drop_label.configure(text=Path(path).name)
        self.start_btn.configure(state=NORMAL)

    def _browse_file(self, filetypes) -> None:
        path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[*filetypes, ("All files", "*.*")],
        )
        if path:
            self._set_input_file(path)

    def _on_drop(self, event) -> None:
        raw = event.data
        path = raw.strip("{}") if raw.startswith("{") else raw.split()[0]
        if Path(path).is_file():
            self._set_input_file(path)

    def _append_log(self, text: str) -> None:
        self.log.configure(state=NORMAL)
        self.log.insert(END, text)
        self.log.see(END)
        self.log.configure(state=DISABLED)

    def _start(self) -> None:
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Already running", "Stop the current run first")
            return

        try:
            if self.mode == "image":
                image = check_input_file(self.input_path, "an image file")
                cmd = [
                    sys.executable, str(IMAGE_SCRIPT),
                    "--image", image,
                    "--show",
                ]
            else:
                speed = str(parse_speed(self.speed_var.get()))
                if self.mode == "camera":
                    video_arg = parse_camera_index(self.camera_index_var.get())
                else:
                    video_arg = check_input_file(self.input_path, "a video file")

                cmd = [
                    sys.executable, str(DETECTION_SCRIPT),
                    "--speed", speed,
                    "--video", video_arg,
                ]
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return

        self._append_log(f"$ {' '.join(cmd)}\n")
        self.stop_btn.configure(state=NORMAL)

        def run() -> None:
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            try:
                self.proc = subprocess.Popen(
                    cmd, cwd=str(_ROOT), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
            except OSError as exc:
                self.root.after(0, self._append_log, f"Failed to start: {exc}\n")
                self.root.after(0, self._on_finished)
                return
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.root.after(0, self._append_log, line)
            self.proc.wait()
            self.root.after(0, self._on_finished)

        threading.Thread(target=run, daemon=True).start()

    def _on_finished(self) -> None:
        self._append_log("\n[Process finished]\n")
        self.stop_btn.configure(state=DISABLED)

    def _stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.stop_btn.configure(state=DISABLED)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
