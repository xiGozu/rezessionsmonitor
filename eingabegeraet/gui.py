"""GUI für den DIY-Eingabegerät-Treiber.

Start:  python gui.py
Not-Aus: F12 (global, falls pynput verfügbar) oder Esc im Fenster.
"""

import queue
import time
import tkinter as tk
from tkinter import messagebox, ttk

from treiber import Runner, available_backends

NOTAUS_TASTE = "F12"


class App:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.runner = None
        self.backend = None
        self.next_tab_at = None
        self.backends = available_backends()

        root.title("DIY-Eingabegerät – Testtreiber")
        root.minsize(460, 520)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<Escape>", lambda _e: self.stop())

        self._build_ui()
        self._start_hotkey()
        self.root.after(50, self._poll)

    # ---------- Aufbau ----------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Treiber / Backend").grid(row=0, column=0, sticky="w", **pad)
        self.backend_var = tk.StringVar(value=next(iter(self.backends)))
        self.backend_box = ttk.Combobox(frm, textvariable=self.backend_var, state="readonly",
                                        values=list(self.backends), width=36)
        self.backend_box.grid(row=0, column=1, columnspan=3, sticky="we", **pad)

        actions = ttk.LabelFrame(frm, text="Aktionen", padding=8)
        actions.grid(row=1, column=0, columnspan=4, sticky="we", **pad)
        self.left_var = tk.BooleanVar(value=True)
        self.right_var = tk.BooleanVar(value=True)
        self.tab_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(actions, text="Linke Maustaste gedrückt halten",
                        variable=self.left_var).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Checkbutton(actions, text="Rechte Maustaste gedrückt halten",
                        variable=self.right_var).grid(row=1, column=0, columnspan=4, sticky="w")
        ttk.Checkbutton(actions, text="Tabulator drücken, zufällig alle",
                        variable=self.tab_var).grid(row=2, column=0, sticky="w")
        self.min_var = tk.DoubleVar(value=4.0)
        self.max_var = tk.DoubleVar(value=8.0)
        ttk.Spinbox(actions, from_=0.1, to=3600, increment=0.5, width=6,
                    textvariable=self.min_var).grid(row=2, column=1, padx=4)
        ttk.Label(actions, text="bis").grid(row=2, column=2)
        ttk.Spinbox(actions, from_=0.1, to=3600, increment=0.5, width=6,
                    textvariable=self.max_var).grid(row=2, column=3, padx=4)
        ttk.Label(actions, text="s").grid(row=2, column=4, sticky="w")

        ttk.Label(frm, text="Startverzögerung (s)").grid(row=2, column=0, sticky="w", **pad)
        self.delay_var = tk.IntVar(value=3)
        ttk.Spinbox(frm, from_=0, to=60, width=6, textvariable=self.delay_var).grid(
            row=2, column=1, sticky="w", **pad)

        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=4, sticky="we", **pad)
        self.start_btn = ttk.Button(btns, text="▶ Start", command=self.start)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.stop_btn = ttk.Button(btns, text=f"■ Stop ({NOTAUS_TASTE} / Esc)",
                                   command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(4, 0))

        status = ttk.LabelFrame(frm, text="Status", padding=8)
        status.grid(row=4, column=0, columnspan=4, sticky="we", **pad)
        self.status_var = tk.StringVar(value="Bereit")
        self.next_var = tk.StringVar(value="–")
        self.count_var = tk.StringVar(value="0")
        for i, (label, var) in enumerate((("Zustand:", self.status_var),
                                          ("Nächster Tab in:", self.next_var),
                                          ("Tabs gesendet:", self.count_var))):
            ttk.Label(status, text=label).grid(row=i, column=0, sticky="w")
            ttk.Label(status, textvariable=var, font=("TkDefaultFont", 10, "bold")).grid(
                row=i, column=1, sticky="w", padx=8)

        self.log = tk.Text(frm, height=10, state="disabled", wrap="word")
        self.log.grid(row=5, column=0, columnspan=4, sticky="nsew", **pad)
        frm.rowconfigure(5, weight=1)
        frm.columnconfigure(1, weight=1)

    # ---------- Not-Aus ----------
    def _start_hotkey(self):
        try:
            from pynput import keyboard

            def on_press(key):
                if key == keyboard.Key.f12:
                    self.events.put(("hotkey", None))

            self.listener = keyboard.Listener(on_press=on_press, daemon=True)
            self.listener.start()
            self._log(f"Globaler Not-Aus aktiv: {NOTAUS_TASTE}")
        except Exception as exc:
            self.listener = None
            self._log(f"Globaler Not-Aus nicht verfügbar ({exc}). Esc im Fenster nutzen.")

    # ---------- Steuerung ----------
    def start(self):
        if self.runner and self.runner.alive:
            return
        try:
            tab_min, tab_max = float(self.min_var.get()), float(self.max_var.get())
            delay = int(self.delay_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Eingabe", "Bitte gültige Zahlen eingeben.")
            return
        if not (self.left_var.get() or self.right_var.get() or self.tab_var.get()):
            messagebox.showinfo("Eingabe", "Mindestens eine Aktion auswählen.")
            return

        try:
            backend_cls = self.backends[self.backend_var.get()]
            self.backend = backend_cls(self._log_threadsafe)
            self.runner = Runner(
                self.backend, lambda kind, value: self.events.put((kind, value)),
                hold_left=self.left_var.get(), hold_right=self.right_var.get(),
                tab_enabled=self.tab_var.get(), tab_min=tab_min, tab_max=tab_max,
                start_delay=delay,
            )
        except PermissionError:
            self._close_backend()
            messagebox.showerror("Rechte fehlen",
                                 "Kein Zugriff auf /dev/uinput.\nSiehe README: udev-Regel / Gruppe 'input'.")
            return
        except Exception as exc:
            self._close_backend()
            messagebox.showerror("Start fehlgeschlagen", f"{type(exc).__name__}: {exc}")
            return

        self.count_var.set("0")
        self._set_running(True)
        self._log(f"Start mit Backend: {self.backend.name}")
        self.runner.start()

    def stop(self):
        if self.runner and self.runner.alive:
            self.status_var.set("Stoppe …")
            self.runner.stop()

    def _set_running(self, running):
        self.start_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        self.backend_box.configure(state="disabled" if running else "readonly")

    def _close_backend(self):
        if self.backend:
            try:
                self.backend.close()
            except Exception:
                pass
        self.backend = None

    # ---------- Ereignisse aus dem Worker ----------
    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                self._handle(kind, value)
        except queue.Empty:
            pass
        if self.next_tab_at is not None:
            self.next_var.set(f"{max(0.0, self.next_tab_at - time.monotonic()):.1f} s")
        self.root.after(50, self._poll)

    def _handle(self, kind, value):
        if kind == "hotkey":
            self._log(f"{NOTAUS_TASTE} – Not-Aus")
            self.stop()
        elif kind == "countdown":
            self.status_var.set(f"Startet in {value} s – Zielfenster jetzt aktivieren!")
        elif kind == "running":
            self.status_var.set("Läuft")
        elif kind == "next_tab":
            self.next_tab_at = time.monotonic() + value
        elif kind == "tab":
            self.count_var.set(str(value))
        elif kind == "log":
            self._log(value)
        elif kind == "error":
            self._log(f"FEHLER: {value}")
        elif kind == "stopped":
            self.next_tab_at = None
            self.next_var.set("–")
            self.status_var.set("Gestoppt – alle Tasten losgelassen")
            self._close_backend()
            self._set_running(False)

    def _log_threadsafe(self, text):
        self.events.put(("log", text))

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", f"{time.strftime('%H:%M:%S')}  {text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def on_close(self):
        if self.runner and self.runner.alive:
            self.runner.stop()
            self.runner.join(timeout=2)  # sorgt dafür, dass die Maustasten losgelassen werden
        self._close_backend()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
