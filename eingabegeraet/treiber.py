"""Treiber-Schicht für das DIY-Eingabegerät.

Ein "Backend" ist die Stelle, an der Eingaben beim Betriebssystem ankommen:

* SimulationBackend  – schreibt nur ins Log (zum gefahrlosen Testen)
* PynputBackend      – sendet Eingaben über pynput (Windows, macOS, Linux/X11)
* UinputBackend      – legt unter Linux echte virtuelle Eingabegeräte
                       (/dev/uinput) an; der Kernel sieht sie wie USB-Geräte

Der Runner steuert den Ablauf (Tasten halten, Tab im Zufallsintervall)
in einem eigenen Thread und lässt beim Stoppen garantiert alles wieder los.
"""

import platform
import random
import threading
import time

BUTTONS = ("left", "right")


class Backend:
    name = "Basis"

    def press(self, button):
        raise NotImplementedError

    def release(self, button):
        raise NotImplementedError

    def tap_tab(self):
        raise NotImplementedError

    def close(self):
        pass


class SimulationBackend(Backend):
    name = "Simulation (nur Log)"

    def __init__(self, log):
        self._log = log

    def press(self, button):
        self._log(f"[SIM] {button} gedrückt")

    def release(self, button):
        self._log(f"[SIM] {button} losgelassen")

    def tap_tab(self):
        self._log("[SIM] Tab")


class PynputBackend(Backend):
    name = "pynput (Windows / macOS / Linux-X11)"

    def __init__(self, log):
        from pynput import keyboard, mouse

        self._mouse = mouse.Controller()
        self._keyboard = keyboard.Controller()
        self._buttons = {"left": mouse.Button.left, "right": mouse.Button.right}
        self._tab = keyboard.Key.tab

    def press(self, button):
        self._mouse.press(self._buttons[button])

    def release(self, button):
        self._mouse.release(self._buttons[button])

    def tap_tab(self):
        self._keyboard.press(self._tab)
        time.sleep(random.uniform(0.04, 0.09))
        self._keyboard.release(self._tab)


class UinputBackend(Backend):
    name = "uinput – virtuelles Gerät (Linux)"

    def __init__(self, log):
        from evdev import UInput
        from evdev import ecodes as e

        self._e = e
        # Maus braucht REL_X/REL_Y, sonst erkennt libinput sie nicht als Maus.
        self._mouse = UInput(
            {e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE],
             e.EV_REL: [e.REL_X, e.REL_Y, e.REL_WHEEL]},
            name="DIY-Eingabegeraet Maus",
        )
        self._keyboard = UInput({e.EV_KEY: [e.KEY_TAB]}, name="DIY-Eingabegeraet Tastatur")
        self._codes = {"left": e.BTN_LEFT, "right": e.BTN_RIGHT}
        # Dem System kurz Zeit geben, die neuen Geräte einzubinden.
        time.sleep(0.5)
        log(f"Virtuelle Geräte angelegt: {self._mouse.device.path}, {self._keyboard.device.path}")

    def _key(self, device, code, value):
        device.write(self._e.EV_KEY, code, value)
        device.syn()

    def press(self, button):
        self._key(self._mouse, self._codes[button], 1)

    def release(self, button):
        self._key(self._mouse, self._codes[button], 0)

    def tap_tab(self):
        self._key(self._keyboard, self._e.KEY_TAB, 1)
        time.sleep(random.uniform(0.04, 0.09))
        self._key(self._keyboard, self._e.KEY_TAB, 0)

    def close(self):
        self._mouse.close()
        self._keyboard.close()


def _importable(module):
    try:
        __import__(module)
        return True
    except Exception:
        return False


def available_backends():
    """Liefert {Anzeigename: Klasse} aller auf diesem System nutzbaren Backends."""
    result = {}
    if platform.system() == "Linux" and _importable("evdev"):
        result[UinputBackend.name] = UinputBackend
    if _importable("pynput"):
        result[PynputBackend.name] = PynputBackend
    result[SimulationBackend.name] = SimulationBackend
    return result


class Runner:
    """Führt die Testsequenz in einem Hintergrund-Thread aus.

    on_event(art, wert) wird aus dem Worker-Thread aufgerufen mit:
    ("countdown", sek), ("running", None), ("next_tab", sek),
    ("tab", anzahl), ("log", text), ("error", text), ("stopped", None)
    """

    def __init__(self, backend, on_event, *, hold_left=True, hold_right=True,
                 tab_enabled=True, tab_min=4.0, tab_max=8.0, start_delay=3):
        if tab_min <= 0 or tab_max < tab_min:
            raise ValueError("Intervall ungültig: es muss 0 < Min ≤ Max gelten.")
        self.backend = backend
        self.on_event = on_event
        self.hold = [b for b, on in (("left", hold_left), ("right", hold_right)) if on]
        self.tab_enabled = tab_enabled
        self.tab_min = tab_min
        self.tab_max = tab_max
        self.start_delay = int(start_delay)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        self._thread.join(timeout)

    @property
    def alive(self):
        return self._thread.is_alive()

    def _run(self):
        held = []
        tabs = 0
        try:
            for remaining in range(self.start_delay, 0, -1):
                self.on_event("countdown", remaining)
                if self._stop.wait(1):
                    return

            for button in self.hold:
                self.backend.press(button)
                held.append(button)
                self.on_event("log", f"{'Linke' if button == 'left' else 'Rechte'} Maustaste gedrückt (gehalten)")
            self.on_event("running", None)

            if not self.tab_enabled:
                self._stop.wait()
                return

            while True:
                wait = random.uniform(self.tab_min, self.tab_max)
                self.on_event("next_tab", wait)
                if self._stop.wait(wait):
                    return
                self.backend.tap_tab()
                tabs += 1
                self.on_event("tab", tabs)
                self.on_event("log", f"Tab #{tabs} (nach {wait:.2f} s)")
        except Exception as exc:  # Fehler im Backend dürfen Tasten nicht "kleben" lassen
            self.on_event("error", f"{type(exc).__name__}: {exc}")
        finally:
            for button in reversed(held):
                try:
                    self.backend.release(button)
                    self.on_event("log", f"{'Linke' if button == 'left' else 'Rechte'} Maustaste losgelassen")
                except Exception as exc:
                    self.on_event("error", f"Loslassen fehlgeschlagen ({button}): {exc}")
            self.on_event("stopped", None)
