# DIY-Eingabegerät – Testtreiber mit GUI

Software-Treiber zum Testen selbstgebauter Eingabegeräte. Die Testsequenz:

- linke Maustaste gedrückt halten
- rechte Maustaste gedrückt halten
- Tabulator zufällig alle 4–8 s drücken (Intervall einstellbar)

## Installation

```bash
cd eingabegeraet
pip install -r requirements.txt
python gui.py
```

Unter Linux wird zusätzlich `python3-tk` benötigt (`sudo apt install python3-tk`).

## Backends („Treiber“)

| Backend | System | Funktionsweise |
|---|---|---|
| **uinput – virtuelles Gerät** | Linux | Legt über `/dev/uinput` zwei echte virtuelle Geräte an („DIY-Eingabegeraet Maus“ / „… Tastatur“). Der Kernel behandelt sie wie angesteckte USB-Geräte, funktioniert auch unter Wayland. |
| **pynput** | Windows, macOS, Linux/X11 | Sendet Eingaben über die System-API (Windows: `SendInput`). |
| **Simulation** | überall | Schreibt nur ins Log – zum gefahrlosen Ausprobieren. |

### Linux: Rechte für uinput

```bash
sudo modprobe uinput
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-uinput.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG input $USER   # danach neu anmelden
```

Die angelegten Geräte prüfen: `cat /proc/bus/input/devices` oder `sudo evtest`.

### macOS

Python/Terminal braucht unter *Systemeinstellungen → Datenschutz & Sicherheit → Bedienungshilfen*
die Berechtigung, Eingaben zu steuern.

## Bedienung & Sicherheit

1. Aktionen und Intervall wählen, **Start** klicken.
2. Während der **Startverzögerung** (Standard 3 s) ins Zielfenster wechseln.
3. Stoppen mit **F12** (global) oder **Esc** im Fenster.

Beim Stoppen, bei Fehlern und beim Schließen des Fensters werden alle gehaltenen
Tasten immer wieder losgelassen. Bleibt eine Maustaste trotzdem „hängen“: einmal
physisch links und rechts klicken.

Hinweis: Solange die linke Taste gehalten wird, können Klicks in der GUI anders
reagieren als gewohnt – darum lieber F12 verwenden.

## Nächster Schritt: echte Hardware

`arduino/testeingabe/testeingabe.ino` macht dasselbe als echtes USB-HID-Gerät
auf einem **Arduino Leonardo / Pro Micro** (ATmega32U4). Ein Schalter zwischen
**Pin 2 und GND** schaltet die Sequenz an/aus. Ist der Schalter offen, sendet das
Board nichts. So lässt es sich auch nach dem Flashen jederzeit sicher abstecken
oder neu programmieren.

## Aufbau des Codes

- `treiber.py` – Backends und `Runner` (Ablauf in eigenem Thread, Zufallsintervall, garantiertes Loslassen)
- `gui.py` – Tkinter-Oberfläche, Status, Countdown bis zum nächsten Tab, Log, Not-Aus
