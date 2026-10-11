# DiyHid – eigener Windows-11-Treiber (virtuelle Maus + Tastatur)

Ein echter Kernel-Treiber (KMDF), der über Microsofts **Virtual HID Framework (VHF)**
ein HID-Gerät anmeldet. Windows lädt dafür seine normalen Maus- und Tastaturtreiber
(`mouhid.sys`, `kbdhid.sys`). Im Geräte-Manager erscheinen eine **HID-konforme Maus** und
eine **HID-Tastatur**, genau wie bei einem USB-Gerät.

```
GUI (Python) ──DeviceIoControl──► DiyHid.sys ──► vhf.sys ──► HID-Stack ──► mouhid/kbdhid ──► Windows
```

> ⚠️ **Vorsicht:** Fehler in Kernel-Treibern führen zum Bluescreen. Teste den Treiber am
> besten zuerst in einer virtuellen Maschine (Hyper-V, VirtualBox, VMware).

## 1. Treiber herunterladen

Der Treiber wird automatisch per GitHub Actions gebaut:
Repository → **Actions** → „Windows-Treiber bauen (DiyHid)“ → neuester Lauf → unten
unter *Artifacts* **DiyHid-x64** herunterladen und entpacken.

Inhalt: `DiyHid.sys`, `DiyHid.inf`, `DiyHid.cat`, `DiyHid-Testzertifikat.cer`.

## 2. Testmodus einschalten (einmalig)

Der Treiber ist nur **testsigniert**. Für eine echte Signatur bräuchtest du ein
EV-Zertifikat und eine Freigabe durch Microsoft. Deshalb:

1. **Secure Boot im BIOS/UEFI ausschalten.** Sonst meldet Windows: „Der Wert ist durch die
   Secure Boot-Richtlinie geschützt“.
2. Eingabeaufforderung **als Administrator**:
   ```
   bcdedit /set testsigning on
   ```
3. Neu starten. Unten rechts steht nun „Testmodus“.

## 3. Installieren

1. `Win + R` → `hdwwiz` → **Weiter**
2. **Hardware manuell aus einer Liste wählen und installieren** → Weiter
3. **Alle Geräte anzeigen** → Weiter → **Datenträger…** → `DiyHid.inf` auswählen
4. „DIY-Eingabegeraet (virtuelle Maus + Tastatur)“ → Weiter
5. Bei der roten Warnung **„Diese Treibersoftware trotzdem installieren“** wählen

Prüfen im Geräte-Manager:
- *Systemgeräte* → „DIY-Eingabegeraet (virtuelle Maus + Tastatur)“
- *Mäuse und andere Zeigegeräte* → zusätzliche „HID-konforme Maus“
- *Tastaturen* → zusätzliche „HID-Tastatur“

## 4. Benutzen

Die GUI **als Administrator** starten (der Treiber erlaubt nur Admins den Zugriff):

```
cd eingabegeraet
python gui.py
```

Als Backend **„DIY-HID-Kerneltreiber (Windows)“** wählen und auf Start klicken.

Wird die GUI geschlossen oder stürzt sie ab, lässt der Treiber selbst alle Tasten los.

## 5. Deinstallieren

1. Geräte-Manager → *Systemgeräte* → „DIY-Eingabegeraet …“ → Rechtsklick → **Gerät deinstallieren**
   → Haken bei **„Treiber für dieses Gerät entfernen“**
2. Testmodus wieder aus: `bcdedit /set testsigning off`, danach Secure Boot wieder einschalten und neu starten

## Selbst bauen

Visual Studio 2022/2026 mit der Workload „Desktopentwicklung mit C++“ und dem
**Windows Driver Kit (WDK)**. Danach `driver/DiyHid.vcxproj` öffnen und *Release | x64* bauen.
Alternativ ohne installierten WDK, wie in der CI:

```
nuget restore packages.config -PackagesDirectory packages
msbuild driver\DiyHid.vcxproj /p:Configuration=Release /p:Platform=x64
```

## Schnittstelle (für eigene Programme)

`CreateFile("\\.\DiyHid", GENERIC_WRITE, …)`, danach `DeviceIoControl` mit IOCTL `0x0022A000`
und einem dieser Reports als Eingabepuffer:

| Report | Bytes |
|---|---|
| Maus | `01, Tasten (Bit0 links, Bit1 rechts, Bit2 Mitte), X, Y, Rad` (X/Y/Rad: −127…127) |
| Tastatur | `02, Modifier, 00, Taste1 … Taste6` (HID-Usage-Codes, z. B. Tab = `0x2B`) |

Ungültige Reports lehnt der Treiber mit `STATUS_INVALID_PARAMETER` ab.
