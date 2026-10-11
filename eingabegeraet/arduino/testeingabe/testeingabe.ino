// DIY-Eingabegerät – Hardware-Version des Tests
// Board: Arduino Leonardo / Pro Micro (ATmega32U4, natives USB-HID)
//
// Schalter zwischen Pin 2 und GND:
//   geschlossen -> linke + rechte Maustaste halten, Tab alle 4–8 s (zufällig)
//   offen       -> alles loslassen (Not-Aus, auch nach dem Flashen sicher)

#include <Mouse.h>
#include <Keyboard.h>

const int PIN_AKTIV = 2;
const unsigned long TAB_MIN_MS = 4000;
const unsigned long TAB_MAX_MS = 8000;

bool aktiv = false;
unsigned long naechsterTab = 0;

void planeNaechstenTab() {
  naechsterTab = millis() + random(TAB_MIN_MS, TAB_MAX_MS + 1);
}

void setup() {
  pinMode(PIN_AKTIV, INPUT_PULLUP);
  randomSeed(analogRead(A0));  // offener Analogpin als Zufallsquelle
  Mouse.begin();
  Keyboard.begin();
}

void loop() {
  bool sollAktiv = digitalRead(PIN_AKTIV) == LOW;

  if (sollAktiv && !aktiv) {
    Mouse.press(MOUSE_LEFT);
    Mouse.press(MOUSE_RIGHT);
    planeNaechstenTab();
    aktiv = true;
  } else if (!sollAktiv && aktiv) {
    Mouse.release(MOUSE_LEFT);
    Mouse.release(MOUSE_RIGHT);
    Keyboard.releaseAll();
    aktiv = false;
  }

  if (aktiv && (long)(millis() - naechsterTab) >= 0) {
    Keyboard.press(KEY_TAB);
    delay(random(40, 90));
    Keyboard.release(KEY_TAB);
    planeNaechstenTab();
  }

  delay(10);  // entprellt den Schalter grob
}
