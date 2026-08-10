/*
 * BCI_Display_v2.ino — ST7789 240x240 / Arduino Mega 2560
 *
 * Control serial a 115200 baud:
 *   0 → secuencia completa del pulgar (pretrial + feedback)
 *   1 → secuencia completa del meñique (pretrial + feedback)
 *   2 → ambos motores encendidos continuamente
 *   3 → reposo inmediato; todas las salidas apagadas
 *
 * Los pines y el diseño visual se conservan sin cambios respecto a la
 * versión aprobada. Cualquier comando válido interrumpe el estado actual.
 *
 * Replica ExperimentDriver_Offline.py / visualization.py:
 *
 *   IZQUIERDA : círculo azul  (REST)  ← draw_ball_fill()
 *   CENTRO    : cruz de fijación
 *   DERECHA   : cuadrado rojo (MI)    ← draw_arrow_fill()  [es un rect en Python]
 *
 *   Intertrial  → ambas figuras como outline blanco
 *   Pretrial MI → solo cuadrado outline + barra roja L→R + indicador rojo arriba
 *   Feedback MI → cuadrado llenándose rojo L→R
 *   Pretrial REST → solo círculo outline + barra azul R→L + indicador azul arriba
 *   Feedback REST → círculo llenándose azul abajo→arriba
 */

#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>

#define TFT_CS  10
#define TFT_DC   9
#define TFT_RST  8

Adafruit_ST7789 tft(TFT_CS, TFT_DC, TFT_RST);

// ── Colores ──────────────────────────────────────────────────
#define BLACK  0x0000
#define WHITE  0xFFFF

inline uint16_t rgb(uint8_t r, uint8_t g, uint8_t b) {
  return ((uint16_t)(r & 0xF8) << 8) | ((uint16_t)(g & 0xFC) << 3) | (b >> 3);
}

const uint16_t MI_RED     = rgb(255, 50,  50);
const uint16_t REST_BLUE  = rgb(0,   120, 255);
const uint16_t DARK_GRAY  = rgb(60,  60,  60);
const uint16_t LOGO_GREEN = rgb(60,  168,  50);
const uint16_t LOGO_RED   = rgb(215,  38,  38);
const uint16_t BURNT_ORG  = rgb(191,  87,   0);

// ── Layout ───────────────────────────────────────────────────
#define W   240
#define H   240
#define CX  120   // centro pantalla — cruz aquí
#define CY  115   // centro vertical de las figuras y la cruz

// Círculo REST — mitad izquierda
#define BALL_X   58
#define BALL_R   44

// Cuadrado MI — mitad derecha (mismo radio que el círculo)
#define BOX_CX  182
#define BOX_R    44   // half-size

// Indicador pequeño arriba-centro (NEXT_INDICATOR en Python, pos 0.50, 0.28)
#define IND_X   120
#define IND_Y    50
#define IND_R    11

// Barra de countdown
#define BAR_X    55
#define BAR_Y   210
#define BAR_W   130
#define BAR_H     7

// ── Motores vibratorios ──────────────────────────────────────
// Conectar vía transistor NPN (BC547/2N2222): pin → 470Ω → base
#define MOTOR_PINKY_PIN  5   // meñique — trial rojo (MI)
#define MOTOR_THUMB_PIN  6   // pulgar  — trial azul (REST)

// Patrón sensorial (pretrial): pulso 150ms ON cada 500ms
// Patrón motor    (feedback) : continuo ON
#define SENS_PERIOD   500
#define SENS_ON_TIME  150

#define GLOVE_PIN  12   // activa/desactiva el guante en cualquier trial

#define FLEX_PINKY_PIN  18
#define FLEX_THUMB_PIN  14

// ── Fases ────────────────────────────────────────────────────
#define INTERTRIAL    0
#define PRETRIAL_MI   1
#define FEEDBACK_MI   2
#define PRETRIAL_REST 3
#define FEEDBACK_REST 4
#define IDLE          5
#define MOTORS_CONTINUOUS 6

// ── Estado global ────────────────────────────────────────────
int           phase        = INTERTRIAL;
int           nextAfterITI = PRETRIAL_MI;
unsigned long t0           = 0;
int           boxFillX     = 0;   // relleno cuadrado MI
int           ballFillY    = 0;   // relleno círculo REST
int           barFill      = 0;

#define DUR_ITI      2000UL
#define DUR_PRETRIAL 2500UL
#define DUR_FEEDBACK 4000UL

// ════════════════════════════════════════════════════════════
//  DIBUJO — EXPERIMENTO
// ════════════════════════════════════════════════════════════

void drawCross() {
  tft.fillRect(CX - 16, CY -  2, 32,  4, WHITE);
  tft.fillRect(CX -  2, CY - 16,  4, 32, WHITE);
}

// Outline del círculo (REST) — izquierda
void drawBallOutline(uint16_t col) {
  tft.drawCircle(BALL_X, CY, BALL_R,     col);
  tft.drawCircle(BALL_X, CY, BALL_R - 1, col);
}

// Outline del cuadrado (MI) — derecha
void drawBoxOutline(uint16_t col) {
  tft.drawRect(BOX_CX - BOX_R,     CY - BOX_R,     BOX_R*2,     BOX_R*2,     col);
  tft.drawRect(BOX_CX - BOX_R + 1, CY - BOX_R + 1, BOX_R*2 - 2, BOX_R*2 - 2, col);
}

// Relleno incremental cuadrado: izquierda → derecha
void advanceBoxFill(float progress) {
  int x0 = BOX_CX - BOX_R + 2;
  int x1 = BOX_CX + BOX_R - 2;
  int tx = x0 + (int)((x1 - x0) * progress);
  if (tx > x1) tx = x1;
  if (tx <= boxFillX) return;
  tft.fillRect(boxFillX, CY - BOX_R + 2, tx - boxFillX, BOX_R*2 - 4, MI_RED);
  boxFillX = tx;
}

// Relleno incremental círculo: abajo → arriba
void advanceBallFill(float progress) {
  int ty = CY + BALL_R - (int)(2 * BALL_R * progress);
  if (ty < CY - BALL_R) ty = CY - BALL_R;
  if (ty >= ballFillY)  return;
  for (int y = ballFillY - 1; y >= ty; y--) {
    int dy = y - CY;
    int dx = (int)sqrt((float)(BALL_R*BALL_R - dy*dy));
    if (dx > 1)
      tft.drawFastHLine(BALL_X - dx + 1, y, (dx-1)*2, REST_BLUE);
  }
  ballFillY = ty;
}

// Barra de countdown incremental
void advanceBar(float progress, bool isMI) {
  int target = (int)(BAR_W * progress);
  if (target > BAR_W) target = BAR_W;
  if (target <= barFill) return;
  int delta = target - barFill;
  uint16_t col = isMI ? MI_RED : REST_BLUE;
  if (isMI)
    tft.fillRect(BAR_X + barFill,        BAR_Y, delta, BAR_H, col);
  else
    tft.fillRect(BAR_X + BAR_W - target, BAR_Y, delta, BAR_H, col);
  barFill = target;
}

// Indicador pequeño arriba-centro
void drawIndicator(bool isMI) {
  if (isMI)
    tft.fillRect(IND_X - IND_R, IND_Y - IND_R, IND_R*2, IND_R*2, MI_RED);
  else
    tft.fillCircle(IND_X, IND_Y, IND_R, REST_BLUE);
}

// Aplica todas las salidas de acuerdo con la fase actual.
// Se llama también al entrar en una fase para que el apagado sea inmediato.
void updateOutputs() {
  unsigned long elapsed = millis() - t0;
  bool sensOn = (elapsed % SENS_PERIOD) < SENS_ON_TIME;

  switch (phase) {
    case PRETRIAL_MI:                              // sensorial meñique
      digitalWrite(MOTOR_PINKY_PIN, sensOn);
      digitalWrite(MOTOR_THUMB_PIN, LOW);
      break;
    case FEEDBACK_MI:                              // motor meñique continuo
      digitalWrite(MOTOR_PINKY_PIN, HIGH);
      digitalWrite(MOTOR_THUMB_PIN, LOW);
      break;
    case PRETRIAL_REST:                            // sensorial pulgar
      digitalWrite(MOTOR_PINKY_PIN, LOW);
      digitalWrite(MOTOR_THUMB_PIN, sensOn);
      break;
    case FEEDBACK_REST:                            // motor pulgar continuo
      digitalWrite(MOTOR_PINKY_PIN, LOW);
      digitalWrite(MOTOR_THUMB_PIN, HIGH);
      break;
    case MOTORS_CONTINUOUS:                        // comando serial 2
      digitalWrite(MOTOR_PINKY_PIN, HIGH);
      digitalWrite(MOTOR_THUMB_PIN, HIGH);
      break;
    default:                                       // intertrial: apagados
      digitalWrite(MOTOR_PINKY_PIN, LOW);
      digitalWrite(MOTOR_THUMB_PIN, LOW);
      break;
  }

  // Guante: solo ON durante feedback (no en pretrial ni intertrial)
  bool gloveOn = (phase == FEEDBACK_MI || phase == FEEDBACK_REST);
  digitalWrite(GLOVE_PIN, gloveOn ? HIGH : LOW);

  // Pines activos únicamente mientras aparece la indicación "Flex".
  digitalWrite(FLEX_PINKY_PIN, phase == FEEDBACK_MI ? HIGH : LOW);
  digitalWrite(FLEX_THUMB_PIN, phase == FEEDBACK_REST ? HIGH : LOW);
}

void centeredText(const char* txt, int y, uint16_t col, uint8_t sz = 2) {
  tft.setTextColor(col, BLACK);
  tft.setTextSize(sz);
  int x = (W - (int)strlen(txt) * 6 * sz) / 2;
  if (x < 0) x = 0;
  tft.setCursor(x, y);
  tft.print(txt);
}

void enterPhase(int p) {
  phase     = p;
  t0        = millis();
  boxFillX  = BOX_CX - BOX_R + 2;
  ballFillY = CY + BALL_R;
  barFill   = 0;
  tft.fillScreen(BLACK);

  switch (p) {

    case IDLE:
    case MOTORS_CONTINUOUS:
      // Los nuevos modos no añaden elementos al diseño aprobado.
      break;

    case INTERTRIAL:
      drawBallOutline(WHITE);   // círculo outline izquierda
      drawBoxOutline(WHITE);    // cuadrado outline derecha
      drawCross();
      break;

    case PRETRIAL_MI:
      drawBoxOutline(WHITE);
      tft.fillRect(BAR_X, BAR_Y, BAR_W, BAR_H, DARK_GRAY);
      drawIndicator(true);
      centeredText("Prepare: PINKY", 180, WHITE);
      drawCross();
      break;

    case FEEDBACK_MI:
      drawBoxOutline(WHITE);
      centeredText("Flex PINKY", 180, WHITE);
      drawCross();
      break;

    case PRETRIAL_REST:
      drawBallOutline(WHITE);
      tft.fillRect(BAR_X, BAR_Y, BAR_W, BAR_H, DARK_GRAY);
      drawIndicator(false);
      centeredText("Prepare: THUMB", 180, WHITE);
      drawCross();
      break;

    case FEEDBACK_REST:
      drawBallOutline(WHITE);
      centeredText("Flex THUMB", 180, WHITE);
      drawCross();
      break;
  }

  updateOutputs();
}

void processSerialCommand() {
  while (Serial.available() > 0) {
    char command = (char)Serial.read();

    switch (command) {
      case 0:
      case '0':
        Serial.println(F("CMD 0: THUMB"));
        enterPhase(PRETRIAL_REST);
        break;

      case 1:
      case '1':
        Serial.println(F("CMD 1: PINKY"));
        enterPhase(PRETRIAL_MI);
        break;

      case 2:
      case '2':
        Serial.println(F("CMD 2: MOTORS CONTINUOUS"));
        enterPhase(MOTORS_CONTINUOUS);
        break;

      case 3:
      case '3':
        Serial.println(F("CMD 3: IDLE"));
        enterPhase(IDLE);
        break;

      // Ignorar terminadores de línea y cualquier otro carácter.
      default:
        break;
    }
  }
}

// ════════════════════════════════════════════════════════════
//  SPLASH — UT LONGHORN
// ════════════════════════════════════════════════════════════

void drawThickBezier(int x0, int y0, int ccx, int ccy,
                     int x1, int y1, int r0, int r1, uint16_t col) {
  for (int i = 0; i <= 28; i++) {
    float t = i / 28.0f, mt = 1.0f - t;
    int x = (int)(mt*mt*x0 + 2.0f*mt*t*ccx + t*t*x1);
    int y = (int)(mt*mt*y0 + 2.0f*mt*t*ccy + t*t*y1);
    tft.fillCircle(x, y, r0 + (int)((r1 - r0) * t), col);
  }
}

void drawUTLonghorn() {
  tft.fillScreen(BLACK);
  drawThickBezier( 80, 82,  28, 26,   5, 105, 15, 7, BURNT_ORG);
  drawThickBezier(160, 82, 212, 26, 235, 105, 15, 7, BURNT_ORG);
  const int hcx = 120, hcy = 142;
  const int px[] = { 80,160,175,180,170,167,150,120, 90, 73, 70, 60, 65};
  const int py[] = { 82, 82,102,120,130,150,172,198,172,150,130,120,102};
  for (int i = 0; i < 13; i++) {
    int j = (i + 1) % 13;
    tft.fillTriangle(hcx, hcy, px[i], py[i], px[j], py[j], BURNT_ORG);
  }
  tft.fillCircle(120,  82, 40, BURNT_ORG);
  tft.fillCircle( 65, 115, 10, BURNT_ORG);
  tft.fillCircle(175, 115, 10, BURNT_ORG);
  tft.fillCircle(120, 195, 14, BURNT_ORG);
  delay(2500);
  tft.fillScreen(BLACK);
  delay(300);
}

// ════════════════════════════════════════════════════════════
//  SPLASH — LOGO CNBI
// ════════════════════════════════════════════════════════════

void drawGear(int cx, int cy, int r, uint16_t col) {
  for (int i = 0; i < 6; i++) {
    float a = i * 3.14159f / 3.0f;
    int tx = cx + (int)((r + 4) * cos(a));
    int ty = cy + (int)((r + 4) * sin(a));
    tft.fillRect(tx - 3, ty - 3, 6, 6, col);
  }
  tft.fillCircle(cx, cy, r, col);
  tft.fillCircle(cx, cy, r / 3, BLACK);
}

void drawCNBILogo() {
  tft.fillScreen(BLACK);
  const int bx = 72, by = 105;
  tft.fillCircle(bx,      by,      44, LOGO_GREEN);
  tft.fillCircle(bx -  8, by -  4, 40, LOGO_GREEN);
  tft.fillCircle(bx +  8, by -  4, 40, LOGO_GREEN);
  tft.fillCircle(bx - 18, by - 32, 20, LOGO_GREEN);
  tft.fillCircle(bx + 18, by - 32, 20, LOGO_GREEN);
  tft.fillRect(bx - 3, by - 50, 6, 55, BLACK);
  int lx = bx - 14;
  tft.drawLine(lx-10,by-28,lx+ 6,by-20,BLACK);
  tft.drawLine(lx+ 6,by-20,lx- 4,by- 6,BLACK);
  tft.drawLine(lx- 4,by- 6,lx+ 8,by+ 6,BLACK);
  tft.drawLine(lx-14,by+12,lx+ 4,by+22,BLACK);
  tft.drawLine(lx+ 4,by+22,lx- 8,by+34,BLACK);
  int rx = bx + 14;
  tft.drawLine(rx+10,by-28,rx- 6,by-20,BLACK);
  tft.drawLine(rx- 6,by-20,rx+ 4,by- 6,BLACK);
  tft.drawLine(rx+ 4,by- 6,rx- 8,by+ 6,BLACK);
  tft.drawLine(rx+14,by+12,rx- 4,by+22,BLACK);
  tft.drawLine(rx- 4,by+22,rx+ 8,by+34,BLACK);
  drawGear(bx - 18, by + 62, 13, LOGO_RED);
  drawGear(bx + 10, by + 66, 10, LOGO_RED);
  tft.setTextColor(LOGO_GREEN, BLACK);
  tft.setTextSize(3);
  tft.setCursor(145, 96);
  tft.print("cnbi");
  delay(2500);
  tft.fillScreen(BLACK);
  delay(200);
}

// ════════════════════════════════════════════════════════════
//  SETUP / LOOP
// ════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  pinMode(MOTOR_PINKY_PIN, OUTPUT);
  pinMode(MOTOR_THUMB_PIN, OUTPUT);
  pinMode(GLOVE_PIN, OUTPUT);
  
  pinMode(FLEX_PINKY_PIN, OUTPUT);
  pinMode(FLEX_THUMB_PIN, OUTPUT);

  digitalWrite(MOTOR_PINKY_PIN, LOW);
  digitalWrite(MOTOR_THUMB_PIN, LOW);
  digitalWrite(GLOVE_PIN, LOW);
  digitalWrite(FLEX_PINKY_PIN, LOW);
  digitalWrite(FLEX_THUMB_PIN, LOW);
  
  tft.init(240, 240);
  tft.setRotation(3);
  drawUTLonghorn();
  drawCNBILogo();
  enterPhase(IDLE);

  Serial.println(F("BCI Display v2 ready"));
  Serial.println(F("0=THUMB, 1=PINKY, 2=MOTORS ON, 3=ALL OFF"));


}

void loop() {
  processSerialCommand();
  updateOutputs();

  unsigned long elapsed = millis() - t0;
  float progress;

  switch (phase) {

    case INTERTRIAL:
      if (elapsed >= DUR_ITI)
        enterPhase(nextAfterITI);
      break;

    case PRETRIAL_MI:
      progress = min((float)elapsed / DUR_PRETRIAL, 1.0f);
      advanceBar(progress, true);
      if (elapsed >= DUR_PRETRIAL) enterPhase(FEEDBACK_MI);
      break;

    case FEEDBACK_MI:
      progress = min((float)elapsed / DUR_FEEDBACK, 1.0f);
      advanceBoxFill(progress);
      if (elapsed >= DUR_FEEDBACK) {
        enterPhase(IDLE);
      }
      break;

    case PRETRIAL_REST:
      progress = min((float)elapsed / DUR_PRETRIAL, 1.0f);
      advanceBar(progress, false);
      if (elapsed >= DUR_PRETRIAL) enterPhase(FEEDBACK_REST);
      break;

    case FEEDBACK_REST:
      progress = min((float)elapsed / DUR_FEEDBACK, 1.0f);
      advanceBallFill(progress);
      if (elapsed >= DUR_FEEDBACK) {
        enterPhase(IDLE);
      }
      break;

    case IDLE:
    case MOTORS_CONTINUOUS:
      // Permanecer aquí hasta recibir otro comando serial.
      break;
  }
}
