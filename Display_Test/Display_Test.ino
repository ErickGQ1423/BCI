/*
 * Display_Test.ino
 * Prueba minima: rellena la pantalla con colores alternos.
 */

#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>

#define TFT_CS   10
#define TFT_DC    9
#define TFT_RST   8

Adafruit_ST7789 tft(TFT_CS, TFT_DC, TFT_RST);  // hardware SPI

void setup() {
  Serial.begin(115200);

  Serial.println("Iniciando display...");
  tft.init(240, 240);
  tft.setRotation(0);
  Serial.println("Display init OK");
}

void loop() {
  tft.fillScreen(0xF800);
  Serial.println("ROJO");
  delay(1000);

  tft.fillScreen(0x07E0);
  Serial.println("VERDE");
  delay(1000);

  tft.fillScreen(0x001F);
  Serial.println("AZUL");
  delay(1000);

  tft.fillScreen(0xFFFF);
  tft.setTextColor(0x0000);
  tft.setTextSize(3);
  tft.setCursor(40, 100);
  tft.print("HOLA!");
  Serial.println("BLANCO + TEXTO");
  delay(1000);
}
