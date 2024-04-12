#include <FastLED.h>

#define LED_PIN     7
#define NUM_LEDS    24

CRGB leds[NUM_LEDS];

void setup() {
  FastLED.addLeds<WS2812, LED_PIN, GRB>(leds, NUM_LEDS);
}

void loop() {
  for (int i = 0; i < NUM_LEDS; i++) {
    if (i % 3 == 0) {
      leds[i] = CRGB(0, 255, 0);
      FastLED.show();
    }
  }
}
  
