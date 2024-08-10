#include <math.h>
#include <Wire.h>
#include "SparkFun_Displacement_Sensor_Arduino_Library.h" // Click here to get the library: http://librarymanager/All#SparkFun_Displacement_Sensor

ADS myFlexSensor;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  while (!Serial)
    ;
  Serial.println("SparkFun Displacement Sensor");
//  pinMode(PWMPX,OUTPUT);
//  pinMode(PWMNX,OUTPUT);
//  pinMode(PWMPY,OUTPUT);
//  pinMode(PWMNY,OUTPUT);

  Wire.begin();

  if (myFlexSensor.begin() == false)
  {
    Serial.println(F("No sensor detected. Check wiring. Freezing..."));
    while (1)
      ;
  }
}

void loop() {
  // put your main code here, to run repeatedly:
  if (myFlexSensor.available() == true) {
          float BX = myFlexSensor.getX();
          float BY = myFlexSensor.getY();
          Serial.print("Data,");
          Serial.print(BX);
          Serial.print(",");
          Serial.print(BY);
          Serial.println();
  }
  delay(50);
}
