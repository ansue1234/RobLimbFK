#include <Wire.h>
#include "SparkFun_Displacement_Sensor_Arduino_Library.h" // Click here to get the library: http://librarymanager/All#SparkFun_Displacement_Sensor


ADS myFlexSensor;
//PWM of X,Y,-X,-Y
int PWMPX = 5;
int PWMNX = 4;
int PWMPY = 3;
int PWMNY = 2;

int PWMSignal = 0;
int SampleRate = 50;
int forloop = 80;
int forlooprefresh = 1000;
int hold = 20000;

void setup()
{
  Serial.begin(9600);
  while (!Serial)
    ;
  Serial.println("SparkFun Displacement Sensor Example");
  pinMode(PWMPX,OUTPUT);
  pinMode(PWMNX,OUTPUT);
  pinMode(PWMPY,OUTPUT);
  pinMode(PWMNY,OUTPUT);

  Wire.begin();

  if (myFlexSensor.begin() == false)
  {
    Serial.println(F("No sensor detected. Check wiring. Freezing..."));
    while (1)
      ;
  }
}

void SMApx()
{
  Serial.println("Characterizing Positive X");
  Serial.println("What PWM would you want to test?");
  //Serial.println(Serial.available());
  while (Serial.available()==1){
  }
    PWMSignal = Serial.parseInt();
    {
      for (int i=0; i<=forloop; i++) {
      if (myFlexSensor.available() == true) {
      analogWrite(PWMPX,PWMSignal);
      float time = millis();
      Serial.print(time);
      Serial.print(",");
      Serial.print(myFlexSensor.getX());
      Serial.print(",");
      Serial.print(myFlexSensor.getY());
      Serial.print(",");
      Serial.print(PWMSignal);
      Serial.println();
      delay(SampleRate);
      }
      }
      analogWrite(PWMPX,0);
      delay(hold);
    }
}

void SMAnx()
{
  Serial.println("Characterizing Negative X");
  Serial.println("What PWM would you want to test?");
  while (Serial.available()==1){
  }
    PWMSignal = Serial.parseInt();
    {
      for (int i=0; i<=forloop; i++){
      if (myFlexSensor.available() == true) {
      analogWrite(PWMNX,PWMSignal);
      float time = millis();
      Serial.print(time);
      Serial.print(",");
      Serial.print(myFlexSensor.getX());
      Serial.print(",");
      Serial.print(myFlexSensor.getY());
      Serial.print(",");
      Serial.print(PWMSignal);
      Serial.println();
      delay(SampleRate);
      }
      }
      analogWrite(PWMNX,0);
      delay(hold);
    }
}

void SMApy()
{
  Serial.println("Characterizing Positive Y");
  Serial.println("What PWM would you want to test?");
  while (Serial.available()==1){
  }
    PWMSignal = Serial.parseInt();
    {
      for (int i=0; i<=forloop; i++){
      if (myFlexSensor.available() == true) {
      analogWrite(PWMPY,PWMSignal);
      float time = millis();
      Serial.print(time);
      Serial.print(",");
      Serial.print(myFlexSensor.getX());
      Serial.print(",");
      Serial.print(myFlexSensor.getY());
      Serial.print(",");
      Serial.print(PWMSignal);
      Serial.println();
      delay(SampleRate);
      }
      }
      analogWrite(PWMPY,0);
      delay(hold);
    }
}

void SMAny()
{
  Serial.println("Characterizing Negative Y");
  Serial.println("What PWM would you want to test?");
  while (Serial.available()==1){
  }
    PWMSignal = Serial.parseInt();
    {
      for (int i=0; i<=forloop; i++) {
      if (myFlexSensor.available() == true) {
      analogWrite(PWMNY,PWMSignal);
      float time = millis();
      Serial.print(time);
      Serial.print(",");
      Serial.print(myFlexSensor.getX());
      Serial.print(",");
      Serial.print(myFlexSensor.getY());
      Serial.print(",");
      Serial.print(PWMSignal);
      Serial.println();
      delay(SampleRate);
      }
      }
      analogWrite(PWMNY,0);
      delay(hold);
    }
}

void loop()
{
  if (myFlexSensor.available() == true) {
  //for (int i=0; i<=forlooprefresh; i++) {
    Serial.print(myFlexSensor.getX());
    Serial.print(",");
    Serial.print(myFlexSensor.getY());
    Serial.println();
  //}
  }

  if(Serial.available()) {
    int SMA = Serial.read();
    if (SMA == 48) {
      //Serial.println(SMA);
      SMApx();
    }
    if (SMA == 49) { 
      //Serial.println(SMA);
      SMAnx();
    }
    if (SMA == 50)
      SMApy();
    if (SMA == 51)
      SMAny();

      //Serial.println("Not a valid command");
      //Serial.println(SMA);
      //delay(10000);
      
  delay(SampleRate);
  }
  //Serial.readString(); //Flush?
}