#include <Wire.h>
#include "SparkFun_Displacement_Sensor_Arduino_Library.h" // Click here to get the library: http://librarymanager/All#SparkFun_Displacement_Sensor


ADS myFlexSensor;
//PWM of X,Y,-X,-Y
int PWMPX = 5;
int PWMNX = 4;
int PWMPY = 3;
int PWMNY = 2;

int PWMSignal = 0;
int antag_pwm_sig = 15;
int SampleRate = 50;
int forloop = 80;
int forlooprefresh = 1000;
long hold = 115000;
float max_angle_limit = 110.0;
float zero_tolerance = 2.5;
float x_bias = -10.0;
float y_bias = -12.0;
bool start = true;

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

int sign(float val) {
  if (val > 0) {
    return 1;
  } else {
    return -1;
  }
}


bool characterize_sma(int pwm_signal, int pwm_pin, int antag_pwm_pin)
{

  bool max_reached = false;
  for (int i=0; i<=forloop; i++) {
    if (myFlexSensor.available() == true) {
      analogWrite(pwm_pin, pwm_signal);
      float time = millis();
      float x_deg = myFlexSensor.getX();
      float y_deg = myFlexSensor.getY();
      Serial.print(time);
      Serial.print(",");
      Serial.print(x_deg);
      Serial.print(",");
      Serial.print(y_deg);
      Serial.print(",");
      Serial.print(pwm_signal);
      Serial.println();
//      Program stop if SMA above 110 deg
      if ((abs(x_deg) >= 110.0) || (abs(y_deg) >= 110.0)) {
        max_reached = true;
        break;
      }
      delay(SampleRate);
    }
  }
  analogWrite(pwm_pin,0);
  delay(5000);

  // Adding antagonistic actuation
  if (myFlexSensor.available() == true) {
      float x_deg = myFlexSensor.getX() - x_bias;
      float y_deg = myFlexSensor.getY() - y_bias;
      // check if currently characterizing x or y
      if (pwm_pin == 5 || pwm_pin == 4) {
        // characterizing x
        int c = 0;
        int sgn = sign(x_deg);
        float target = -10*sgn;
//        while (abs(x_deg) > zero_tolerance && sign(x_deg) == sgn) {
        while ((abs(x_deg - target) > zero_tolerance)) {
          if (myFlexSensor.available() == true) {
            Serial.print("activating antagonistic x, ");
            Serial.print(antag_pwm_pin);
            Serial.print(", ");
            Serial.print(x_deg);
            Serial.println();
            analogWrite(antag_pwm_pin, antag_pwm_sig);
            if ((abs(x_deg) >= 115) || (abs(y_deg) >= 115)) {
              break;
            }
            delay(SampleRate);
            x_deg = myFlexSensor.getX() - x_bias;
            c++;
          }
        }
      } else {
        // characterizing y
        int c = 0;
        int sgn = sign(y_deg);
        float target = -10*sgn;
        while ((abs(y_deg - target) > zero_tolerance)) {
          if (myFlexSensor.available() == true) {
            Serial.println("activating antagonistic y");
            Serial.print(antag_pwm_pin);
            Serial.print(", ");
            Serial.print(y_deg);
            Serial.println();
            analogWrite(antag_pwm_pin, antag_pwm_sig);
            if ((abs(x_deg) >= 115) || (abs(y_deg) >= 115)) {
              break;
            }
            delay(SampleRate);
            y_deg = myFlexSensor.getY() - y_bias;
            c++;
          }
        }
        
      }
        
  }

  analogWrite(antag_pwm_pin, 0);
  delay(hold);
  return max_reached;    
}


void loop()
{

  Serial.println("waiting to start...");
//  if (Serial.available()) {
//    int SMA = Serial.read();
//    if (SMA == 48) {
//      //Serial.println(SMA);
//      start = true;
//    }
//  }
  delay(5000);
//  start = true;
  if (myFlexSensor.available() == true && start) {
      Serial.println("Characterizing Positive X");
      for (int i = PWMSignal; i < 256; i++) {
        bool max_reached = characterize_sma(i, PWMPX, PWMNX);
        if (max_reached) {
          break;
        }
      }
      Serial.println("Characterizing Negative X");
      for (int i = PWMSignal; i < 256; i++) {
        bool max_reached = characterize_sma(i, PWMNX, PWMPX);
        if (max_reached) {
          break;
        }
      }
      Serial.println("Characterizing Positive Y");
      for (int i = PWMSignal; i < 256; i++) {
        bool max_reached = characterize_sma(i, PWMPY, PWMNY);
        if (max_reached) {
          break;
        }
      }
      Serial.println("Characterizing Negative Y");
      for (int i = PWMSignal; i < 256; i++) {
        bool max_reached = characterize_sma(i, PWMNY, PWMPY);
        if (max_reached) {
          break;
        }
      }
      start = false;
  }
  delay(SampleRate);
  //Serial.readString(); //Flush?
}
