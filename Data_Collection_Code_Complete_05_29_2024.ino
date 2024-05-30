
#include <math.h>
#include <Wire.h>
#include "SparkFun_Displacement_Sensor_Arduino_Library.h" // Click here to get the library: http://librarymanager/All#SparkFun_Displacement_Sensor

ADS myFlexSensor;
//PWM of X,Y,-X,-Y
int PWMPX = 3;//5;
int PWMNX = 5;//4;
int PWMPY = 6;//3;
int PWMNY = 9;//2;

int PWMPX_Sig_learned = 0;
int PWMNX_Sig_learned = 0;
int PWMPY_Sig_learned = 0;
int PWMNY_Sig_learned = 0;

int Length_of_Sample = 10;
int SampleRate = 50;

//Set Min and Max based on velocities of limb
int PWMMaxPX = 10;
int PWMMinPX = 30;
int PWMMaxNX = 10;
int PWMMinNX = 30;
int PWMMaxPY = 10;
int PWMMinPY = 30;
int PWMMaxNY = 10;
int PWMMinNY = 30;

float PWMPX_Sig = 0;
float PWMNX_Sig = 0;
float PWMPY_Sig = 0;
float PWMNY_Sig = 0;

float BX;
float BY;

int Cool_Time = 20000;

int for_loop_refresh = 100;

int kill = 0;

float PWM_signals[11];

//////////////////////////////////////////Generate Arrays

float* PWMRange(int PWMMax, int PWMMin)
{
  float increment = (PWMMax - PWMMin)/10;
  for (int i=0; i<=10; i++) {
    if (i == 0) {
    PWM_signals[i] = 0;
    }
    else {
      PWM_signals[i] = round(PWMMax - increment*i); //round to nearest integer for PWM
    }
  }
  return PWM_signals; 
}

void setup()
{
  Serial.begin(9600);
  while (!Serial)
    ;
  Serial.println("SparkFun Displacement Sensor");
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

///////////////////////////////////////////////Refresh Loop can remove 

  for (int i=0; i<=for_loop_refresh; i++) {
  if (myFlexSensor.available() == true) {
    Serial.print(myFlexSensor.getX());
    Serial.print(",");
    Serial.print(myFlexSensor.getY());
    Serial.println();
    }
  }
  Serial.println("Refresh Complete");

}



///////////////////////////////////////////////////////////Display Data Output and Actuate SMA
int DataWrite(int PWMPX_Sig_learned, int PWMNX_Sig_learned, int PWMPY_Sig_learned, int PWMNY_Sig_learned, int PWMMaxPX, int PWMMinPX, int PWMMaxPY, int PWMMinPY, int PWMMaxNX, int PWMMinNX, int PWMMaxNY, int PWMMinNY)
{
      for (int iter = 0; iter < Length_of_Sample; iter++) {
      if (myFlexSensor.available() == true) {
      BX = myFlexSensor.getX();
      BY = myFlexSensor.getY();
      }
      float time = millis();
      Serial.print(time);
      Serial.print(",");
      Serial.print(BX);
      Serial.print(",");
      Serial.print(BY);
      Serial.print(",");
      float Learned_PX = PWMPX_Sig_learned;
      Serial.print(Learned_PX);
      Serial.print(",");
      float Learned_NX = PWMNX_Sig_learned;
      Serial.print(Learned_NX);
      Serial.print(",");
      float Learned_PY = PWMPY_Sig_learned;
      Serial.print(Learned_PY);
      Serial.print(",");
      float Learned_NY = PWMNY_Sig_learned;
      Serial.print(Learned_NY);
      float* PXarray = PWMRange(PWMMaxPX, PWMMinPX);
      PWMPX_Sig = PXarray[PWMPX_Sig_learned];
      float* NXarray = PWMRange(PWMMaxNX, PWMMinNX);
      PWMNX_Sig = NXarray[PWMNX_Sig_learned];
      float* PYarray = PWMRange(PWMMaxPY, PWMMinPY);
      PWMPY_Sig = PYarray[PWMPY_Sig_learned];
      float* NYarray = PWMRange(PWMMaxNY, PWMMinNY);
      PWMNY_Sig = NYarray[PWMNY_Sig_learned];
      analogWrite(PWMPX,PWMPX_Sig);
      analogWrite(PWMNX,PWMNX_Sig);
      analogWrite(PWMPY,PWMPY_Sig);
      analogWrite(PWMNY,PWMNY_Sig);
      Serial.print(",");
      Serial.print(PWMPX_Sig);
      Serial.print(",");
      Serial.print(PWMNX_Sig);
      Serial.print(",");
      Serial.print(PWMPY_Sig);
      Serial.print(",");
      Serial.print(PWMNY_Sig);
      Serial.println();
      delay(SampleRate);
      }
      if (abs(BX) > 110 || abs(BY) > 110) {
        kill = 1;
      }
      else {
        kill = 0;
      }
      return kill;
    
}

void loop()

{
  
///////////////////////////////////////////////////Random Selection

int Number_of_SMA = random(1,3);
int SMA_Selected = random(1,5);
int Array_Value_Selected1 = random(1,11);
int Array_Value_Selected2 = random(1,11);
int Right_Left = random(1,2);

////////////////////////////////////////////////////1 SMA

if (Number_of_SMA == 1) {
  if (SMA_Selected == 1) {
    PWMPX_Sig_learned = Array_Value_Selected1;
    PWMNX_Sig_learned = 0;
    PWMPY_Sig_learned = 0;
    PWMNY_Sig_learned = 0;
    int kill = DataWrite(PWMPX_Sig_learned,PWMNX_Sig_learned,PWMPY_Sig_learned,PWMNY_Sig_learned, PWMMaxPX, PWMMinPX, PWMMaxPY, PWMMinPY, PWMMaxNX, PWMMinNX, PWMMaxNY, PWMMinNY);
  }
  if (SMA_Selected == 2) {
    PWMPX_Sig_learned = 0;
    PWMNX_Sig_learned = Array_Value_Selected1;
    PWMPY_Sig_learned = 0;
    PWMNY_Sig_learned = 0;
    int kill = DataWrite(PWMPX_Sig_learned,PWMNX_Sig_learned,PWMPY_Sig_learned,PWMNY_Sig_learned, PWMMaxPX, PWMMinPX, PWMMaxPY, PWMMinPY, PWMMaxNX, PWMMinNX, PWMMaxNY, PWMMinNY);
  }
  if (SMA_Selected == 3) {
    PWMPX_Sig_learned = 0;
    PWMNX_Sig_learned = 0;
    PWMPY_Sig_learned = Array_Value_Selected1;
    PWMNY_Sig_learned = 0;
    int kill = DataWrite(PWMPX_Sig_learned,PWMNX_Sig_learned,PWMPY_Sig_learned,PWMNY_Sig_learned, PWMMaxPX, PWMMinPX, PWMMaxPY, PWMMinPY, PWMMaxNX, PWMMinNX, PWMMaxNY, PWMMinNY);
  }
  if (SMA_Selected == 4) {
    PWMPX_Sig_learned = 0;
    PWMNX_Sig_learned = 0;
    PWMPY_Sig_learned = 0;
    PWMNY_Sig_learned = Array_Value_Selected1;
    int kill = DataWrite(PWMPX_Sig_learned,PWMNX_Sig_learned,PWMPY_Sig_learned,PWMNY_Sig_learned, PWMMaxPX, PWMMinPX, PWMMaxPY, PWMMinPY, PWMMaxNX, PWMMinNX, PWMMaxNY, PWMMinNY);
  }
} 

///////////////////////////////////////////////2 SMA

if (Number_of_SMA == 2) {
  if (SMA_Selected == 1) {
    PWMPX_Sig_learned = Array_Value_Selected1;
    PWMNX_Sig_learned = 0;
    if (Right_Left == 1) {
    PWMPY_Sig_learned = Array_Value_Selected2;
    PWMNY_Sig_learned = 0;
    }
    if (Right_Left == 2) {
    PWMPY_Sig_learned = 0;
    PWMNY_Sig_learned = Array_Value_Selected2;
    }
    int kill = DataWrite(PWMPX_Sig_learned,PWMNX_Sig_learned,PWMPY_Sig_learned,PWMNY_Sig_learned, PWMMaxPX, PWMMinPX, PWMMaxPY, PWMMinPY, PWMMaxNX, PWMMinNX, PWMMaxNY, PWMMinNY);
  }
  if (SMA_Selected == 2) {
    PWMPX_Sig_learned = 0;
    PWMNX_Sig_learned = Array_Value_Selected1;
    if (Right_Left == 1) {
    PWMPY_Sig_learned = Array_Value_Selected2;
    PWMNY_Sig_learned = 0;
    }
    if (Right_Left == 2) {
    PWMPY_Sig_learned = 0;
    PWMNY_Sig_learned = Array_Value_Selected2;
    }
    int kill = DataWrite(PWMPX_Sig_learned,PWMNX_Sig_learned,PWMPY_Sig_learned,PWMNY_Sig_learned, PWMMaxPX, PWMMinPX, PWMMaxPY, PWMMinPY, PWMMaxNX, PWMMinNX, PWMMaxNY, PWMMinNY);
  }
  if (SMA_Selected == 3) {
    PWMPY_Sig_learned = Array_Value_Selected1;
    PWMNY_Sig_learned = 0;
    if (Right_Left == 1) {
    PWMPX_Sig_learned = Array_Value_Selected2;
    PWMNX_Sig_learned = 0;
    }
    if (Right_Left == 2) {
    PWMPX_Sig_learned = 0;
    PWMNX_Sig_learned = Array_Value_Selected2;
    }
    }
    int kill = DataWrite(PWMPX_Sig_learned,PWMNX_Sig_learned,PWMPY_Sig_learned,PWMNY_Sig_learned, PWMMaxPX, PWMMinPX, PWMMaxPY, PWMMinPY, PWMMaxNX, PWMMinNX, PWMMaxNY, PWMMinNY);
  }
  if (SMA_Selected == 4) {
    PWMPY_Sig_learned = 0;
    PWMNY_Sig_learned = Array_Value_Selected1;
    if (Right_Left == 1) {
    PWMPX_Sig_learned = Array_Value_Selected2;
    PWMNX_Sig_learned = 0;
    }
    if (Right_Left == 2) {
    PWMPX_Sig_learned = 0;
    PWMNX_Sig_learned = Array_Value_Selected2;
    }
    int kill = DataWrite(PWMPX_Sig_learned,PWMNX_Sig_learned,PWMPY_Sig_learned,PWMNY_Sig_learned, PWMMaxPX, PWMMinPX, PWMMaxPY, PWMMinPY, PWMMaxNX, PWMMinNX, PWMMaxNY, PWMMinNY);
  }

///////////////////////////Safety Mechanism 

if (kill == 1) {
  analogWrite(PWMPX, 0);
  analogWrite(PWMNX, 0);
  analogWrite(PWMPY, 0);
  analogWrite(PWMNY, 0);
  delay(Cool_Time);
}

} 


 


