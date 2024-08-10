#include <math.h>
#include <Wire.h>
#include "SparkFun_Displacement_Sensor_Arduino_Library.h" // Click here to get the library: http://librarymanager/All#SparkFun_Displacement_Sensor

ADS myFlexSensor;
String received_cmd;
int x_throttle;
int y_throttle;
const float PWMMaxPX = 35.00;
const float PWMMinPX = 20.00;
const float PWMMaxNX = 35.00;
const float PWMMinNX = 20.00;
const float PWMMaxPY = 35.00;
const float PWMMinPY = 20.00;
const float PWMMaxNY = 35.00;
const float PWMMinNY = 20.00;

const int num_PWM_signals = 21;

float PWM_x_signals[num_PWM_signals];
float PWM_y_signals[num_PWM_signals];

const int PWMPX = 5;
const int PWMNX = 3;
const int PWMPY = 2;
const int PWMNY = 4;
const int sample_period = 50;

const int max_state = 100; //Boundary of state space

void PWMRange(float* pwm_signals, float PWM_P_Max, float PWM_P_Min, float PWM_N_Max, float PWM_N_Min){
  int num_intervals = num_PWM_signals / 2;
  float p_increment = (PWM_P_Max - PWM_P_Min)/(num_intervals - 1); 
  float n_increment = (PWM_N_Max - PWM_N_Min)/(num_intervals - 1);
  //Serial.println(increment);
  // Negatives
  for (int i=0; i<num_intervals; i++) {
    pwm_signals[i] = (-PWM_N_Max + i*n_increment)*-1;
  }
  pwm_signals[num_intervals] = 0;
  // Positives
  for (int j=0; j<num_intervals; j++) {
    pwm_signals[num_intervals + j + 1] = PWM_P_Min + j*p_increment;
  }
}



void setup() {
  // put your setup code here, to run once:
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
  PWMRange(PWM_x_signals, PWMMaxPX, PWMMinPX, PWMMaxNX, PWMMinNX);
  PWMRange(PWM_y_signals, PWMMaxPY, PWMMinPY, PWMMaxNY, PWMMinNY);
}

void loop() {
  // put your main code here, to run repeatedly:
  while (Serial.available() > 0 ) {
    received_cmd = Serial.readStringUntil('\n');
  }
  int comma = received_cmd.indexOf(',');
  if (comma != -1) {
    float interval_range = (num_PWM_signals - 1) / 2;
    x_throttle = round(interval_range * received_cmd.substring(0, comma).toFloat());
    y_throttle = round(interval_range * received_cmd.substring(comma + 1,received_cmd.length()).toFloat());
  }
  float actual_px_pwm = 0;
  float actual_py_pwm = 0;
  float actual_nx_pwm = 0;
  float actual_ny_pwm = 0;

  int num_intervals = num_PWM_signals / 2;

  int throttle_x = x_throttle + num_intervals;
  if (x_throttle > 0) {
    actual_px_pwm = PWM_x_signals[throttle_x];
  } else if (x_throttle < 0) {
    actual_nx_pwm = PWM_x_signals[throttle_x];
  }
  
  int throttle_y = y_throttle + num_intervals;
  if (y_throttle > 0) {
    actual_py_pwm = PWM_y_signals[throttle_y];
  } else if (y_throttle < 0) {
    actual_ny_pwm = PWM_y_signals[throttle_y];
  }
        
  if (myFlexSensor.available() == true) {
          float BX = myFlexSensor.getX();
          float BY = myFlexSensor.getY();
          Serial.print("Data,");
          Serial.print(BX);
          Serial.print(",");
          Serial.print(BY);
          Serial.print(",");
          Serial.print(x_throttle);
          Serial.print(",");
          Serial.print(y_throttle);
          Serial.println();
          if ((abs(BX) < max_state) && (abs(BY) < max_state)) {
            analogWrite(PWMPX,actual_px_pwm);
            analogWrite(PWMNX,actual_nx_pwm);
            analogWrite(PWMPY,actual_py_pwm);
            analogWrite(PWMNY,actual_ny_pwm);
          } else {
            analogWrite(PWMPX,0);
            analogWrite(PWMNX,0);
            analogWrite(PWMPY,0);
            analogWrite(PWMNY,0);
          }
  }
  delay(sample_period);
}
