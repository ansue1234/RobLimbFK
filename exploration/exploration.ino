
#include <math.h>
#include <Wire.h>
#include "SparkFun_Displacement_Sensor_Arduino_Library.h" // Click here to get the library: http://librarymanager/All#SparkFun_Displacement_Sensor

struct Coord
{
  /* data */
  float x;
  float y;
};
struct Cell
{
  /* data */
  int x;
  int y;
};

ADS myFlexSensor;
//PWM of X,Y,-X,-Y
const int PWMPX = 5;
const int PWMNX = 3;
const int PWMPY = 2;
const int PWMNY = 4;

const int Length_of_Sample = 10;
const int SampleRate = 50;

const int max_state = 100; //Boundary of state space

//Set Min and Max based on velocities of limb
//Blue Limbs
//const float PWMMaxPX = 30.00;
//const float PWMMinPX = 20.00;
//const float PWMMaxNX = 30.00;
//const float PWMMinNX = 20.00;
//const float PWMMaxPY = 34.00;
//const float PWMMinPY = 21.00;
//const float PWMMaxNY = 30.00;
//const float PWMMinNY = 19.00;

//Teal thin SMA limbs
//const float PWMMaxPX = 20.00;
//const float PWMMinPX = 10.00;
//const float PWMMaxNX = 20.00;
//const float PWMMinNX = 10.00;
//const float PWMMaxPY = 20.00;
//const float PWMMinPY = 10.00;
//const float PWMMaxNY = 20.00;
//const float PWMMinNY = 10.00;

// Purple limbs

const float PWMMaxPX = 40.00;
const float PWMMinPX = 20.00;
const float PWMMaxNX = 40.00;
const float PWMMinNX = 20.00;
const float PWMMaxPY = 40.00;
const float PWMMinPY = 20.00;
const float PWMMaxNY = 40.00;
const float PWMMinNY = 20.00;

const int Cool_Time = 20000;

const int for_loop_refresh = 100;

int kill = 0;
const int num_PWM_signals = 21;

float PWM_x_signals[num_PWM_signals];
float PWM_y_signals[num_PWM_signals];

// Maximum change in the binned signal, i.e. the number of index in the PWM array 
const int max_change_throttle_x = 3;
const int max_change_throttle_y = 3;
// Exploration map
const int num_rows = 6;
const int num_cols = 6;
const int num_cells = num_rows*num_cols;
int count_grid [num_cells] = {0}; // x is columns, y is rows
float running_max = 0.0; // using float to prevent integer division
float total_counts = 0.0;
Coord* current_coord;
Cell* current_cell;
Coord* sampled_coord;
Cell* sampled_cell;
int current_x_throttle = 0;
int current_y_throttle = 0;

int x_y_to_index(float x, float y) {
  float resolution_y = 2*max_state / num_rows;
  float resolution_x = 2*max_state / num_cols;
  int row = (int) ((y + max_state) / resolution_y);
  int col = (int) ((x + max_state) / resolution_x);
  row = max(0, min(num_rows - 1, row));
  col = max(0, min(num_cols - 1, col));
  return row*num_cols + col;
}

void update_count_grid(float x, float y) {
  int index = x_y_to_index(x, y);
//  Serial.print("index: ");
//  Serial.println(index);
  count_grid[index] += 1;
  total_counts += 1;
  if (count_grid[index] > running_max) {
    running_max = count_grid[index];
  }
}

void index_to_x_y(int index, Coord* x_y) {
  float resolution_y = 2*max_state / num_rows;
  float resolution_x = 2*max_state / num_cols;
  int row = index / num_cols;
  int col = index % num_cols;
  float x = (2.0 * col + 1.0) / 2.0 * resolution_x - max_state;
  float y = (2.0 * row + 1.0) / 2.0 * resolution_y - max_state;
  x_y->x = x;
  x_y->y = y;
}

void get_sampled_x_y(float sampled_val, Coord* x_y) {
  // sampled_val is between 0 and 1
  int counts_tot = (running_max + 1) * num_rows * num_cols - total_counts;
  float target = sampled_val * counts_tot;
  for (int i = 0; i < num_rows*num_cols; i++) {
    target -= ((running_max + 1) - count_grid[i]);
    if (target <= 0) {
      return index_to_x_y(i, x_y);
    }
  }
}

void x_y_to_cell(float x, float y, Cell* cell) {
  float resolution_y = 2 * max_state / num_rows;
  float resolution_x = 2 * max_state / num_cols;
  int row = (int) ((y + max_state) / resolution_y);
  int col = (int) ((x + max_state) / resolution_x);
  cell->x = max(0, min(num_cols - 1, col));
  cell->y = max(0, min(num_rows - 1, row));
}

void get_sampled_cell(Cell* cell, Coord* coord){
  float rand_select_thresh = random(0, 1000) / 1000.0;
  get_sampled_x_y(rand_select_thresh, coord);
  x_y_to_cell(coord->x, coord->y, cell);
}

//////////////////////////////////////////Generate Arrays
// P_Max, P_Min, N_Max, N_Min stands for positive max, positive min, negative max, negative min respectively
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

///////////////////////////////////////////////////////////Display Data Output and Actuate SMA
int DataWrite(int x_throttle, int y_throttle)
{
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

    for (int iter = 0; iter < Length_of_Sample; iter++) {
      if (myFlexSensor.available() == true) {
        Serial.print("Iter: ");
        Serial.println(iter);
        current_coord->x = myFlexSensor.getX();
        current_coord->y = myFlexSensor.getY();
        float t = millis();
        Serial.print(t);
        Serial.print(",");
        Serial.print(current_coord->x);
        Serial.print(",");
        Serial.print(current_coord->y);
        
        //Serial.println("NX 20-35");
        if (((actual_px_pwm != 0) && (actual_nx_pwm != 0)) || ((actual_py_pwm != 0) && (actual_ny_pwm != 0))) {
          Serial.println("antagonistic pair both activated!!!!!");   
          return 1;         
        } else {
          analogWrite(PWMPX,actual_px_pwm);
          analogWrite(PWMNX,actual_nx_pwm);
          analogWrite(PWMPY,actual_py_pwm);
          analogWrite(PWMNY,actual_ny_pwm);
        }
        Serial.print(",");
        Serial.print((float)x_throttle);
        Serial.print(",");
        Serial.print((float)y_throttle);
        Serial.print(",");
        Serial.print(actual_px_pwm);
        Serial.print(",");
        Serial.print(actual_nx_pwm);
        Serial.print(",");
        Serial.print(actual_py_pwm);
        Serial.print(",");
        Serial.print(actual_ny_pwm);
        Serial.println();
        if (abs(current_coord->x) > max_state || abs(current_coord->y) > max_state) {
          Serial.println();
          Serial.print("exceeding");
          Serial.println();
          return 1;
        }
        update_count_grid(current_coord->x, current_coord->y);
        delay(SampleRate);
      }
    }      
    return 0;
}

int get_next_throttle(int current_throttle, int max_change_throttle, int diff) {
  if (diff == 0) {
    if (current_throttle > 0) {
      // Decelerate move in negative direction
      diff = -1;
    } else if (current_throttle < 0) {
      // Decelerate move in positive direction
      diff = 1;
    } 
  }

  if (diff > 0) {
    // move in positive direction
    int throttle_bound = max_change_throttle;
    if (current_throttle + max_change_throttle >= num_PWM_signals / 2) {
      throttle_bound = num_PWM_signals / 2 - current_throttle;
    }
    int throttle_change = random(0, throttle_bound + 1);
    return current_throttle + throttle_change;
  } else if (diff < 0) {
    //move in negative
    int throttle_bound = -max_change_throttle;
    if (current_throttle - max_change_throttle <= -num_PWM_signals / 2) {
      throttle_bound = -num_PWM_signals / 2 - current_throttle;
    }
    int throttle_change = random(throttle_bound, 1);
    return current_throttle + throttle_change;
  } else {
    return current_throttle;
  }
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
  randomSeed(analogRead(0));

  Wire.begin();

  if (myFlexSensor.begin() == false)
  {
    Serial.println(F("No sensor detected. Check wiring. Freezing..."));
    while (1)
      ;
  }

///////////////////////////////////////////////Refresh Loop can remove 
  current_coord = (Coord*)malloc(sizeof(Coord));
  sampled_coord = (Coord*)malloc(sizeof(Coord));
  sampled_cell = (Cell*)malloc(sizeof(Cell));
  current_cell = (Cell*)malloc(sizeof(Cell));

  for (int i=0; i<=for_loop_refresh; i++) {
    if (myFlexSensor.available() == true) {
      Serial.print(myFlexSensor.getX());
      Serial.print(",");
      Serial.print(myFlexSensor.getY());
      Serial.println();
      current_coord->x = myFlexSensor.getX();
      current_coord->y = myFlexSensor.getY();
      x_y_to_cell(current_coord->x, current_coord->y, current_cell);
    }
  }
  Serial.println("Refresh Complete");
  get_sampled_cell(sampled_cell, sampled_coord);
  randomSeed(analogRead(0));
  PWMRange(PWM_x_signals, PWMMaxPX, PWMMinPX, PWMMaxNX, PWMMinNX);
  PWMRange(PWM_y_signals, PWMMaxPY, PWMMinPY, PWMMaxNY, PWMMinNY);
  for (int i=0; i<num_PWM_signals; i++) {
    Serial.print(PWM_x_signals[i]);
    Serial.print(",");
    Serial.print(PWM_y_signals[i]);
    Serial.println();
  }
  Serial.println("PWM Signals Generated");
  Serial.println("Setup Complete");
}

void loop()
{
  
  ///////////////////////////////////////////////////Random Selection
  float dx = sampled_coord->x - current_coord->x;
  float dy = sampled_coord->y - current_coord->y;

  x_y_to_cell(current_coord->x, current_coord->y, current_cell);
  Serial.print("Sampled Pos");
  Serial.print(sampled_coord->x);
  Serial.print(",");
  Serial.print(sampled_coord->y);
  Serial.println();
  Serial.print("Current Pos");
  Serial.print(current_coord->x);
  Serial.print(",");
  Serial.print(current_coord->y);
  Serial.println();

  get_sampled_cell(sampled_cell, sampled_coord);
  int x_diff = sampled_cell->x - current_cell->x;
  int y_diff = sampled_cell->y - current_cell->y;
  // Moving else where and then sampling once within range of target
  if (dx*dx + dy*dy <= 400) {
//    get_sampled_cell(sampled_cell, sampled_coord);
    int rand_dir = random(0, 8);
    if (rand_dir == 0) {
      x_diff = 1;
      y_diff = 1;
    } else if (rand_dir == 1) {
      x_diff = 1;
      y_diff = 0;
    } else if (rand_dir == 2) {
      x_diff = 1;
      y_diff = -1;
    } else if (rand_dir == 3) {
      x_diff = 0;
      y_diff = 1;
    } else if (rand_dir == 4) {
      x_diff = 0;
      y_diff = -1;
    } else if (rand_dir == 5) {
      x_diff = -1;
      y_diff = 1;
    } else if (rand_dir == 6) {
      x_diff = -1;
      y_diff = 0;
    } else {
      x_diff = -1;
      y_diff = -1;
    }
  }

  current_x_throttle = get_next_throttle(current_x_throttle, max_change_throttle_x, x_diff);
  current_y_throttle = get_next_throttle(current_y_throttle, max_change_throttle_y, y_diff);
  Serial.println("current throttle:");
  Serial.println(current_x_throttle);
  Serial.println(current_y_throttle);
  kill = DataWrite(current_x_throttle, current_y_throttle);
  
  Serial.println("count grid ");
  for (int i=0; i < num_cells; i++) {
    Serial.print(count_grid[i]);
    Serial.print(",");
  }
  Serial.println();
  
  
  ///////////////////////////Safety Mechanism 
  
  if (kill == 1) {
    analogWrite(PWMPX, 0);
    analogWrite(PWMNX, 0);
    analogWrite(PWMPY, 0);
    analogWrite(PWMNY, 0);
    current_x_throttle = 0;
    current_y_throttle = 0;
    get_sampled_cell(sampled_cell, sampled_coord);
    delay(Cool_Time);
  }
} 


 
