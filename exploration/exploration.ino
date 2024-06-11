
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
const int PWMNX = 4;
const int PWMPY = 3;
const int PWMNY = 2;

int PWMPX_Sig_learned = 0;
int PWMNX_Sig_learned = 0;
int PWMPY_Sig_learned = 0;
int PWMNY_Sig_learned = 0;

const int Length_of_Sample = 10;
const int SampleRate = 50;

const int max_state = 110; //Boundary of state space

//Set Min and Max based on velocities of limb
const float PWMMaxPX = 30.00;
const float PWMMinPX = 20.00;
const float PWMMaxNX = 35.00;
const float PWMMinNX = 20.00;
const float PWMMaxPY = 34.00;
const float PWMMinPY = 21.00;
const float PWMMaxNY = 30.00;
const float PWMMinNY = 19.00;

float BX;
float BY;

const int Cool_Time = 20000;

const int for_loop_refresh = 100;

int kill = 0;

float PWM_signals[11];

// Exploration map
const int num_rows = 22;
const int num_cols = 22;
const int num_cells = num_rows*num_cols;
int count_grid [num_cells] = {0}; // x is columns, y is rows
float running_max = 0.0; // using float to prevent integer division
float total_counts = 0.0;
Coord* current_coord;

int x_y_to_index(float x, float y) {
  float resolution_y = max_state / num_rows;
  float resolution_x = max_state / num_cols;
  int row = (int) ((y - max_state) / resolution_y);
  int col = (int) ((x - max_state) / resolution_x);
  return row*num_cols + col;
}

void update_count_grid(float x, float y) {
  int index = x_y_to_index(x, y);
  count_grid[index] += 1;
  total_counts += 1;
  if (count_grid[index] > running_max) {
    running_max = count_grid[index];
  }
}

Coord* index_to_x_y(int index) {
  float resolution_y = max_state / num_rows;
  float resolution_x = max_state / num_cols;
  int row = index / num_cols;
  int col = index % num_cols;
  float x = col * resolution_x - max_state;
  float y = row * resolution_y - max_state;
  Coord* x_y = (Coord*) malloc(sizeof(Coord));
  x_y->x = x;
  x_y->y = y;
  return x_y;
}

Coord* get_sampled_x_y(float sampled_val) {
  // sampled_val is between 0 and 1
  int counts_tot = (running_max + 1) * num_rows * num_cols - total_counts;
  float target = sampled_val * counts_tot;
  for (int i = 0; i < num_rows*num_cols; i++) {
    target -= ((running_max + 1) - count_grid[i]);
    if (target <= 0) {
      return index_to_x_y(i);
    }
  }
}

Cell* x_y_to_cell(float x, float y) {
  float resolution_y = 2 * max_state / num_rows;
  float resolution_x = 2 * max_state / num_cols;
  int row = (int) ((y - max_state) / resolution_y);
  int col = (int) ((x - max_state) / resolution_x);
  row = max(0, min(num_cells - 1, row))
  col = max(0, min(num_cells - 1, col))
  Cell* cell = (Cell*) malloc(sizeof(Cell));
  cell->x = col;
  cell->y = row;
  return cell;
}

//////////////////////////////////////////Generate Arrays

float* PWMRange(float PWMMax, float PWMMin){
  float increment = (PWMMax - PWMMin)/9; 
  //Serial.println(increment);
  for (int i=0; i<=10; i++) {
    if (i == 0) {
      PWM_signals[i] = 0;
    //Serial.println(PWM_signals[i]);
    } else {
      PWM_signals[i] = round(PWMMin + increment*(i-1)); //round to nearest integer for PWM
      //Serial.println((PWM_signals[i]);
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

  current_coord = (Coord*)malloc(sizeof(Coord));

}



///////////////////////////////////////////////////////////Display Data Output and Actuate SMA
int DataWrite(int PWMPX_Sig_learned, int PWMNX_Sig_learned, int PWMPY_Sig_learned, int PWMNY_Sig_learned, int PWMMaxPX, int PWMMinPX, int PWMMaxPY, int PWMMinPY, int PWMMaxNX, int PWMMinNX, int PWMMaxNY, int PWMMinNY)
{
      for (int iter = 0; iter < Length_of_Sample; iter++) {
        if (myFlexSensor.available() == true) {
          current_coord->x = myFlexSensor.getX();
          current_coord->y = myFlexSensor.getY();
          float t = millis();
          Serial.print(t);
          Serial.print(",");
          Serial.print(current_coord->x);
          Serial.print(",");
          Serial.print(current_coord->y);
          Serial.print(",");
          Serial.print((float) PWMPX_Sig_learned);
          Serial.print(",");
          Serial.print((float) PWMNX_Sig_learned);
          Serial.print(",");
          Serial.print((float) PWMPY_Sig_learned);
          Serial.print(",");
          Serial.print((float) PWMNY_Sig_learned);
          //Serial.println("PX 20-30");
          float* PXarray = PWMRange(PWMMaxPX, PWMMinPX);
          int PWMPX_Sig = PXarray[PWMPX_Sig_learned];
          //Serial.println("NX 20-35");
          float* NXarray = PWMRange(PWMMaxNX, PWMMinNX);
          int PWMNX_Sig = NXarray[PWMNX_Sig_learned];
          //Serial.println("PY 21-34");
          float* PYarray = PWMRange(PWMMaxPY, PWMMinPY);
          int PWMPY_Sig = PYarray[PWMPY_Sig_learned];
          //Serial.println("NY 19-30");
          float* NYarray = PWMRange(PWMMaxNY, PWMMinNY);
          int PWMNY_Sig = NYarray[PWMNY_Sig_learned];
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

void loop()
{
  
  ///////////////////////////////////////////////////Random Selection
  
  float rand_select_thresh = random(0, 1000) / 1000.0;
  Coord* sampled_coord = get_sampled_x_y(rand_select_thresh);
  Cell* sampled_cell = x_y_to_cell(sampled_coord->x, sampled_coord->y);
  Cell* current_cell = x_y_to_cell(current_coord->x, current_coord->y);
  
  int x_diff = sampled_cell->x - current_cell->x;
  int y_diff = sampled_cell->y - current_cell->y;
  
  if (x_diff == 0 && y_diff == 0) {
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

  if (x_diff > 0) {
    PWMPX_Sig_learned = random(0, 12);
    PWMNX_Sig_learned = 0;
  } else if (x_diff == 0) {
    PWMPX_Sig_learned = 0;
    PWMNX_Sig_learned = 0;
  } else {
    PWMPX_Sig_learned = 0;
    PWMNX_Sig_learned = random(0, 12);
  }

  if (y_diff > 0) {
    PWMPY_Sig_learned = random(0, 12);
    PWMNY_Sig_learned = 0;
  } else if (y_diff == 0) {
    PWMPY_Sig_learned = 0;
    PWMNY_Sig_learned = 0;
  } else {
    PWMPY_Sig_learned = 0;
    PWMNY_Sig_learned = random(0, 12);
  }
  kill = DataWrite(PWMPX_Sig_learned, PWMNX_Sig_learned, PWMPY_Sig_learned, PWMNY_Sig_learned, PWMMaxPX, PWMMinPX, PWMMaxPY, PWMMinPY, PWMMaxNX, PWMMinNX, PWMMaxNY, PWMMinNY);
  ///////////////////////////Safety Mechanism 
  
  if (kill == 1) {
    analogWrite(PWMPX, 0);
    analogWrite(PWMNX, 0);
    analogWrite(PWMPY, 0);
    analogWrite(PWMNY, 0);
    delay(Cool_Time);
  }
  free(sampled_cell);
  free(sampled_coord);
} 


 
