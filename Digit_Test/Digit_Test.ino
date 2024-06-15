// //SMA Output
// int SMA1 = 2;//3
// int SMA2 = 3;//5
// int SMA3 = 4;//6
// int SMA4 = 5;//9

// int PWMValue = 190;
// int TimeA = 3500;

// void setup() {
// Serial.begin(115200);
//   // put your setup code here, to run once:
// pinMode(SMA1, OUTPUT);
// pinMode(SMA2, OUTPUT);
// pinMode(SMA3, OUTPUT);
// pinMode(SMA4, OUTPUT);
// }

// void loop() {
//   // put your main code here, to run repeatedly:
//  if (Serial.available()>0){
// Serial.println("Digit Control Test");
// int command = Serial.read();
// if (command == 50){
//   analogWrite(SMA1, PWMValue);
//   Serial.println("Command 2");
//   Serial.println("Pin 3");
//   delay(TimeA);
//   Serial.println("stop");
//   analogWrite(SMA1, 0);
//   analogWrite(SMA2, 0);
//   analogWrite(SMA3, 0);
//   analogWrite(SMA4, 0);
// }

// else if (command == 51){
//   analogWrite(SMA2, PWMValue);
//   Serial.println("Command 3");
//   Serial.println("Pin 5");
//   delay(TimeA);
//   Serial.println("stop");
//   analogWrite(SMA1, 0);
//   analogWrite(SMA2, 0);
//   analogWrite(SMA3, 0);
//   analogWrite(SMA4, 0);
// }

// else if (command == 52){  
//   analogWrite(SMA3, PWMValue);
//   Serial.println("Command 4");
//   Serial.println("Pin 6");
//   delay(TimeA);
//   Serial.println("stcd op");
//   analogWrite(SMA1, 0);
//   analogWrite(SMA2, 0);
//   analogWrite(SMA3, 0);
//   analogWrite(SMA4, 0);
// }

// else if (command == 53){
//   analogWrite(SMA4, PWMValue);
//   Serial.println("Command 5");
//   Serial.println("Pin 9");
//   delay(TimeA);
//   Serial.println("stop");
//   analogWrite(SMA1, 0);
//   analogWrite(SMA2, 0);
//   analogWrite(SMA3, 0);
//   analogWrite(SMA4, 0);
// }

// else{
// analogWrite(SMA1, 0);
// analogWrite(SMA2, 0);
// analogWrite(SMA3, 0);
// analogWrite(SMA4, 0);
// }

// }
// }
//include arduino.h

//SMA Output
int SMA1 = 5;
int SMA2 = 4;
int SMA3 = 3;
int SMA4 = 2;

int PWMValue = 30;
int TimeA = 2000;

void setup() {
Serial.begin(9600);
  // put your setup code here, to run once:
pinMode(SMA1, OUTPUT);
pinMode(SMA2, OUTPUT);
pinMode(SMA3, OUTPUT);
pinMode(SMA4, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
 if (Serial.available()>0){
Serial.println("Digit Control Test");
int command = Serial.read();
if (command == 100){
  analogWrite(SMA1, PWMValue);
  Serial.println("Command 2");
  Serial.println("PX");
  delay(TimeA);
  analogWrite(SMA1, 0);
  analogWrite(SMA2, 0);
  analogWrite(SMA3, 0);
  analogWrite(SMA4, 0);
}

else if (command == 97){
  analogWrite(SMA2, PWMValue);
  Serial.println("Command 3");
  Serial.println("NX");
  delay(TimeA);
  analogWrite(SMA1, 0);
  analogWrite(SMA2, 0);
  analogWrite(SMA3, 0);
  analogWrite(SMA4, 0);
}

else if (command == 119){  
  analogWrite(SMA3, PWMValue);
  Serial.println("Command 4");
  Serial.println("PY");
  delay(TimeA);
  analogWrite(SMA1, 0);
  analogWrite(SMA2, 0);
  analogWrite(SMA3, 0);
  analogWrite(SMA4, 0);
}

else if (command == 115){
  analogWrite(SMA4, PWMValue);
  Serial.println("Command 5");
  Serial.println("NY");
  delay(TimeA);
  analogWrite(SMA1, 0);
  analogWrite(SMA2, 0);
  analogWrite(SMA3, 0);
  analogWrite(SMA4, 0);
}
else if (command == 101){
  analogWrite(SMA1, PWMValue);
  analogWrite(SMA3, PWMValue);
  Serial.println("Command 6");
  Serial.println("PX, PY");
  delay(TimeA);
  analogWrite(SMA1, 0);
  analogWrite(SMA2, 0);
  analogWrite(SMA3, 0);
  analogWrite(SMA4, 0);
}
else if (command == 120){
  analogWrite(SMA1, PWMValue);
  analogWrite(SMA4, PWMValue);
  Serial.println("Command 6");
  Serial.println("PX, NY");
  delay(TimeA);
  analogWrite(SMA1, 0);
  analogWrite(SMA2, 0);
  analogWrite(SMA3, 0);
  analogWrite(SMA4, 0);
}else if (command == 122){
  analogWrite(SMA2, PWMValue);
  analogWrite(SMA4, PWMValue);
  Serial.println("Command 6");
  Serial.println("NX, NY");
  delay(TimeA);
  analogWrite(SMA1, 0);
  analogWrite(SMA2, 0);
  analogWrite(SMA3, 0);
  analogWrite(SMA4, 0);
}
else if (command == 113){
  analogWrite(SMA2, PWMValue);
  analogWrite(SMA3, PWMValue);
  Serial.println("Command 6");
  Serial.println("NX, PY");
  delay(TimeA);
  analogWrite(SMA1, 0);
  analogWrite(SMA2, 0);
  analogWrite(SMA3, 0);
  analogWrite(SMA4, 0);
}
else{
analogWrite(SMA1, 0);
analogWrite(SMA2, 0);
analogWrite(SMA3, 0);
analogWrite(SMA4, 0);
}

}
}