#include <Stepper.h>
const int stepsPerRevolution = 2048;
int wavelength;
float angle;
int diffraction_grating = 1000000; // in Lines per m
bool running = false;
Stepper Stepper1 = Stepper(stepsPerRevolution, 8, 9, 10, 11);
Stepper Stepper2 = Stepper(stepsPerRevolution, 7, 6, 5, 4);
void setup() {
  pinMode(13, OUTPUT);
  Stepper1.setSpeed(5);
  Stepper2.setSpeed(5);
  wavelength = 510; //in nm
  angle = asin(wavelength*diffraction_grating*10^(-9))*(360/(2*PI));
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) 
    {
        switch(Serial.read())
        {
        case '0':
            running = true;
            Serial.print('\n');
            Serial.print("Reset diffraction grating");
            digitalWrite(13, LOW);
            Stepper2.step((stepsPerRevolution/360)*angle);
            Stepper1.step(-(stepsPerRevolution/360)*angle);
            break;
        case '1':
            running = false;
            Serial.print('\n');
            Serial.print("Diffraction grating angled");
            digitalWrite(13, HIGH);
            Stepper1.step((stepsPerRevolution/360)*angle);
            Stepper2.step(-(stepsPerRevolution/360)*angle);
            break;
        }
    }
}
