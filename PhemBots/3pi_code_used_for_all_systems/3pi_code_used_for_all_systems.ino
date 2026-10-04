#include <Wire.h>
#include <Pololu3piPlus32U4.h>

using namespace Pololu3piPlus32U4;

#define ROBOT_I2C_ADDR 0x20

Motors motors;
Encoders encoders;

const float WHEELBASE_MM        = 85.0f;
const float WHEEL_DIAMETER_MM   = 32.0f;
const float COUNTS_PER_REV      = 358.3f;
const float WHEEL_CIRCUMFERENCE = PI * WHEEL_DIAMETER_MM;
const float COUNTS_PER_MM       = COUNTS_PER_REV / WHEEL_CIRCUMFERENCE;

const int FORWARD_SPEED = 60;
const int TURN_SPEED    = 60;

enum RobotState { IDLE, TURNING, DRIVING };
RobotState state = IDLE;

int32_t  turn_counts_target  = 0;
int32_t  drive_counts_target = 0;
bool     turn_direction      = true;

unsigned long lastCommandTime      = 0;
const unsigned long COMMAND_TIMEOUT_MS = 1000;

volatile bool    new_command    = false;
volatile uint8_t cmd_buf[5];
volatile int     cmd_len        = 0;

int32_t angleToEncoderCounts(float angle_rad) {
  float wheel_travel_mm = (abs(angle_rad) * WHEELBASE_MM) / 2.0f;
  return (int32_t)(wheel_travel_mm * COUNTS_PER_MM);
}

void receiveCommand(int numBytes) {
  cmd_len = 0;
  while (Wire.available() && cmd_len < 5) {
    cmd_buf[cmd_len++] = Wire.read();
  }
  while (Wire.available()) Wire.read();
  new_command = true;
}

void processCommand() {
  if (cmd_len < 1) return;

  byte cmd = cmd_buf[0];
  lastCommandTime = millis();

  if (cmd == 0x00) {
    motors.setSpeeds(0, 0);
    state = IDLE;
    return;
  }

  if (state == TURNING && cmd == 0x05) return;

  switch(cmd) {
    case 0x01:
      motors.setSpeeds(FORWARD_SPEED, FORWARD_SPEED);
      state = DRIVING;
      break;

    case 0x02:
      motors.setSpeeds(-FORWARD_SPEED, -FORWARD_SPEED);
      state = DRIVING;
      break;

    case 0x05:
      if (cmd_len >= 5) {
        float angle;
        memcpy(&angle, (const void*)&cmd_buf[1], sizeof(float));
        turn_counts_target = angleToEncoderCounts(angle);
        turn_direction = angle > 0;
        encoders.getCountsAndResetLeft();
        encoders.getCountsAndResetRight();
        if (turn_direction) {
          motors.setSpeeds(-TURN_SPEED, TURN_SPEED);
        } else {
          motors.setSpeeds(TURN_SPEED, -TURN_SPEED);
        }
        state = TURNING;
      }
      break;

    case 0x06:
      if (cmd_len >= 5) {
        float distance_mm;
        memcpy(&distance_mm, (const void*)&cmd_buf[1], sizeof(float));
        drive_counts_target = (int32_t)(distance_mm * COUNTS_PER_MM);
        encoders.getCountsAndResetLeft();
        encoders.getCountsAndResetRight();
        motors.setSpeeds(FORWARD_SPEED, FORWARD_SPEED);
        state = DRIVING;
      }
      break;

    default:
      motors.setSpeeds(0, 0);
      state = IDLE;
      break;
  }
}

void requestEvent() {
  if (state == IDLE) {
    Wire.write(0x01);
  } else {
    Wire.write(0x00);
  }
}

void setup() {
  Wire.begin(ROBOT_I2C_ADDR);
  Wire.onReceive(receiveCommand);
  Wire.onRequest(requestEvent);
  lastCommandTime = millis();
}

void loop() {
  if (new_command) {
    new_command = false;
    processCommand();
  }

  if (state == TURNING) {
    int32_t avg = (abs(encoders.getCountsLeft()) + abs(encoders.getCountsRight())) / 2;
    if (avg >= turn_counts_target) {
      motors.setSpeeds(0, 0);
      state = IDLE;
    }
  }

  if (state == DRIVING) {
    int32_t avg = (abs(encoders.getCountsLeft()) + abs(encoders.getCountsRight())) / 2;
    if (avg >= drive_counts_target) {
      motors.setSpeeds(0, 0);
      state = IDLE;
    }
  }

  if (millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    motors.setSpeeds(0, 0);
    state = IDLE;
  }
}