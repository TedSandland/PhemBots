#include <M5Unified.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <math.h>

WiFiMulti WiFiMulti;
WiFiUDP udp;
WiFiClient trackingClient;
const int udpPort = 4210;

const char* trackingServerIp = "192.168.8.2";
const uint16_t trackingServerPort = 5000;

#pragma pack(push,1)
typedef struct {
  uint8_t marker_id;
  float x;
  float y;
  float theta;
  uint32_t sequence;
  uint8_t quality;
} pose_packet_t;
#pragma pack(pop)

IPAddress local_ip;
uint8_t MY_ID = 0; 

float my_tracked_row = -1;
float my_tracked_col = -1;
float my_tracked_heading = 0;
bool  have_tracked_pose = false;

const float ARENA_WIDTH_M   = 0.9f;
const float ARENA_HEIGHT_M  = 0.652f;
const float ARENA_ORIGIN_X  = -0.320f;
const float ARENA_ORIGIN_Y  = -0.356f;

const int HEATMAP_COLS     = 30;
const int HEATMAP_ROWS     = 30;
const int HEATMAP_SIZE     = HEATMAP_COLS * HEATMAP_ROWS;
const int MAX_ROBOTS       = 10;
const int ROBOT_STATE_SIZE = 10;
const int MAX_PAYLOAD      = 4096;

const float   OTHER_ROBOT_DISCOUNT = 0.4f;
const int     ROBOT_RADIUS         = 4;
const float   HEADING_THRESHOLD    = 0.3f;
const uint8_t FORWARD_SPEED        = 35;
const uint8_t TURN_SPEED           = 20;
const float   REWARD_WEIGHT        = 0.8f;
const float   DISTANCE_WEIGHT      = 50.0f;
const float CELL_WIDTH_MM   = (ARENA_WIDTH_M / 30.0f) * 1000.0f;
const float CELL_HEIGHT_MM  = (ARENA_HEIGHT_M / 30.0f) * 1000.0f;

int last_target_row = -1;
int last_target_col = -1;

struct RobotState {
  uint8_t id;
  uint8_t grid_row;
  uint8_t grid_col;
  float   heading;
  bool    active;
};

struct DijkstraCell {
  float cost;
  int   row;
  int   col;
};

struct Segment {
  float turn_angle;
  float drive_distance;
};

EXT_RAM_ATTR float        adjustedHeatmap[HEATMAP_SIZE];
EXT_RAM_ATTR uint8_t      reassembly_buf[MAX_PAYLOAD];

RobotState robots[MAX_ROBOTS];
uint8_t    num_robots = 0;

int path_length = 0;

int full_path_length = 0;

enum MovementState {
  STATE_IDLE,
  STATE_TURNING,
  STATE_DRIVING
};

MovementState move_state       = STATE_IDLE;

int assigned_target_row = -1;
int assigned_target_col = -1;

int     reassembly_received = 0;
int     reassembly_offset   = 0;
uint8_t reassembly_total    = 0;
bool    packet_complete     = false;
uint8_t current_seq         = 255;

void showDebug(const char* msg);

const byte dictionary[255][2] = {{181, 50}, {15, 154}, {51, 45}, {153, 70}, {84, 158}, {121, 205}, {158, 46}, {196, 242}, {254, 218}, {207, 86}, {249, 145}, {17, 167}, {14, 183}, {42, 15}, {36, 177}, {38, 62}, {70, 101}, {102, 0}, {108, 94}, {118, 175}, {134, 139}, {176, 43}, {204, 213}, {221, 130}, {254, 71}, {148, 113}, {172, 228}, {165, 84}, {33, 35}, {52, 111}, {68, 21}, {87, 178}, {158, 207}, {240, 203}, {8, 174}, {9, 41}, {24, 117}, {4, 255}, {13, 246}, {28, 90}, {23, 24}, {42, 40}, {50, 140}, {56, 178}, {36, 232}, {46, 235}, {45, 63}, {75, 100}, {80, 46}, {80, 19}, {81, 148}, {85, 104}, {93, 65}, {95, 151}, {104, 1}, {104, 103}, {97, 36}, {97, 233}, {107, 18}, {111, 229}, {103, 223}, {126, 27}, {128, 160}, {131, 68}, {139, 162}, {147, 122}, {132, 108}, {133, 42}, {133, 156}, {156, 137}, {159, 161}, {187, 124}, {188, 4}, {182, 91}, {191, 200}, {183, 171}, {202, 31}, {201, 98}, {217, 88}, {211, 213}, {204, 152}, {199, 160}, {197, 55}, {233, 93}, {249, 37}, {251, 187}, {238, 42}, {247, 77}, {53, 117}, {138, 173}, {118, 23}, {10, 207}, {6, 75}, {45, 193}, {73, 216}, {67, 244}, {79, 54}, {79, 211}, {105, 228}, {112, 199}, {122, 110}, {180, 234}, {237, 79}, {252, 231}, {254, 166}, {0, 37}, {0, 67}, {10, 136}, {10, 134}, {2, 111}, {0, 28}, {0, 151}, {8, 55}, {10, 49}, {9, 198}, {11, 1}, {9, 251}, {11, 88}, {16, 130}, {24, 45}, {16, 120}, {16, 115}, {18, 116}, {18, 177}, {26, 249}, {19, 6}, {12, 14}, {12, 241}, {4, 51}, {12, 159}, {14, 242}, {14, 253}, {7, 76}, {15, 164}, {7, 47}, {5, 181}, {15, 145}, {7, 219}, {30, 228}, {20, 57}, {29, 128}, {21, 200}, {31, 139}, {21, 186}, {29, 177}, {32, 128}, {40, 233}, {34, 162}, {40, 83}, {42, 240}, {34, 247}, {41, 64}, {33, 70}, {41, 185}, {43, 156}, {43, 178}, {56, 202}, {56, 46}, {48, 7}, {56, 231}, {58, 73}, {58, 101}, {50, 93}, {59, 136}, {57, 29}, {59, 211}, {38, 71}, {39, 128}, {47, 170}, {45, 20}, {37, 222}, {37, 83}, {47, 119}, {52, 72}, {60, 168}, {60, 65}, {52, 13}, {52, 251}, {54, 154}, {61, 224}, {53, 106}, {61, 9}, {61, 237}, {63, 196}, {63, 108}, {55, 206}, {61, 92}, {61, 118}, {55, 176}, {63, 23}, {63, 255}, {72, 229}, {66, 104}, {74, 45}, {65, 96}, {73, 81}, {65, 221}, {75, 223}, {88, 79}, {90, 72}, {88, 22}, {80, 93}, {90, 250}, {90, 181}, {81, 35}, {91, 138}, {89, 25}, {81, 53}, {76, 105}, {70, 193}, {78, 11}, {68, 95}, {78, 89}, {77, 131}, {77, 125}, {71, 216}, {71, 115}, {92, 133}, {94, 68}, {86, 43}, {92, 187}, {85, 195}, {95, 110}, {95, 235}, {93, 18}, {85, 94}, {98, 112}, {98, 21}, {97, 194}, {107, 32}, {99, 69}, {107, 92}, {107, 91}, {120, 12}, {122, 207}, {120, 127}, {121, 128}, {113, 229}, {113, 116}, {121, 182}, {113, 211}, {123, 51}, {100, 106}, {102, 168}, {110, 167}, {110, 145}, {101, 34}, {109, 203}, {103, 141}, {109, 49}, {126, 128}, {126, 226}, {126, 141}, {116, 210}, {124, 50}};

void drawAruCo(int index) {
  int x_offset = (320 - 240) / 2;
  int grid_w = 240 / 4;

  M5.Display.fillScreen(BLACK);

  int count = 15;
  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 4; x++) {
      if (count < 8) {
        if ((dictionary[index][1] >> count) & 0x01) {
          M5.Display.fillRect(x_offset + (x * grid_w), y * grid_w, grid_w, grid_w, WHITE);
        }
      } else {
        if ((dictionary[index][0] >> (count - 8)) & 0x01) {
          M5.Display.fillRect(x_offset + (x * grid_w), y * grid_w, grid_w, grid_w, WHITE);
        }
      }
      count--;
    }
  }
}


bool reassemblePacket(uint8_t* buf, int len) {
  // Serial.print("packet length:");
  // Serial.println(len);
  if (len < 5) return false;

  uint8_t seq_num       = buf[0];
  uint8_t packet_num    = buf[1];
  uint8_t total_packets = buf[2];
  uint8_t is_last       = buf[3];
  uint8_t* data         = buf + 4;
  int     data_len      = len - 4;

  if (packet_num == 0) {
    current_seq         = seq_num;
    reassembly_offset   = 0;
    reassembly_received = 0;
    reassembly_total    = total_packets;
    packet_complete     = false;
  }

  if (seq_num != current_seq) return false;

  if (reassembly_offset + data_len <= MAX_PAYLOAD) {
    memcpy(reassembly_buf + reassembly_offset, data, data_len);
    reassembly_offset += data_len;
    reassembly_received++;
  }

  if (is_last) {
    packet_complete = true;
    return true;
  }

  return false;
}

void parsePacket(uint8_t* buf, int len) {
  if (len < HEATMAP_SIZE) return;
  memcpy(adjustedHeatmap, buf, HEATMAP_SIZE * sizeof(float));
  Serial.print("buf[0]:");
  Serial.print(buf[0]);
  Serial.print(" buf[100]:");
  Serial.print(buf[100]);
  Serial.print(" buf[465]:");
  Serial.print(buf[465]);
  Serial.print(" buf[899]:");
  Serial.print(buf[899]);
  Serial.print(" len:");
  Serial.println(len);
  int heatmap_bytes = HEATMAP_SIZE * sizeof(float);
  if (len <= heatmap_bytes) return;
  num_robots = buf[heatmap_bytes];
  if (num_robots > MAX_ROBOTS) num_robots = MAX_ROBOTS;
  for (int i = 0; i < MAX_ROBOTS; i++) robots[i].active = false;
  int offset = heatmap_bytes + 1;
  for (int i = 0; i < num_robots; i++) {
    if (offset + ROBOT_STATE_SIZE > len) break;
    robots[i].id       = buf[offset];
    robots[i].grid_row = buf[offset + 1];
    robots[i].grid_col = buf[offset + 2];
    memcpy(&robots[i].heading, buf + offset + 4, sizeof(float));
    robots[i].active   = true;

    if (robots[i].id == MY_ID) {
        if (buf[offset + 8] != assigned_target_row || buf[offset + 9] != assigned_target_col) {
        }
        assigned_target_row = buf[offset + 8];
        assigned_target_col = buf[offset + 9];
      }

    offset += ROBOT_STATE_SIZE;
  }
}

#define ROBOT_I2C_ADDR 0x20

void sendMotorCommand(uint8_t cmd, uint8_t value = 100) {
  Wire.beginTransmission(ROBOT_I2C_ADDR);
  Wire.write(cmd);
  Wire.write(value);
  Wire.endTransmission();
  delay(5);
}

void sendTurnCommand(float angle_rad) {
  Wire.beginTransmission(ROBOT_I2C_ADDR);
  Wire.write(0x05);
  byte angle_bytes[4];
  memcpy(angle_bytes, &angle_rad, sizeof(float));
  Wire.write(angle_bytes, 4);
  Wire.endTransmission();
  delay(5);
}

void sendDriveCommand(float distance_mm) {
  Wire.beginTransmission(ROBOT_I2C_ADDR);
  Wire.write(0x06);
  byte dist_bytes[4];
  memcpy(dist_bytes, &distance_mm, sizeof(float));
  Wire.write(dist_bytes, 4);
  Wire.endTransmission();
  delay(5);
}

bool robotComplete() {
  Wire.requestFrom(ROBOT_I2C_ADDR, 1);
  if (Wire.available()) {
    return Wire.read() == 0x01;
  }
  return false;
}

void showDebug(const char* msg) {
  M5.Display.setTextSize(1);
  M5.Display.setTextColor(TFT_BLACK, TFT_WHITE);
  M5.Display.setCursor(0, 0);
  M5.Display.print(msg);
}

void showBattery() {
  int battery = M5.Power.getBatteryLevel();
  char bat[16];
  sprintf(bat, "%d%%", battery);
  M5.Display.setTextSize(2);
  M5.Display.setTextColor(TFT_GREEN, TFT_BLACK);
  M5.Display.setCursor(M5.Display.width() - 50, 0);
  M5.Display.print(bat);

  if (M5.Power.isCharging()) {
    M5.Display.fillCircle(M5.Display.width() - 55, 8, 5, TFT_GREEN);
  }
}

float angleDiff(float target, float current) {
  float diff = target - current;
  while (diff > M_PI)  diff -= 2 * M_PI;
  while (diff < -M_PI) diff += 2 * M_PI;
  return diff;
}

float getCellReward(int row, int col, float* heatmap) {
  if (row < 0 || row >= HEATMAP_ROWS || col < 0 || col >= HEATMAP_COLS) return 0;
  return heatmap[row * HEATMAP_COLS + col];
}

float directionToHeading(int dr, int dc) {
  if (dr < 0 && dc == 0)  return -M_PI;
  if (dr > 0 && dc == 0)  return 0.0f;
  if (dr == 0 && dc > 0)  return -M_PI / 2;
  if (dr == 0 && dc < 0)  return M_PI / 2;
  if (dr < 0 && dc > 0)   return -M_PI * 3 / 4;
  if (dr < 0 && dc < 0)   return M_PI * 3 / 4;
  if (dr > 0 && dc > 0)   return -M_PI / 4;
  if (dr > 0 && dc < 0)   return M_PI / 4;
  return 0.0f;
}

float cellDistance(int dr, int dc) {
  if (dr != 0 && dc != 0)
    return sqrt(CELL_WIDTH_MM * CELL_WIDTH_MM + CELL_HEIGHT_MM * CELL_HEIGHT_MM);
  if (dc != 0) return CELL_WIDTH_MM;
  return CELL_HEIGHT_MM;
}

void connectTracking() {
  trackingClient.setTimeout(1);
  int err;
  do {
    err = trackingClient.connect(trackingServerIp, trackingServerPort);
    delay(250);
  } while (err <= 0);
}

void updateTrackingPose() {
  if (!trackingClient.connected()) {
    trackingClient.stop();
    delay(100);
    connectTracking();
    return;
  }

  while (trackingClient.available() >= sizeof(pose_packet_t)) {
    pose_packet_t pose;
    memset(&pose, 0, sizeof(pose_packet_t));
    int bytes = trackingClient.readBytes((char*)&pose, sizeof(pose_packet_t));
    if (bytes == sizeof(pose_packet_t) && pose.marker_id == MY_ID && pose.quality > 0) {
      // uint8_t raw_quality = ((uint8_t*)&pose)[17];
      // char dbg[32];
      // sprintf(dbg, "id:%d q:%d rq:%d", pose.marker_id, pose.quality, raw_quality);
      // showDebug(dbg);

      float grid_x = (pose.x - ARENA_ORIGIN_X) / ARENA_WIDTH_M * HEATMAP_COLS;
      float grid_y = (pose.y - ARENA_ORIGIN_Y) / ARENA_HEIGHT_M * HEATMAP_ROWS;

      my_tracked_col = constrain(grid_x, 0, HEATMAP_COLS - 1);
      my_tracked_row = constrain(grid_y, 0, HEATMAP_ROWS - 1);
      my_tracked_heading = pose.theta;
      have_tracked_pose = true;
    }
  }
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  M5.Power.begin();
  Wire.begin();
  Serial.begin(115200);

  M5.Display.fillScreen(BLACK);
  M5.Display.setTextColor(WHITE);
  M5.Display.setCursor(0, 0);
  M5.Display.println("Connecting WiFi...");

  WiFiMulti.addAP("YOUR_WIFI_NAME", "YOUR_WIFI_PASSWORD");
  while (WiFiMulti.run() != WL_CONNECTED) {

    delay(500);
  }
  delay(500);
  local_ip = WiFi.localIP();
  MY_ID = local_ip[3];
  M5.Display.fillScreen(BLACK);
  M5.Display.setTextColor(WHITE);
  M5.Display.setTextSize(2);
  M5.Display.setCursor(0, 40);
  M5.Display.print("IP: ");
  M5.Display.println(local_ip);
  M5.Display.print("ID: ");
  M5.Display.println(MY_ID);
  M5.Display.print("Struct size: ");
  M5.Display.println(sizeof(pose_packet_t));
  delay(5000);
  M5.Display.setBrightness(128);
  

  if (MY_ID > 0 && MY_ID < 200) {
    drawAruCo(MY_ID);
  }

  udp.begin(udpPort);
  connectTracking();
}

void loop() {
  showBattery();
  updateTrackingPose();

  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    uint8_t buf[2048];
    int len = udp.read(buf, sizeof(buf));
    if (len > 0) {
      bool complete = reassemblePacket(buf, len);

      if (complete) {
        packet_complete = false;
        parsePacket(reassembly_buf, reassembly_offset);

        // if (!have_tracked_pose) {
        //   sendMotorCommand(0x00);
        //   showDebug("No fix");
        //   move_state = STATE_IDLE;
        //   return;
        // }

        int   my_row     = (int)my_tracked_row;
        int   my_col     = (int)my_tracked_col;
        float my_heading = my_tracked_heading;

        switch (move_state) {

          case STATE_IDLE: {
            if (assigned_target_row < 0 || assigned_target_col < 0) break;

            float dr = (float)(assigned_target_row - my_row);
            float dc = (float)(assigned_target_col - my_col);
            float distance_mm = sqrt((dr*CELL_HEIGHT_MM)*(dr*CELL_HEIGHT_MM) + (dc*CELL_WIDTH_MM)*(dc*CELL_WIDTH_MM));
            float target_heading = atan2(-dc, dr);

            while (target_heading > M_PI)  target_heading -= 2 * M_PI;
            while (target_heading < -M_PI) target_heading += 2 * M_PI;

            float error = angleDiff(target_heading, my_heading);

           if (abs(error) > HEADING_THRESHOLD) {
              sendTurnCommand(-error);
              move_state = STATE_TURNING;
            } else {
              sendDriveCommand(distance_mm);
              move_state = STATE_DRIVING;
            }
            break;
          }

          case STATE_TURNING: {
            if (robotComplete()) {
              float dr = (float)(assigned_target_row - my_row);
              float dc = (float)(assigned_target_col - my_col);
              float distance_mm = sqrt((dr*CELL_HEIGHT_MM)*(dr*CELL_HEIGHT_MM) + (dc*CELL_WIDTH_MM)*(dc*CELL_WIDTH_MM));
              sendDriveCommand(distance_mm);
              move_state = STATE_DRIVING;
            }
            break;
          }

          case STATE_DRIVING: {
            if (robotComplete()) {
              move_state = STATE_IDLE;
            }
            break;
          }
        }
      }
    }
  }
}
