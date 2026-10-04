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
const int ROBOT_STATE_SIZE = 8;
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
const int     MAX_PATH_STEPS       = 4;
const int     MAX_SEGMENTS         = 8;
const unsigned long DRIVE_TIMEOUT_MS = 5000;

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
EXT_RAM_ATTR float        dijkstra_dist[HEATMAP_SIZE];
EXT_RAM_ATTR bool         dijkstra_visited[HEATMAP_SIZE];
EXT_RAM_ATTR int          dijkstra_prev_row[HEATMAP_SIZE];
EXT_RAM_ATTR int          dijkstra_prev_col[HEATMAP_SIZE];
EXT_RAM_ATTR DijkstraCell dijkstra_queue[HEATMAP_SIZE];
EXT_RAM_ATTR uint8_t      reassembly_buf[MAX_PAYLOAD];

RobotState robots[MAX_ROBOTS];
uint8_t    num_robots = 0;

int path_rows[MAX_PATH_STEPS];
int path_cols[MAX_PATH_STEPS];
int path_length = 0;

EXT_RAM_ATTR int full_path_rows[676];
EXT_RAM_ATTR int full_path_cols[676];
int full_path_length = 0;

Segment segments[MAX_SEGMENTS];
int     segment_count   = 0;
int     current_segment = 0;

enum MovementState {
  STATE_IDLE,
  STATE_TURNING,
  STATE_DRIVING
};

MovementState move_state       = STATE_IDLE;
unsigned long drive_started_ms = 0;

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
  // Serial.print("pkt length:");
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

void smoothHeatmap(float* heatmap) {
  static float temp[HEATMAP_SIZE];
  int border = 2;
  
  for (int r = 0; r < HEATMAP_ROWS; r++) {
    for (int c = 0; c < HEATMAP_COLS; c++) {
      if (r < border || r >= HEATMAP_ROWS - border ||
          c < border || c >= HEATMAP_COLS - border) {
        temp[r * HEATMAP_COLS + c] = heatmap[r * HEATMAP_COLS + c];
        continue;
      }

      float sum = 0;
      int count = 0;
      for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
          int nr = r + dr;
          int nc = c + dc;
          if (nr >= border && nr < HEATMAP_ROWS - border &&
              nc >= border && nc < HEATMAP_COLS - border) {
            sum += heatmap[nr * HEATMAP_COLS + nc];
            count++;
          }
        }
      }
      temp[r * HEATMAP_COLS + c] = count > 0 ? sum / count : heatmap[r * HEATMAP_COLS + c];
    }
  }
  memcpy(heatmap, temp, HEATMAP_SIZE * sizeof(float));
}

void buildAdjustedHeatmap(float* heatmap, uint8_t myId, int* my_path_rows, int* my_path_cols, int my_path_length) {

  for (int r = 0; r < MAX_ROBOTS; r++) {
    if (!robots[r].active) continue;
    if (robots[r].id == myId) continue;

    int sim_row = robots[r].grid_row;
    int sim_col = robots[r].grid_col;

    int sim_path_rows[MAX_PATH_STEPS];
    int sim_path_cols[MAX_PATH_STEPS];
    int sim_path_length = 0;

    dijkstra_simulate(sim_row, sim_col, heatmap,
                      sim_path_rows, sim_path_cols, sim_path_length);

    bool discounted[HEATMAP_SIZE] = {false};

    for (int step = 0; step < sim_path_length; step++) {
      int pr = sim_path_rows[step];
      int pc = sim_path_cols[step];

      int my_step = -1;
      for (int s = 0; s < my_path_length; s++) {
        if (my_path_rows[s] == pr && my_path_cols[s] == pc) {
          my_step = s;
          break;
        }
      }

      if (my_step == -1 || step < my_step) {
        for (int dr = -ROBOT_RADIUS; dr <= ROBOT_RADIUS; dr++) {
          for (int dc = -ROBOT_RADIUS; dc <= ROBOT_RADIUS; dc++) {
            int nr = pr + dr;
            int nc = pc + dc;
            if (nr >= 0 && nr < HEATMAP_ROWS && nc >= 0 && nc < HEATMAP_COLS) {
              int idx = nr * HEATMAP_COLS + nc;
              if (!discounted[idx]) {
                heatmap[idx] *= OTHER_ROBOT_DISCOUNT;
                discounted[idx] = true;
              }
            }
          }
        }
      }
    }
  }
  float density_bonus[HEATMAP_SIZE] = {0};
  for (int r = 1; r < HEATMAP_ROWS - 1; r++) {
    for (int c = 1; c < HEATMAP_COLS - 1; c++) {
      float cell_val = heatmap[r * HEATMAP_COLS + c];
      if (cell_val <= 600.0f) continue;

      int mess_neighbours = 0;
      for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
          if (dr == 0 && dc == 0) continue;
          if (heatmap[(r+dr) * HEATMAP_COLS + (c+dc)] > 600.0f) mess_neighbours++;
        }
      }
      density_bonus[r * HEATMAP_COLS + c] = cell_val * (mess_neighbours / 8.0f) * 0.5f;
    }
  }

  for (int i = 0; i < HEATMAP_SIZE; i++) {
    heatmap[i] += density_bonus[i];
  }
}

void dijkstra_simulate(int start_row, int start_col, float* heatmap, 
                       int* sim_path_rows, int* sim_path_cols, int& sim_path_length) {
  for (int i = 0; i < HEATMAP_SIZE; i++) {
    dijkstra_dist[i]     = 1e9f;
    dijkstra_visited[i]  = false;
    dijkstra_prev_row[i] = -1;
    dijkstra_prev_col[i] = -1;
  }

  int start_idx = start_row * HEATMAP_COLS + start_col;
  dijkstra_dist[start_idx] = 0;

  int queue_size = 0;
  dijkstra_queue[queue_size++] = {0.0f, start_row, start_col};

  int dr[] = {-1, 1, 0, 0, -1, -1, 1, 1};
  int dc[] = {0, 0, -1, 1, -1, 1, -1, 1};
  float step_cost[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.414f, 1.414f, 1.414f, 1.414f};

  float best_reward = -1;
  int   best_row    = start_row;
  int   best_col    = start_col;

  while (queue_size > 0) {
    int min_idx = 0;
    for (int i = 1; i < queue_size; i++) {
      if (dijkstra_queue[i].cost < dijkstra_queue[min_idx].cost) min_idx = i;
    }

    DijkstraCell current = dijkstra_queue[min_idx];
    dijkstra_queue[min_idx] = dijkstra_queue[--queue_size];

    int curr_idx = current.row * HEATMAP_COLS + current.col;
    if (dijkstra_visited[curr_idx]) continue;
    dijkstra_visited[curr_idx] = true;

    float cell_reward = getCellReward(current.row, current.col, heatmap);
    if (cell_reward > best_reward) {
      best_reward = cell_reward;
      best_row    = current.row;
      best_col    = current.col;
    }

    for (int d = 0; d < 8; d++) {
      int nr = current.row + dr[d];
      int nc = current.col + dc[d];

      int next_idx = nr * HEATMAP_COLS + nc;
      if (dijkstra_visited[next_idx]) continue;

      float reward    = getCellReward(nr, nc, heatmap);
      float move_cost = step_cost[d] * DISTANCE_WEIGHT / (1.0f + reward * REWARD_WEIGHT);
      float new_cost  = dijkstra_dist[curr_idx] + move_cost;

      if (new_cost < dijkstra_dist[next_idx]) {
        dijkstra_dist[next_idx]     = new_cost;
        dijkstra_prev_row[next_idx] = current.row;
        dijkstra_prev_col[next_idx] = current.col;

        if (queue_size < HEATMAP_SIZE) {
          dijkstra_queue[queue_size++] = {new_cost, nr, nc};
        }
      }
    }
  }

  int raw_rows[676];
  int raw_cols[676];
  int raw_len = 0;

  int curr_r = best_row;
  int curr_c = best_col;

  while (raw_len < 676) {
    raw_rows[raw_len] = curr_r;
    raw_cols[raw_len] = curr_c;
    raw_len++;

    int idx = curr_r * HEATMAP_COLS + curr_c;
    int pr  = dijkstra_prev_row[idx];
    int pc  = dijkstra_prev_col[idx];

    if (pr == -1 || (pr == start_row && pc == start_col)) break;

    curr_r = pr;
    curr_c = pc;
  }

  sim_path_length = 0;
  int path_start = max(0, raw_len - MAX_PATH_STEPS);
  for (int i = raw_len - 1; i >= path_start; i--) {
    sim_path_rows[sim_path_length] = raw_rows[i];
    sim_path_cols[sim_path_length] = raw_cols[i];
    sim_path_length++;
  }
}

void dijkstra(int start_row, int start_col, float* heatmap) {
  for (int i = 0; i < HEATMAP_SIZE; i++) {
    dijkstra_dist[i]     = 1e9f;
    dijkstra_visited[i]  = false;
    dijkstra_prev_row[i] = -1;
    dijkstra_prev_col[i] = -1;
  }

  int start_idx = start_row * HEATMAP_COLS + start_col;
  dijkstra_dist[start_idx] = 0;

  int queue_size = 0;
  dijkstra_queue[queue_size++] = {0.0f, start_row, start_col};

  int dr[] = {-1, 1, 0, 0, -1, -1, 1, 1};
  int dc[] = {0, 0, -1, 1, -1, 1, -1, 1};
  float step_cost[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.414f, 1.414f, 1.414f, 1.414f};

  float best_reward = -1;
  int   best_row    = start_row;
  int   best_col    = start_col;

  while (queue_size > 0) {
    int min_idx = 0;
    for (int i = 1; i < queue_size; i++) {
      if (dijkstra_queue[i].cost < dijkstra_queue[min_idx].cost) min_idx = i;
    }

    DijkstraCell current = dijkstra_queue[min_idx];
    dijkstra_queue[min_idx] = dijkstra_queue[--queue_size];

    int curr_idx = current.row * HEATMAP_COLS + current.col;
    if (dijkstra_visited[curr_idx]) continue;
    dijkstra_visited[curr_idx] = true;

    float cell_reward = getCellReward(current.row, current.col, heatmap);
    if (cell_reward > best_reward) {
      best_reward = cell_reward;
      best_row    = current.row;
      best_col    = current.col;
    }

    for (int d = 0; d < 8; d++) {
      int nr = current.row + dr[d];
      int nc = current.col + dc[d];

      int next_idx = nr * HEATMAP_COLS + nc;
      if (dijkstra_visited[next_idx]) continue;

      float reward    = getCellReward(nr, nc, heatmap);
      float move_cost = step_cost[d] * DISTANCE_WEIGHT / (1.0f + reward * REWARD_WEIGHT);
      float new_cost  = dijkstra_dist[curr_idx] + move_cost;

      if (new_cost < dijkstra_dist[next_idx]) {
        dijkstra_dist[next_idx]     = new_cost;
        dijkstra_prev_row[next_idx] = current.row;
        dijkstra_prev_col[next_idx] = current.col;

        if (queue_size < HEATMAP_SIZE) {
          dijkstra_queue[queue_size++] = {new_cost, nr, nc};
        }
      }
    }
  }

  int raw_rows[676];
  int raw_cols[676];
  int raw_len = 0;

  int curr_r = best_row;
  int curr_c = best_col;

  while (raw_len < 676) {
    raw_rows[raw_len] = curr_r;
    raw_cols[raw_len] = curr_c;
    raw_len++;

    int idx = curr_r * HEATMAP_COLS + curr_c;
    int pr  = dijkstra_prev_row[idx];
    int pc  = dijkstra_prev_col[idx];

    if (pr == -1 || (pr == start_row && pc == start_col)) break;

    curr_r = pr;
    curr_c = pc;
  }

  full_path_length = 0;
  for (int i = raw_len - 1; i >= 0; i--) {
    full_path_rows[full_path_length] = raw_rows[i];
    full_path_cols[full_path_length] = raw_cols[i];
    full_path_length++;
  }

  path_length = min(full_path_length, MAX_PATH_STEPS);
  for (int i = 0; i < path_length; i++) {
    path_rows[i] = full_path_rows[i];
    path_cols[i] = full_path_cols[i];
  }
}

void buildSegments(int start_row, int start_col, float current_heading) {
  segment_count   = 0;
  current_segment = 0;

  if (path_length == 0) return;

  int   curr_r      = start_row;
  int   curr_c      = start_col;
  float curr_heading = current_heading;

  int seg_dr   = path_rows[0] - start_row;
  int seg_dc   = path_cols[0] - start_col;
  float seg_dist = 0;

  for (int i = 0; i < path_length && segment_count < MAX_SEGMENTS - 1; i++) {
    int next_r = path_rows[i];
    int next_c = path_cols[i];
    int step_dr = next_r - curr_r;
    int step_dc = next_c - curr_c;

    if (step_dr == seg_dr && step_dc == seg_dc) {
      seg_dist += cellDistance(step_dr, step_dc);
    } else {
      float target_heading = directionToHeading(seg_dr, seg_dc);
      float turn_angle     = angleDiff(target_heading, curr_heading);
      float max_dist = seg_dist;
      if (seg_dr < 0 && curr_r < 3)
        max_dist = min(max_dist, (float)curr_r * CELL_HEIGHT_MM);
      if (seg_dr > 0 && curr_r > HEATMAP_ROWS - 4)
        max_dist = min(max_dist, (float)(HEATMAP_ROWS - 1 - curr_r) * CELL_HEIGHT_MM);
      if (seg_dc < 0 && curr_c < 3)
        max_dist = min(max_dist, (float)curr_c * CELL_WIDTH_MM);
      if (seg_dc > 0 && curr_c > HEATMAP_COLS - 4)
        max_dist = min(max_dist, (float)(HEATMAP_COLS - 1 - curr_c) * CELL_WIDTH_MM);
      max_dist = max(max_dist, CELL_HEIGHT_MM);
      segments[segment_count++] = {-turn_angle, max_dist};
      curr_heading = target_heading;

      seg_dr   = step_dr;
      seg_dc   = step_dc;
      seg_dist = cellDistance(step_dr, step_dc);
    }

    curr_r = next_r;
    curr_c = next_c;
  }

  if (seg_dist > 0 && segment_count < MAX_SEGMENTS) {
    float target_heading = directionToHeading(seg_dr, seg_dc);
    float turn_angle     = angleDiff(target_heading, curr_heading);
    float max_dist = seg_dist;
    if (seg_dr < 0 && curr_r < 3)
      max_dist = min(max_dist, (float)curr_r * CELL_HEIGHT_MM);
    if (seg_dr > 0 && curr_r > HEATMAP_ROWS - 4)
      max_dist = min(max_dist, (float)(HEATMAP_ROWS - 1 - curr_r) * CELL_HEIGHT_MM);
    if (seg_dc < 0 && curr_c < 3)
      max_dist = min(max_dist, (float)curr_c * CELL_WIDTH_MM);
    if (seg_dc > 0 && curr_c > HEATMAP_COLS - 4)
      max_dist = min(max_dist, (float)(HEATMAP_COLS - 1 - curr_c) * CELL_WIDTH_MM);
    max_dist = max(max_dist, CELL_HEIGHT_MM);
    segments[segment_count++] = {-turn_angle, max_dist};
  }
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
            dijkstra(my_row, my_col, adjustedHeatmap);

            buildAdjustedHeatmap(adjustedHeatmap, MY_ID, full_path_rows, full_path_cols, full_path_length);

            dijkstra(my_row, my_col, adjustedHeatmap);

            buildSegments(my_row, my_col, my_heading);

            if (segment_count > 0) {
              float turn = segments[0].turn_angle;
              if (abs(turn) > HEADING_THRESHOLD) {
                sendTurnCommand(turn);
                move_state = STATE_TURNING;
              } else {
                sendDriveCommand(segments[0].drive_distance);
                drive_started_ms = millis();
                move_state = STATE_DRIVING;
              }
            }
            break;
          }

          case STATE_TURNING: {
            if (robotComplete()) {
              sendDriveCommand(segments[current_segment].drive_distance);
              drive_started_ms = millis();
              move_state = STATE_DRIVING;
            }
            break;
          }

          case STATE_DRIVING: {
            bool done = robotComplete();
            bool timeout = millis() - drive_started_ms > DRIVE_TIMEOUT_MS;

            showBattery();

            if (done || timeout) {
              current_segment++;

              if (current_segment >= segment_count) {
                current_segment = 0;
                move_state = STATE_IDLE;
              } else {
                float turn = segments[current_segment].turn_angle;
                if (abs(turn) > HEADING_THRESHOLD) {
                  sendTurnCommand(turn);
                  move_state = STATE_TURNING;
                } else {
                  sendDriveCommand(segments[current_segment].drive_distance);
                  drive_started_ms = millis();
                }
              }
            }
            break;
          }
        }
      }
    }
  }
}
