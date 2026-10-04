import cv2
import numpy as np
import time
import socket
import struct
import threading
import csv
import os
from datetime import datetime

GRID_COLS = 30
GRID_ROWS = 30

RED_VALUE   = 1500.0
GREEN_VALUE = 700.0
BLUE_VALUE  = 80.0

ACCUMULATE_RATE = 6.25
MESS_WEIGHT       = 3.0
ACCUMULATE_WEIGHT = 1.0
ACCUMULATE_CAP = 500.0

HEATMAP_UPDATE_INTERVAL = 1

UDP_PORT = 4210
MAX_PACKET_SIZE = 1400
ROBOT_IPS = ["192.168.8.181", "192.168.8.94", "192.168.8.65"]

EDGE_PENALTY = 0.0

RESULTS_FOLDER = r"./results"
TRIAL_DURATION_S = 180
os.makedirs(RESULTS_FOLDER, exist_ok=True)

recording = False
auto_mess = False
system_active = False
trial_start_time = None
trial_number = 0
trial_system_name = ""
trial_writer = None
trial_file = None
cell_visit_counts = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
active_messes = []
snapshot_interval = 1.0
last_snapshot_time = 0.0
auto_mess_random = False
random_mess_events = []
total_reward_collected = 0.0
robot_reward_collected = {}
record_video = False
video_frames = []
trial_filename = ""

# Automatic mess events for M mode: (seconds, row, col, mess)
MESS_EVENTS = [
    (30, 15, 15, RED_VALUE),
    (60, 5, 5, BLUE_VALUE),
    (90, 10, 20, GREEN_VALUE),
    (90, 22, 8, BLUE_VALUE),
    (120, 22, 8, RED_VALUE),
    (150, 8, 25, GREEN_VALUE),
]

robot_waypoint_index = {}
robot_strip_assignment = {}

TRACKING_SERVER_IP = "192.168.8.2"
TRACKING_SERVER_PORT = 8000

ARENA_WIDTH_M  = 0.9
ARENA_HEIGHT_M = 0.652
ARENA_ORIGIN_X = -0.320
ARENA_ORIGIN_Y = -0.356

ROBOT_TIMEOUT = 1.0
previous_robot_positions = {}
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

accumulate_layer = np.ones((GRID_ROWS, GRID_COLS), dtype=np.float32)
mess_layer = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

last_time = time.time()
frame_count = 0

robot_positions = {}
robot_last_seen = {}
tracking_lock = threading.Lock()
tracking_positions = {}

STRIP_WAYPOINTS = {
    0: [(4,4), (25,4), (25,6), (4,6), (4,8), (25,8), (25,10), (4,10)],
    1: [(4,13), (25,13), (25,15), (4,15), (4,17), (25,17), (25,19), (4,19)],
    2: [(4,22), (25,22), (25,24), (4,24), (4,26), (25,26)]
}

def generate_random_mess_events(max_messes=8, trial_duration=180):
    import random
    events = []
    row_min, row_max = 4, GRID_ROWS - 5
    col_min, col_max = 4, GRID_COLS - 5
    colours = [RED_VALUE, GREEN_VALUE, BLUE_VALUE]
    times = sorted(random.sample(range(15, trial_duration - 10), max_messes))
    for t in times:
        row = random.randint(row_min, row_max)
        col = random.randint(col_min, col_max)
        colour = random.choice(colours)
        events.append((t, row, col, colour))
    return events

def get_robot_strip(col):
    if col <= 10:
        return 0
    elif col <= 18:
        return 1
    else:
        return 2

def update_waypoints(robot_positions):
    for robot_id, robot in robot_positions.items():
        r, c = robot['grid']
        
        if robot_id not in robot_strip_assignment:
            robot_strip_assignment[robot_id] = get_robot_strip(c)
            robot_waypoint_index[robot_id] = 0
        
        strip = robot_strip_assignment[robot_id]
        waypoints = STRIP_WAYPOINTS[strip]
        current_idx = robot_waypoint_index[robot_id]
        target_r, target_c = waypoints[current_idx]
        
        dist = ((r - target_r)**2 + (c - target_c)**2)**0.5
        if dist <= 1:
            robot_waypoint_index[robot_id] = (current_idx + 1) % len(waypoints)

def get_robot_target(robot_id, robot_positions):
    if robot_id not in robot_strip_assignment:
        r, c = robot_positions[robot_id]['grid']
        robot_strip_assignment[robot_id] = get_robot_strip(c)
        robot_waypoint_index[robot_id] = 0
    
    strip = robot_strip_assignment[robot_id]
    waypoints = STRIP_WAYPOINTS[strip]
    idx = robot_waypoint_index[robot_id]
    return waypoints[idx]

def meters_to_grid(x_m, y_m):
    grid_x = (x_m - ARENA_ORIGIN_X) / ARENA_WIDTH_M * GRID_COLS
    grid_y = (y_m - ARENA_ORIGIN_Y) / ARENA_HEIGHT_M * GRID_ROWS
    col = int(np.clip(grid_x, 0, GRID_COLS - 1))
    row = int(np.clip(grid_y, 0, GRID_ROWS - 1))
    return row, col


def tracking_receiver_thread():
    while True:
        try:
            tsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tsock.connect((TRACKING_SERVER_IP, TRACKING_SERVER_PORT))
            buffer = ""

            while True:
                data = tsock.recv(4096).decode('utf-8', errors='ignore')
                if not data:
                    break
                buffer += data
                lines = buffer.split('\n')
                buffer = lines[-1]

                for line in lines[:-1]:
                    line = line.strip()
                    if line.startswith('P,'):
                        parts = line.split(',')
                        if len(parts) >= 7:
                            marker_id = int(parts[1])
                            x = float(parts[2])
                            y = float(parts[3])
                            theta = float(parts[4])
                            quality = int(parts[6])

                            if quality == 0:
                                continue

                            row, col = meters_to_grid(x, y)

                            with tracking_lock:
                                tracking_positions[marker_id] = {
                                    'grid': (row, col),
                                    'heading': theta
                                }
        except (ConnectionRefusedError, ConnectionResetError, OSError):
            time.sleep(1.0)
            continue



def render_heatmap(layer, title, size=(500, 500), max_val=None):
    display = layer.copy()
    min_display = display.min()
    if max_val is None:
        max_val = display.max()
    display = display - min_display
    range_val = max_val - min_display
    if range_val > 0:
        display = display / range_val
    display_uint8 = (display * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(display_uint8, cv2.COLORMAP_JET)
    coloured = cv2.resize(coloured, size, interpolation=cv2.INTER_NEAREST)

    legend_width = 60
    legend = np.zeros((size[1], legend_width, 3), dtype=np.uint8)

    for i in range(size[1]):
        value = int(255 * (1 - i / size[1]))
        colour = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
        legend[i, :40] = colour

    cv2.putText(legend, f'{max_val:.1f}', (2, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(legend, f'{min_display:.1f}', (2, size[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    combined = np.hstack([coloured, legend])
    return combined


def prepare_broadcast_packet(heatmap, robot_positions):
    heatmap_bytes = heatmap.astype(np.float32).tobytes()

    num_robots = len(robot_positions)
    robot_bytes = struct.pack('B', num_robots)

    for robot_id, robot in robot_positions.items():
        r, c = robot['grid']
        heading = robot['heading']
        target_r, target_c = get_robot_target(robot_id, robot_positions)
        robot_bytes += struct.pack('BBBxfBB',
            int(robot_id),
            int(r),
            int(c),
            float(heading),
            int(target_r),
            int(target_c)
        )

    return heatmap_bytes + robot_bytes


seq_num = 0


def send_multipacket(sock, data, ip, port, max_packet_size=1400):
    global seq_num
    total_packets = (len(data) + max_packet_size - 4) // (max_packet_size - 4)
    offset = 0
    packet_num = 0

    while offset < len(data):
        chunk = data[offset:offset + max_packet_size - 4]
        is_last = 1 if offset + len(chunk) >= len(data) else 0
        header = struct.pack('BBBB', seq_num % 256, packet_num, total_packets, is_last)
        sock.sendto(header + chunk, (ip, port))
        offset += len(chunk)
        packet_num += 1

    seq_num += 1

def get_coverage_pct():
    visited = np.sum(cell_visit_counts > 0)
    total = GRID_ROWS * GRID_COLS
    return round(visited / total * 100, 2)

def get_avg_robot_distance():
    positions = [(r, c) for robot in robot_positions.values() 
                 for r, c in [robot['grid']]]
    if len(positions) < 2:
        return 0.0
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dr = positions[i][0] - positions[j][0]
            dc = positions[i][1] - positions[j][1]
            distances.append(round((dr**2 + dc**2)**0.5, 3))
    return round(sum(distances) / len(distances), 3)

def get_near_collisions(threshold=5):
    positions = [(r, c) for robot in robot_positions.values()
                 for r, c in [robot['grid']]]
    count = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dr = positions[i][0] - positions[j][0]
            dc = positions[i][1] - positions[j][1]
            if (dr**2 + dc**2)**0.5 < threshold:
                count += 1
    return count

def start_trial():
    global recording, trial_start_time, trial_number, trial_system_name
    global trial_writer, trial_file, cell_visit_counts
    global mess_paint_time, mess_reached_time, last_snapshot_time

    system_name = input("Enter system name (e.g. main_system, boustrophedon, priority): ").strip()
    if not system_name:
        system_name = "unnamed"

    trial_system_name = system_name

    subfolder = os.path.join(RESULTS_FOLDER, system_name)
    os.makedirs(subfolder, exist_ok=True)
    existing = [f for f in os.listdir(subfolder) if f.startswith('trial_') and f.endswith('.csv')]
    trial_number = len(existing) + 1
    filename = os.path.join(subfolder, f"trial_{trial_number:03d}_{system_name}.csv")
    global trial_filename
    trial_filename = filename
    trial_file = open(filename, 'w', newline='')
    trial_writer = csv.writer(trial_file)
    trial_writer.writerow([
        'timestamp', 'type', 'coverage_pct', 'avg_robot_distance',
        'near_collisions', 'robot_positions',
        'total_reward', 'per_robot_reward',
        'event', 'details'
    ])

    cell_visit_counts = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
    global total_reward_collected, robot_reward_collected
    total_reward_collected = 0.0
    robot_reward_collected = {}
    global active_messes
    active_messes = []
    accumulate_layer[:] = ACCUMULATE_CAP
    mess_layer[:] = 0
    global random_mess_events
    if auto_mess_random:
        random_mess_events = generate_random_mess_events()
        print(f"Random mess events: {random_mess_events}")
    global system_active
    system_active = True
    trial_start_time = time.time()
    last_snapshot_time = time.time()
    recording = True
    global video_frames
    video_frames = []
    print(f"Recording trial {trial_number} - {system_name} - {filename}")

def stop_trial():
    global recording, trial_file, trial_writer

    if not recording:
        return

    trial_writer.writerow([
        round(time.time() - trial_start_time, 2),
        'summary',
        get_coverage_pct(),
        get_avg_robot_distance(),
        get_near_collisions(),
        str({rid: r['grid'] for rid, r in robot_positions.items()}),
        round(float(total_reward_collected), 1),
        str({rid: round(float(v), 1) for rid, v in robot_reward_collected.items()}),
        'trial_end',
        f"total_cells_visited={np.sum(cell_visit_counts > 0)}"
    ])
    trial_writer.writerow(['visit_counts_grid', '', '', '', '', '', '', ''])
    for r in range(GRID_ROWS):
        trial_writer.writerow([f'row_{r}'] + list(cell_visit_counts[r]))

    trial_file.close()
    recording = False
    global record_video, video_frames
    if video_frames:
        video_path = trial_filename.replace('.csv', '.mp4')
        h, w = video_frames[0].shape[:2]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        for frame in video_frames:
            out.write(frame)
        out.release()
        video_frames = []
        print(f"Video saved to {video_path}")
    record_video = False
    global auto_mess
    auto_mess = False
    auto_mess_random = False
    random_mess_events = []
    global system_active
    system_active = False
    print(f"Trial saved.")

def log_snapshot():
    if not recording or trial_writer is None:
        return
    trial_writer.writerow([
        round(time.time() - trial_start_time, 2),
        'snapshot',
        get_coverage_pct(),
        get_avg_robot_distance(),
        get_near_collisions(),
        str({rid: r['grid'] for rid, r in robot_positions.items()}),
        round(float(total_reward_collected), 1),
        str({rid: round(float(v), 1) for rid, v in robot_reward_collected.items()}),
        '',
        ''
    ])

def log_event(event, details=''):
    if not recording or trial_writer is None:
        return
    trial_writer.writerow([
        round(time.time() - trial_start_time, 2),
        'event',
        get_coverage_pct(),
        get_avg_robot_distance(),
        get_near_collisions(),
        str({rid: r['grid'] for rid, r in robot_positions.items()}),
        round(float(total_reward_collected), 1),
        str({rid: round(float(v), 1) for rid, v in robot_reward_collected.items()}),
        event,
        details
    ])
DISPLAY_WIDTH  = 550
DISPLAY_HEIGHT = int(DISPLAY_WIDTH * (ARENA_HEIGHT_M / ARENA_WIDTH_M))
cell_px_x = DISPLAY_WIDTH / GRID_COLS
cell_px_y = DISPLAY_HEIGHT / GRID_ROWS

paint_colour_value = RED_VALUE
paint_colour_name = "RED"
is_painting = False
is_erasing = False


def paint_mouse_callback(event, x, y, flags, param):
    global is_painting, is_erasing

    col = int(x / cell_px_x)
    row = int(y / cell_px_y)

    if event == cv2.EVENT_LBUTTONDOWN:
        is_painting = True
    elif event == cv2.EVENT_LBUTTONUP:
        is_painting = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        is_erasing = True
    elif event == cv2.EVENT_RBUTTONUP:
        is_erasing = False

    if (is_painting or is_erasing) and 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
        painted_new_mess = False
        rr = int(np.clip(row, 0, GRID_ROWS - 1))
        cc = int(np.clip(col, 0, GRID_COLS - 1))
        if is_painting:
            if mess_layer[rr, cc] == 0:
                painted_new_mess = True
            mess_layer[rr, cc] = paint_colour_value
        elif is_erasing:
            mess_layer[rr, cc] = 0.0

        if is_painting and painted_new_mess and recording:
            global mess_paint_time, mess_reached_time
            mess_paint_time = time.time()
            mess_reached_time = None
            log_event('mess_painted', f"colour={paint_colour_name} at grid=({row},{col})")


cv2.namedWindow('Final Heatmap - Paint Mess Here')
cv2.setMouseCallback('Final Heatmap - Paint Mess Here', paint_mouse_callback)

tracking_thread = threading.Thread(target=tracking_receiver_thread, daemon=True)
tracking_thread.start()

print("System running.")
print("Paint controls: 1=Red(hazard) 2=Green(standard) 3=Blue(low priority)")
print("Left click+drag to paint mess, right click+drag to erase, q to quit")

while True:
    now = time.time()
    dt = now - last_time
    last_time = now

    with tracking_lock:
        current_tracking = dict(tracking_positions)

    current_time = time.time()
    for marker_id, pos in current_tracking.items():
        robot_positions[marker_id] = pos
        robot_last_seen[marker_id] = current_time

    for robot_id in list(robot_positions.keys()):
        if current_time - robot_last_seen.get(robot_id, 0) > ROBOT_TIMEOUT:
            robot_positions.pop(robot_id, None)

    update_waypoints(robot_positions)

    for robot_id, robot in robot_positions.items():
        r, c = robot['grid']

        if robot_id in previous_robot_positions:
            prev_r, prev_c = previous_robot_positions[robot_id]
            steps = max(abs(r - prev_r), abs(c - prev_c))
        else:
            prev_r, prev_c = r, c
            steps = 0

        for step in range(steps + 1):
            if steps > 0:
                interp_r = int(round(prev_r + (r - prev_r) * step / steps))
                interp_c = int(round(prev_c + (c - prev_c) * step / steps))
            else:
                interp_r, interp_c = r, c

            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    rr = int(np.clip(interp_r + dr, 0, GRID_ROWS - 1))
                    cc = int(np.clip(interp_c + dc, 0, GRID_COLS - 1))
                    if recording:
                        reward = final_heatmap[rr, cc]
                        total_reward_collected += reward
                        robot_reward_collected[robot_id] = robot_reward_collected.get(robot_id, 0.0) + reward
                        cell_visit_counts[rr, cc] += 1
                    accumulate_layer[rr, cc] = 100
                    if recording and mess_layer[rr, cc] > 0:
                        for mess in active_messes[:]:
                            if abs(rr - mess['row']) <= 1 and abs(cc - mess['col']) <= 1:
                                response_time = round(time.time() - mess['paint_time'], 2)
                                efficiency = round(mess['nearest_distance'] / response_time, 3) if response_time > 0 else 0
                                log_event('mess_reached', f"robot_{robot_id} colour={mess['colour']} at ({mess['row']},{mess['col']}) response_time={response_time}s nearest_distance={mess['nearest_distance']:.2f} efficiency={efficiency}")
                                active_messes.remove(mess)
                                break
                    mess_layer[rr, cc] = 0

        previous_robot_positions[robot_id] = (r, c)

    accumulate_layer += ACCUMULATE_RATE * dt
    accumulate_layer = np.clip(accumulate_layer, 0, ACCUMULATE_CAP)


    final_heatmap = (mess_layer * MESS_WEIGHT) + (accumulate_layer * ACCUMULATE_WEIGHT)
    BORDER = 3
    final_heatmap[0:BORDER, :]  = EDGE_PENALTY
    final_heatmap[-BORDER:, :]  = EDGE_PENALTY
    final_heatmap[:, 0:BORDER]  = EDGE_PENALTY
    final_heatmap[:, -BORDER:]  = EDGE_PENALTY

    display_img = render_heatmap(final_heatmap, 'Final Heatmap', size=(DISPLAY_WIDTH, DISPLAY_HEIGHT))

    grid_only = display_img[:, :DISPLAY_WIDTH]
    for i in range(GRID_ROWS + 1):
        y = int(i * cell_px_y)
        cv2.line(grid_only, (0, y), (DISPLAY_WIDTH, y), (50, 50, 50), 1)
    for i in range(GRID_COLS + 1):
        x = int(i * cell_px_x)
        cv2.line(grid_only, (x, 0), (x, DISPLAY_HEIGHT), (50, 50, 50), 1)

    for robot_id, robot in robot_positions.items():
        r, c = robot['grid']
        target_r, target_c = get_robot_target(robot_id, robot_positions)
        px = int(c * cell_px_x + cell_px_x / 2)
        py = int(r * cell_px_y + cell_px_y / 2)
        tx = int(target_c * cell_px_x + cell_px_x / 2)
        ty = int(target_r * cell_px_y + cell_px_y / 2)
        cv2.line(grid_only, (px, py), (tx, ty), (0, 255, 0), 2)
        cv2.circle(grid_only, (tx, ty), 5, (0, 255, 0), -1)
        cv2.putText(grid_only, f'R{robot_id}', (px, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(grid_only, f'R{robot_id}', (px, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if recording:
        elapsed = time.time() - trial_start_time
        remaining = max(0, TRIAL_DURATION_S - elapsed)
        timer_text = f'{int(remaining//60):01d}:{int(remaining%60):02d}'
        cv2.putText(display_img, timer_text, (DISPLAY_WIDTH - 50, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(display_img, timer_text, (DISPLAY_WIDTH - 50, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(display_img, f'Pen: {paint_colour_name}', (10, DISPLAY_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Final Heatmap - Paint Mess Here', display_img)
    if recording and record_video:
        video_frames.append(display_img.copy())

    if frame_count % HEATMAP_UPDATE_INTERVAL == 0:
        if system_active:
            payload = prepare_broadcast_packet(final_heatmap, robot_positions)
            for ip in ROBOT_IPS:
                send_multipacket(sock, payload, ip, UDP_PORT)

    if recording and time.time() - last_snapshot_time >= snapshot_interval:
        log_snapshot()
        last_snapshot_time = time.time()

    if recording and auto_mess:
        elapsed = time.time() - trial_start_time
        for event_time, event_row, event_col, event_colour in MESS_EVENTS:
            if abs(elapsed - event_time) < 0.5:
                newly_painted = False
                for dr in range(2):
                    for dc in range(2):
                        rr = int(np.clip(event_row + dr, 0, GRID_ROWS - 1))
                        cc = int(np.clip(event_col + dc, 0, GRID_COLS - 1))
                        if mess_layer[rr, cc] == 0:
                            mess_layer[rr, cc] = event_colour
                            newly_painted = True
                if newly_painted:
                    min_dist = min([((r['grid'][0] - event_row)**2 + (r['grid'][1] - event_col)**2)**0.5 
                                   for r in robot_positions.values()]) if robot_positions else -1
                    active_messes.append({
                        'row': event_row, 'col': event_col,
                        'paint_time': time.time(),
                        'nearest_distance': min_dist,
                        'colour': event_colour
                    })
                    robot_pos_str = str({rid: r['grid'] for rid, r in robot_positions.items()})
                    log_event('mess_painted_auto', f"colour={event_colour} at ({event_row},{event_col}) robot_positions={robot_pos_str} nearest_robot_distance={min_dist}")
    if recording and auto_mess_random:
        elapsed = time.time() - trial_start_time
        for event_time, event_row, event_col, event_colour in random_mess_events:
            if abs(elapsed - event_time) < 0.5:
                newly_painted = False
                for dr in range(2):
                    for dc in range(2):
                        rr = int(np.clip(event_row + dr, 0, GRID_ROWS - 1))
                        cc = int(np.clip(event_col + dc, 0, GRID_COLS - 1))
                        if mess_layer[rr, cc] == 0:
                            mess_layer[rr, cc] = event_colour
                            newly_painted = True
                if newly_painted:
                    min_dist = min([((r['grid'][0] - event_row)**2 + (r['grid'][1] - event_col)**2)**0.5 
                                   for r in robot_positions.values()]) if robot_positions else -1
                    active_messes.append({
                        'row': event_row, 'col': event_col,
                        'paint_time': time.time(),
                        'nearest_distance': min_dist,
                        'colour': event_colour
                    })
                    robot_pos_str = str({rid: r['grid'] for rid, r in robot_positions.items()})
                    log_event('mess_painted_auto', f"colour={event_colour} at ({event_row},{event_col}) robot_positions={robot_pos_str} nearest_robot_distance={min_dist}")
    if recording and time.time() - trial_start_time >= TRIAL_DURATION_S:
        print("Trial complete.")
        stop_trial()

    frame_count += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        if recording:
            stop_trial()
        break
    elif key == ord('s'):
        system_active = not system_active
        print("System active" if system_active else "System paused")
    elif key == ord('v'):
        record_video = not record_video
        print("Video recording ON" if record_video else "Video recording OFF")
    elif key == ord('r'):
        if not recording:
            auto_mess = False
            start_trial()
        else:
            print("Already recording")
    elif key == ord('m'):
        if not recording:
            auto_mess = True
            start_trial()
        else:
            print("Already recording")
    elif key == ord('n'):
        if not recording:
            auto_mess_random = True
            start_trial()
        else:
            print("Already recording")
    elif key == ord('e'):
        if recording:
            stop_trial()
        else:
            print("Not currently recording")
    elif key == ord('1'):
        paint_colour_value = RED_VALUE
        paint_colour_name = "RED"
    elif key == ord('2'):
        paint_colour_value = GREEN_VALUE
        paint_colour_name = "GREEN"
    elif key == ord('3'):
        paint_colour_value = BLUE_VALUE
        paint_colour_name = "BLUE"

cv2.destroyAllWindows()