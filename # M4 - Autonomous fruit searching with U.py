# M4 - Autonomous fruit searching with UI + RRT (Level 2)

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import pygame
import threading
import queue
import re

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure

# RRT imports
from rrt_planner import plan_from_files, plot_world_and_path

# YOLO detector (always ON for Level 2 GUI)
from YOLO.detector import Detector


# =========================
# Covariance-triggered scan config
# =========================
THETA_M_SCALE    = 0.30   # metres per rad (unit balancing for yaw)
COV_TRACE_HI     = 0.10   # trigger threshold (weighted trace ~ m^2)
COV_TRACE_LO     = 0.03   # (kept for potential hysteresis use)
RELOCALISE_RADIUS = 0.35  # spin only when within this distance (m) of waypoint

# NEW: spin-on-no-landmarks threshold (forward-drive loop only)
NO_LM_SPIN_ITERS = 20     # consecutive iterations with no landmarks before spinning

# --- Cooldown helpers (use a monotonic clock) ---
SPIN_COOLDOWN = 8.0        # seconds between allowed spins
_last_spin_ts = -1e9       # far in the past so first check never waits
_is_spinning  = False

def _now(): return time.monotonic()
def _can_spin(): return (_now() - _last_spin_ts) >= SPIN_COOLDOWN
def _mark_spin():
    global _last_spin_ts
    _last_spin_ts = _now()


def read_true_map(fname):
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()
        for fruit in fruits:
            search_list.append(fruit.strip())
    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(
                    n_fruit,
                    fruit,
                    np.round(fruit_true_pos[i][0], 1),
                    np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


def _wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


# =========================
# Covariance helpers + spin
# =========================
def robot_cov_trace(ekf, theta_m_scale=THETA_M_SCALE):
    """Weighted trace of robot covariance; scales theta to metres so units match x,y."""
    P = ekf.P[:3, :3].copy()
    W = np.diag([1.0, 1.0, theta_m_scale**2])
    Pw = W @ P @ W
    return float(np.trace(Pw))

def covariance_too_high(ekf):
    tr = robot_cov_trace(ekf)
    print(f"[COV] weighted trace={tr:.4f} (thr={COV_TRACE_HI})")
    return tr > COV_TRACE_HI

def spin_360(turn_dir=1, wheel_tick=35, *, burst_deg=23, pause_s=1.0,
             slice_dt=0.12, max_slices_per_burst=40, max_bursts=8):
    """
    Turn in ~8 bursts of ~45° each, pausing between bursts.
    """
    global _is_spinning
    if _is_spinning:
        return
    _is_spinning = True

    def _wrap(a): return (a + np.pi) % (2*np.pi) - np.pi
    def _dang(a_new, a_old): return _wrap(a_new - a_old)

    target_total = np.deg2rad(180)
    target_burst = np.deg2rad(burst_deg)
    total_accum = 0.0
    theta_prev = get_robot_pose()[2]
    eps = np.deg2rad(0.5)                     # EKF yaw “small change” threshold
    fallback_step = np.deg2rad(3.0) * (1 if turn_dir >= 0 else -1)

    print("[SPIN] starting 360° in bursts...")
    try:
        for b in range(max_bursts):
            if total_accum >= target_total:
                break

            burst_accum = 0.0
            slices = 0
            while burst_accum < target_burst and slices < max_slices_per_burst:
                # 1) command constant turn for slice_dt
                with ppi_lock:
                    lv, rv = ppi.set_velocity([0, turn_dir], turning_tick=wheel_tick)
                time.sleep(slice_dt)

                # 2) predict from wheels
                ekf.predict(measure.Drive(lv, -rv, float(slice_dt)))

                # 3) update from markers (if any)
                with ppi_lock:
                    img = ppi.get_image()
                lms, _ = aruco_det.detect_marker_positions(img)
                lms = _filter_known_lms(lms, ekf)
                ekf.update(lms)

                # 4) accumulate yaw
                theta_now = get_robot_pose()[2]
                dth_meas = _dang(theta_now, theta_prev)

                if abs(dth_meas) < eps:
                    dth = fallback_step
                    theta_prev = _wrap(theta_prev + dth)
                else:
                    dth = dth_meas
                    theta_prev = theta_now

                burst_accum += abs(dth)
                total_accum += abs(dth)
                slices += 1

                if total_accum >= target_total:
                    break

            with ppi_lock:
                ppi.set_velocity([0, 0])
            time.sleep(pause_s)
            print(f"[SPIN] burst {b+1}: ~{np.degrees(burst_accum):.1f}°, total ~{np.degrees(total_accum):.1f}°")

        print(f"[SPIN] done: ~{np.degrees(total_accum):.1f}° total.")
    finally:
        with ppi_lock:
            ppi.set_velocity([0, 0])
        _is_spinning = False



def drive_to_point(waypoint, robot_pose):
    wheel_vel = 30
    Kp_time = 0.4
    min_dt, max_dt = 0.08, 0.4
    max_iterations = 1000
    iters = 0

    desired_rotation = np.arctan2(
        waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
    ang_tol = 3 * np.pi / 180
    rotation_error = _wrap_angle(desired_rotation - robot_pose[2])

    # --- Rotation phase (no relocalise here) ---
    while abs(rotation_error) > ang_tol and iters < max_iterations:
        turn_dir = 1 if rotation_error > 0 else -1
        dt = float(np.clip(Kp_time * abs(rotation_error), min_dt, max_dt))

        with ppi_lock:
            lv, rv = ppi.set_velocity([0, turn_dir], turning_tick=wheel_vel, time=dt)

        drive_meas = measure.Drive(lv, -rv, dt)
        ekf.predict(drive_meas)

        with ppi_lock:
            img = ppi.get_image()
        lms, _ = aruco_det.detect_marker_positions(img)
        for lm in lms:
            print("Detected:", lm.tag, lm.position.ravel())
        lms = _filter_known_lms(lms, ekf)
        ekf.update(lms)

        robot_pose = get_robot_pose()
        rotation_error = _wrap_angle(desired_rotation - robot_pose[2])
        iters += 1

    print(f"Turned to face waypoint. Final error: {abs(rotation_error*180/np.pi):.2f} deg")

    # --- Forward drive phase ---
    pos_tol = 0.05
    dt_drive = 0.2
    ang_tol_drive = 3 * np.pi / 180
    iters = 0
    did_relocalise = False   # spin at most once per waypoint (near the goal)
    no_lm_iters = 0          # NEW: counts consecutive iterations with no landmarks

    dx = waypoint[0] - robot_pose[0]
    dy = waypoint[1] - robot_pose[1]
    dist = float(np.hypot(dx, dy))

    while dist > pos_tol and iters < max_iterations:
        desired_rotation = np.arctan2(
            waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
        heading_err = _wrap_angle(desired_rotation - robot_pose[2])

        if abs(heading_err) > ang_tol_drive:
            turn_dir = 1 if heading_err > 0 else -1
            dt = float(np.clip(Kp_time * abs(heading_err), min_dt, max_dt))
            with ppi_lock:
                lv, rv = ppi.set_velocity([0, turn_dir], turning_tick=wheel_vel, time=dt)
            drive_meas = measure.Drive(lv, -rv, dt)
        else:
            with ppi_lock:
                lv, rv = ppi.set_velocity([1, 0], tick=wheel_vel)
            time.sleep(dt_drive)
            drive_meas = measure.Drive(lv, -rv, dt_drive)

        ekf.predict(drive_meas)
        with ppi_lock:
            img = ppi.get_image()
        lms, _ = aruco_det.detect_marker_positions(img)
        lms = _filter_known_lms(lms, ekf)
        ekf.update(lms)

        # track consecutive iterations with no landmarks during forward drive
        if lms and len(lms) > 0:
            no_lm_iters = 0
        else:
            no_lm_iters += 1
            if no_lm_iters >= NO_LM_SPIN_ITERS and _can_spin():
                print(f"[RELOCALISE] No landmarks for {no_lm_iters} iters -> 360 spin.")
                _mark_spin()  # reserve cooldown BEFORE spinning
                spin_360(turn_dir=1 if heading_err >= 0 else -1,
                        wheel_tick=wheel_vel,
                        burst_deg=45.0,   # ~45° per burst
                        pause_s=1.0,      # pause between bursts
                        slice_dt=0.12)    # motor command duration inside a burst
                no_lm_iters = 0  # reset after spin

        # update current pose & distance to waypoint (use fresh state)
        robot_pose = get_robot_pose()
        dx = waypoint[0] - robot_pose[0]
        dy = waypoint[1] - robot_pose[1]
        dist = float(np.hypot(dx, dy))

        # (Optional, keep existing near-waypoint covariance trigger)
        if (not did_relocalise) and (dist <= RELOCALISE_RADIUS) and covariance_too_high(ekf) and _can_spin():
            print(f"[RELOCALISE] Near goal (dist={dist:.2f} m). Spinning for relocalisation.")
            _mark_spin()
            spin_360(turn_dir=1 if heading_err >= 0 else -1,
                    wheel_tick=wheel_vel,
                    burst_deg=45.0,   # ~45° per burst
                    pause_s=1.0,      # pause between bursts
                    slice_dt=0.12)    # motor command duration inside a burst

            did_relocalise = True

        iters += 1

    print(f"Arrived at {waypoint}")


def get_robot_pose():
    s = ekf.robot.state.ravel()
    return [float(s[0]), float(s[1]), float(s[2])]


def load_aruco_markers(fname):
    with open(fname, 'r') as fd:
        gt = json.load(fd)

    lms = []
    for key, val in gt.items():
        if key.startswith("aruco"):
            m = re.search(r'aruco(\d+)', key)
            if not m:
                continue
            tag = int(m.group(1))
            pos = np.array([[float(val["x"])],
                            [float(val["y"])]], dtype=float)
            cov = (1e-6) * np.eye(2)
            lms.append(measure.Marker(position=pos, tag=tag, covariance=cov))
    return lms


def get_rrt_waypoints(map_file="map.txt", search_file="search_list.txt", start=(0.0, 0.0)):
    result = plan_from_files(map_file, search_file, start=start,
                             max_iters=8000,
                             step_size=0.15,
                             goal_region_radius=0.3,
                             robot_radius=0.09,
                             safety_margin=0.06)
    
    # --- ADD THIS BACK ---
    with open(map_file) as f:
        world = json.load(f)
    plot_world_and_path(world, start, result, color_by_leg=True)

    return result["path"]


def _filter_known_lms(lms, ekf):
    known = set(ekf.taglist)
    kept, skipped = [], []
    for lm in lms:
        for i in range(1,11):
            if lm.tag == i:
                kept.append(lm)
                break
    return kept

#

def _nearest_fruit_distance(frame_rgb, K):
    """
    Returns (min_dist_m, label, conf) across all detected fruits, or (None, None, None) if none valid.
    Uses YOLO's xywh and the pinhole formula with fx from intrinsics K.
    """
    dets, _ = detector.detect_single_image(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    best = (None, None, None)
    for (label, xywh, conf) in dets:
        if label not in true_dims:   # ignore classes we don't have dimensions for
            continue
        d = estimate_distance_from_bbox(K, xywh, label)
        if d is None:
            continue
        if best[0] is None or d < best[0]:
            best = (d, label, conf)
    return best  # (min_dist_m or None, label or None, conf or None)



#Fruit Detector.

true_dims = {'orange': [0.0769 , 0.0747, 0.0719], 'lemon': [0.0502, 0.0664 , 0.0477], 
                                'pear': [0.0707, 0.0733, 0.1113], 'tomato': [0.0706, 0.0688, 0.0602], 
                                'capsicum': [0.0823, 0.0823, 0.0896], 'potato': [0.0678, 0.0947, 0.0549], 
                                'pumpkin': [0.0850, 0.0856, 0.0774], 'garlic': [0.0653, 0.0593, 0.0725]}

def estimate_distance_from_bbox(camera_matrix, bbox_xywh, label):
    """
    Approx distance robot->object centerline using pinhole model on bbox height.
    bbox_xywh: [x_c, y_c, w, h] in pixels (Ultralytics returns coords in original frame scale)
    Uses focal length in pixels from intrinsics (fx).
    """
    fx = float(camera_matrix[0][0]) #camera length in pixels along the x axis
    pix_h = float(bbox_xywh[3])#height of the detected bounding box
    true_h = float(true_dims.get(label, [0,0,0])[2])#true height of the fruits
    if pix_h <= 1 or true_h <= 0:
        return None
    return (true_h / pix_h) * fx



# =========================
# Pygame UI drawing
# =========================
def draw_ui(canvas, frame, yolo_img, ekf, notification=""):
    canvas.fill((0, 0, 0))
    text_colour = (220, 220, 220)
    v_pad = 40
    h_pad = 20

    ekf_view = ekf.draw_slam_state(res=(320, 480 + v_pad), not_pause=True)
    canvas.blit(ekf_view, (2 * h_pad + 320, v_pad))

    robot_view = cv2.resize(frame, (320, 240))
    draw_pygame_window(canvas, robot_view, (h_pad, v_pad))

    detector_view = cv2.resize(yolo_img, (320, 240), cv2.INTER_NEAREST)
    draw_pygame_window(canvas, detector_view, (h_pad, 240 + 2 * v_pad))

    put_caption(canvas, 'SLAM', (2 * h_pad + 320, v_pad))
    put_caption(canvas, 'Detector', (h_pad, 240 + 2 * v_pad))
    put_caption(canvas, 'PiBot Cam', (h_pad, v_pad))

    notif_surface = TEXT_FONT.render(notification, False, text_colour)
    canvas.blit(notif_surface, (h_pad + 10, 596))


def draw_pygame_window(canvas, cv2_img, position):
    cv2_img = np.rot90(cv2_img)
    view = pygame.surfarray.make_surface(cv2_img)
    view = pygame.transform.flip(view, True, False)
    canvas.blit(view, position)


def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
    caption_surface = TITLE_FONT.render(caption, False, text_colour)
    canvas.blit(caption_surface, (position[0], position[1] - 25))


# =========================
# Globals
# =========================
status_text = "Level 2 RRT Navigation"
ppi_lock = threading.Lock()


def motion_worker(legs):
    global status_text
    for i, leg in enumerate(legs):
        for j, wp in enumerate(leg):
            status_text = f"Driving to {np.round(wp, 3)}"
            robot_pose = get_robot_pose()
            drive_to_point(list(wp), robot_pose)
            robot_pose = get_robot_pose()
            print(f"Finished driving to {wp}; pose={robot_pose}")
            with ppi_lock:
                ppi.set_velocity([0, 0])
            status_text = f"Arrived at {np.round(wp, 3)}"

            if j == len(leg) - 1:
                print(f"Reached end of leg {i+1}, pausing 5s...")
                time.sleep(5)


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M3_prac_map_full.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip, args.port)

    scale = float(np.loadtxt("calibration/param/scale.txt", delimiter=','))
    baseline = float(np.loadtxt("calibration/param/baseline.txt", delimiter=','))

    K = np.loadtxt("calibration/param/intrinsic.txt", delimiter=',').reshape(3, 3)
    dist = np.loadtxt("calibration/param/distCoeffs.txt").ravel().reshape(1, -1)

    robot = Robot(wheels_width=baseline, wheels_scale=scale,
                  camera_matrix=K, camera_dist=dist)
    ekf = EKF(robot)

    aruco_det = aruco.aruco_detector(ekf.robot, marker_length=0.07)
    lms = load_aruco_markers(args.map)
    ekf.add_landmarks(lms)

    detector = Detector("YOLO/model/yolov8_model.pt")

    # waypoints = get_rrt_waypoints(args.map, "search_list.txt", start=(0.0, 0.0))
    # print("Waypoints:", waypoints)
    res = plan_from_files(args.map, "search_list.txt", start=(0.0, 0.0))
    with open(args.map) as f:
        world = json.load(f)
    plot_world_and_path(world, (0.0, 0.0), res, color_by_leg=True) 

    pygame.font.init()
    TITLE_FONT = pygame.font.SysFont(None, 28)
    TEXT_FONT = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 Level 2 - RRT Navigation')

    # threading.Thread(target=motion_worker, args=(waypoints,), daemon=True).start()
    threading.Thread(target=motion_worker, args=(res["legs"],), daemon=True).start()

    # === ADD: throttle variable for printing distances ===
    last_print_ts = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with ppi_lock:
            frame = ppi.get_image()
        if frame is None:
            clock.tick(30)
            continue

        # --- detect and draw ArUco markers for UI ---
        measurements, aruco_marked = aruco_det.detect_marker_positions(frame)
        measurements = _filter_known_lms(measurements, ekf)
        # UI-only; predict/update occur in the motion thread

        detections, yolo_vis = detector.detect_single_image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        yolo_vis = cv2.cvtColor(yolo_vis, cv2.COLOR_BGR2RGB)

        # === ADD: print distance for each detection (every 0.5 s) ===
        now_ts = time.monotonic()
        if now_ts - last_print_ts >= 0.5:
            for (label, xywh, conf) in detections:
                dist_m = estimate_distance_from_bbox(K, xywh, label)
                if dist_m is not None:
                    print(f"[DETECT] {label:9s}  conf={conf:0.2f}  approx_dist={dist_m:0.2f} m")
            last_print_ts = now_ts

        # show UI with aruco_marked (green outlines visible)
        draw_ui(canvas, aruco_marked, yolo_vis, ekf, status_text)
        pygame.display.update()
        clock.tick(30)

    with ppi_lock:
        ppi.set_velocity([0, 0])
