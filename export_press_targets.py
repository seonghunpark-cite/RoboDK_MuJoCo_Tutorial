from robodk.robolink import Robolink
from robodk.robolink import ITEM_TYPE_ROBOT, ITEM_TYPE_TOOL, ITEM_TYPE_FRAME, ITEM_TYPE_TARGET
import csv
import math
import os
import time


# =====================================================
# User settings
# =====================================================
ROBOT_NAME = "FAIRINO FR5"
TOOL_NAME = "Tool 1"
FRAME_NAME = "Hemisphere"

TARGET_PREFIX = "Target_D"
OUT_CSV = "robodk_press_targets.csv"

# Unit conversion:
# RoboDK uses mm and deg. MuJoCo uses m and rad.
SAVE_METER_AND_RADIAN_COLUMNS = True

# When True, the script actually moves the robot to each target before reading joints.
# This avoids direct SolveIK() calls and uses RoboDK's own MoveJ behavior.
MOVE_ROBOT_TO_TARGETS = True

# Delay after each MoveJ, seconds. Increase if RoboDK needs more time.
MOVE_WAIT_SEC = 0.05

# If a target causes an exception during MoveJ, skip it rather than stopping.
SKIP_FAILED_TARGETS = True

# Normal direction:
# For an inner spherical surface, outward normal from sphere center to contact point is +p/|p|.
# If your MuJoCo force direction looks reversed, change NORMAL_SIGN to -1.0.
NORMAL_SIGN = 1.0


# =====================================================
# Math helpers
# =====================================================
def normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-9:
        return [0.0, 0.0, 0.0], 0.0
    return [x / n for x in v], n


def pose_to_matrix_values(pose):
    return [
        pose[0, 0], pose[0, 1], pose[0, 2],
        pose[1, 0], pose[1, 1], pose[1, 2],
        pose[2, 0], pose[2, 1], pose[2, 2],
    ]


def flatten_joints(joints):
    vals = list(joints.list())
    vals = [float(v) for v in vals]

    if len(vals) < 6:
        raise ValueError(f"Invalid joint vector: {vals}")

    return vals[:6]


def target_index_key(item):
    name = item.Name()
    tail = name.replace(TARGET_PREFIX, "", 1)
    try:
        return int(tail)
    except Exception:
        return name


# =====================================================
# Connect to RoboDK
# =====================================================
RDK = Robolink()

robot = RDK.Item(ROBOT_NAME, ITEM_TYPE_ROBOT)
tool = RDK.Item(TOOL_NAME, ITEM_TYPE_TOOL)
frame = RDK.Item(FRAME_NAME, ITEM_TYPE_FRAME)

if not robot.Valid():
    raise RuntimeError(f"Robot not found: {ROBOT_NAME}")
if not tool.Valid():
    raise RuntimeError(f"Tool not found: {TOOL_NAME}")
if not frame.Valid():
    raise RuntimeError(f"Frame not found: {FRAME_NAME}")

robot.setPoseTool(tool)
robot.setPoseFrame(frame)

# Find manually taught press targets.
targets = []
for item in RDK.ItemList():
    if item.Type() == ITEM_TYPE_TARGET and item.Name().startswith(TARGET_PREFIX):
        targets.append(item)

targets.sort(key=target_index_key)

if not targets:
    raise RuntimeError(f"No targets found with prefix: {TARGET_PREFIX}")

print(f"Found {len(targets)} targets:")
for t in targets:
    print(" -", t.Name())


# =====================================================
# Export
# =====================================================
rows = []

for target in targets:
    name = target.Name()

    try:
        if MOVE_ROBOT_TO_TARGETS:
            robot.MoveJ(target)
            RDK.Render()
            time.sleep(MOVE_WAIT_SEC)

        joints_deg = flatten_joints(robot.Joints())

        # Target pose in Hemisphere frame.
        # Using absolute pose conversion is safer than target.Pose(),
        # because target.Pose() may depend on how the target was created.
        pose_hemi = frame.PoseAbs().inv() * target.PoseAbs()

        x_mm = float(pose_hemi[0, 3])
        y_mm = float(pose_hemi[1, 3])
        z_mm = float(pose_hemi[2, 3])

        n_vec, radius_mm = normalize([x_mm, y_mm, z_mm])
        nx = NORMAL_SIGN * n_vec[0]
        ny = NORMAL_SIGN * n_vec[1]
        nz = NORMAL_SIGN * n_vec[2]

        # Longitude phi: xy angle around hemisphere center.
        phi_deg = math.degrees(math.atan2(y_mm, x_mm))

        # Polar angle for lower hemisphere:
        # polar=0 at bottom direction, polar=90 at rim.
        if radius_mm > 1e-9:
            val = max(-1.0, min(1.0, -z_mm / radius_mm))
            polar_deg = math.degrees(math.acos(val))
        else:
            polar_deg = float("nan")

        R_values = pose_to_matrix_values(pose_hemi)

        row = [
            name,
            phi_deg,
            polar_deg,
            radius_mm,
            *joints_deg,
            x_mm, y_mm, z_mm,
            *R_values,
            nx, ny, nz,
        ]

        if SAVE_METER_AND_RADIAN_COLUMNS:
            joints_rad = [math.radians(j) for j in joints_deg]
            row.extend([
                *joints_rad,
                x_mm / 1000.0,
                y_mm / 1000.0,
                z_mm / 1000.0,
            ])

        rows.append(row)

        print(
            f"Exported {name}: "
            f"phi={phi_deg:+.2f} deg, polar={polar_deg:.2f} deg, "
            f"r={radius_mm:.2f} mm"
        )

    except Exception as exc:
        message = f"Failed {name}: {exc}"
        if SKIP_FAILED_TARGETS:
            print(message)
            continue
        raise RuntimeError(message) from exc


header = [
    "target_name",
    "phi_deg",
    "polar_deg",
    "radius_mm",
    "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg",
    "x_mm", "y_mm", "z_mm",
    "r11", "r12", "r13",
    "r21", "r22", "r23",
    "r31", "r32", "r33",
    "nx", "ny", "nz",
]

if SAVE_METER_AND_RADIAN_COLUMNS:
    header.extend([
        "j1_rad", "j2_rad", "j3_rad", "j4_rad", "j5_rad", "j6_rad",
        "x_m", "y_m", "z_m",
    ])

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("")
print(f"Saved: {os.path.abspath(OUT_CSV)}")
print(f"Rows: {len(rows)}")
print("")
print("Next step in MuJoCo:")
print("  - replay qpos using j1_rad~j6_rad")
print("  - use x_m,y_m,z_m as the TCP point in Hemisphere frame")
print("  - apply force = 10.0 * [nx, ny, nz]")
