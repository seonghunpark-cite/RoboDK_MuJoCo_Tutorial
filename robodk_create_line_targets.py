from robodk.robolink import Robolink
from robodk.robolink import ITEM_TYPE_ROBOT, ITEM_TYPE_TOOL, ITEM_TYPE_FRAME, ITEM_TYPE_TARGET
from robodk.robomath import Mat
import csv
import math
import os
import time


# =====================================================
# RoboDK settings
# =====================================================
ROBOT_NAME = "FAIRINO FR5"
TOOL_NAME = "Tool 1"
FRAME_NAME = "PlateFrame"

# Orientation reference
SEED_NAME = "SeedPose"

IN_CSV = "./random_line_push_waypoints/3661073676/random_line_waypoints.csv"
OUT_CSV = "./random_line_push_waypoints/3661073676/random_line_robodk_joints.csv"

TARGET_PREFIX = "Line_T"
PROGRAM_NAME = "Random_Line_Push"

MOVE_ROBOT_TO_TARGETS = True
MOVE_WAIT_SEC = 0.01
SKIP_FAILED_TARGETS = True
CREATE_PROGRAM = False


# =====================================================
# Helpers
# =====================================================
def pose_to_R(pose):
    return [
        [pose[0, 0], pose[0, 1], pose[0, 2]],
        [pose[1, 0], pose[1, 1], pose[1, 2]],
        [pose[2, 0], pose[2, 1], pose[2, 2]],
    ]


def pose_from_R_xyz(R, x_mm, y_mm, z_mm):
    return Mat([
        [R[0][0], R[0][1], R[0][2], x_mm],
        [R[1][0], R[1][1], R[1][2], y_mm],
        [R[2][0], R[2][1], R[2][2], z_mm],
        [0.0,     0.0,     0.0,     1.0],
    ])


def flatten_joints(joints):
    vals = [float(v) for v in list(joints.list())]
    if len(vals) < 6:
        raise ValueError(f"Invalid joint vector: {vals}")
    return vals[:6]


def delete_old_items(RDK):
    old_prog = RDK.Item(PROGRAM_NAME)
    if old_prog.Valid():
        old_prog.Delete()

    for item in RDK.ItemList():
        if item.Type() == ITEM_TYPE_TARGET and item.Name().startswith(TARGET_PREFIX):
            item.Delete()


# =====================================================
# Connect
# =====================================================
RDK = Robolink()

robot = RDK.Item(ROBOT_NAME, ITEM_TYPE_ROBOT)
tool = RDK.Item(TOOL_NAME, ITEM_TYPE_TOOL)
frame = RDK.Item(FRAME_NAME, ITEM_TYPE_FRAME)
seed = RDK.Item(SEED_NAME, ITEM_TYPE_TARGET)

if not robot.Valid():
    raise RuntimeError(f"Robot not found: {ROBOT_NAME}")
if not tool.Valid():
    raise RuntimeError(f"Tool not found: {TOOL_NAME}")
if not frame.Valid():
    raise RuntimeError(f"Frame not found: {FRAME_NAME}")
if not seed.Valid():
    raise RuntimeError(f"Seed target not found: {SEED_NAME}")

robot.setPoseTool(tool)
robot.setPoseFrame(frame)

delete_old_items(RDK)

seed_pose = frame.PoseAbs().inv() * seed.PoseAbs()
R_ref = pose_to_R(seed_pose)

program = None
if CREATE_PROGRAM:
    program = RDK.AddProgram(PROGRAM_NAME, robot)
    program.setPoseFrame(frame)
    program.setPoseTool(tool)


# =====================================================
# Read waypoints
# =====================================================
with open(IN_CSV, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    waypoints = list(reader)

print(f"Loaded waypoints: {len(waypoints)}")


# =====================================================
# Create targets and export joints
# =====================================================
rows = []

for idx, row in enumerate(waypoints):
    trial = int(row["trial"])
    sample = int(row["sample"])
    target_name = row["target_name"]

    x_mm = float(row["x_mm"])
    y_mm = float(row["y_mm"])
    z_mm = float(row["z_mm"])

    pose = pose_from_R_xyz(R_ref, x_mm, y_mm, z_mm)

    try:
        target = RDK.AddTarget(target_name, frame, robot)
        target.setPose(pose)
        target.setAsCartesianTarget()

        if CREATE_PROGRAM:
            program.MoveJ(target)

        if MOVE_ROBOT_TO_TARGETS:
            robot.MoveJ(target)
            RDK.Render()
            time.sleep(MOVE_WAIT_SEC)

        joints_deg = flatten_joints(robot.Joints())
        joints_rad = [math.radians(j) for j in joints_deg]

        rows.append([
            trial,
            sample,
            target_name,
            float(row["time_s"]),
            float(row["duration_s"]),
            float(row["x_m"]),
            float(row["y_m"]),
            float(row["z_m"]),
            x_mm,
            y_mm,
            z_mm,
            float(row["force_target_N"]),
            float(row["current_force_N"]),
            *joints_deg,
            *joints_rad,
        ])

        if idx % 100 == 0:
            print(f"{idx + 1}/{len(waypoints)} exported: {target_name}")

    except Exception as exc:
        msg = f"Failed {target_name}: {exc}"
        if SKIP_FAILED_TARGETS:
            print(msg)
            continue
        raise RuntimeError(msg) from exc


# =====================================================
# Save CSV
# =====================================================
header = [
    "trial",
    "sample",
    "target_name",
    "time_s",
    "duration_s",
    "x_m",
    "y_m",
    "z_m",
    "x_mm",
    "y_mm",
    "z_mm",
    "force_target_N",
    "current_force_N",
    "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg",
    "j1_rad", "j2_rad", "j3_rad", "j4_rad", "j5_rad", "j6_rad",
]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("")
print("Done")
print(f"Targets created/exported: {len(rows)}")
print(f"Saved: {os.path.abspath(OUT_CSV)}")