from robodk.robolink import Robolink
from robodk.robolink import ITEM_TYPE_ROBOT, ITEM_TYPE_TOOL, ITEM_TYPE_FRAME, ITEM_TYPE_TARGET
from robodk.robomath import Mat
import math


# =====================================================
# RoboDK item names
# =====================================================
ROBOT_NAME = "FAIRINO FR5"
TOOL_NAME = "Tool 1"
FRAME_NAME = "PlateFrame"

# 있으면 이 Target의 orientation을 사용
# 없으면 현재 robot pose orientation 사용
SEED_NAME = "SeedPose"

PROGRAM_NAME = "Plate_Target_Grid"
TARGET_PREFIX = "Target_"


# =====================================================
# Plate settings [mm]
# =====================================================
PLATE_SIZE_X = 300.0
PLATE_SIZE_Y = 300.0

# PlateFrame 원점이 평면 윗면 중심이면 접촉 z = 0
CONTACT_Z = 0.0

# 가장자리까지 쓰고 싶으면 150
# 가장자리 충돌/오차 피하려면 130~140 추천
X_MIN = -140.0
X_MAX = 140.0
Y_MIN = -140.0
Y_MAX = 140.0

# 타겟 간격 [mm]
GRID_STEP = 10.0

# Target 생성 후 MoveJ 프로그램도 만들지 여부
CREATE_PROGRAM = True


# =====================================================
# Helpers
# =====================================================
def frange(start, stop, step):
    vals = []
    v = start
    while v <= stop + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals


def pose_to_R(pose):
    return [
        [pose[0, 0], pose[0, 1], pose[0, 2]],
        [pose[1, 0], pose[1, 1], pose[1, 2]],
        [pose[2, 0], pose[2, 1], pose[2, 2]],
    ]


def pose_from_R_xyz(R, x, y, z):
    return Mat([
        [R[0][0], R[0][1], R[0][2], x],
        [R[1][0], R[1][1], R[1][2], y],
        [R[2][0], R[2][1], R[2][2], z],
        [0.0,     0.0,     0.0,     1.0],
    ])


def delete_old_targets(RDK):
    for item in RDK.ItemList():
        if item.Type() == ITEM_TYPE_TARGET and item.Name().startswith(TARGET_PREFIX):
            item.Delete()

    old_prog = RDK.Item(PROGRAM_NAME)
    if old_prog.Valid():
        old_prog.Delete()


# =====================================================
# Connect
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

delete_old_targets(RDK)


# =====================================================
# Orientation source
# =====================================================
seed = RDK.Item(SEED_NAME, ITEM_TYPE_TARGET)

if seed.Valid():
    seed_pose = frame.PoseAbs().inv() * seed.PoseAbs()
    R_ref = pose_to_R(seed_pose)
    print(f"Using orientation from seed target: {SEED_NAME}")
else:
    current_pose = frame.PoseAbs().inv() * robot.Pose()
    R_ref = pose_to_R(current_pose)
    print("SeedPose not found. Using current robot TCP orientation.")


# =====================================================
# Generate grid targets
# =====================================================
xs = frange(X_MIN, X_MAX, GRID_STEP)
ys = frange(Y_MIN, Y_MAX, GRID_STEP)

targets = []
idx = 1

for row_i, y in enumerate(ys):
    # 왕복 경로: 로봇 이동이 덜 튐
    x_list = xs if row_i % 2 == 0 else list(reversed(xs))

    for x in x_list:
        pose = pose_from_R_xyz(R_ref, x, y, CONTACT_Z)

        target = RDK.AddTarget(
            f"{TARGET_PREFIX}{idx:03d}",
            frame,
            robot
        )
        target.setPose(pose)
        target.setAsCartesianTarget()

        targets.append(target)
        idx += 1


# =====================================================
# Optional program
# =====================================================
if CREATE_PROGRAM:
    program = RDK.AddProgram(PROGRAM_NAME, robot)
    program.setPoseFrame(frame)
    program.setPoseTool(tool)

    for target in targets:
        program.MoveJ(target)

print("Done")
print(f"Frame: {FRAME_NAME}")
print(f"Targets created: {len(targets)}")
print(f"X range: {X_MIN} ~ {X_MAX} mm")
print(f"Y range: {Y_MIN} ~ {Y_MAX} mm")
print(f"Grid step: {GRID_STEP} mm")
print(f"Program: {PROGRAM_NAME if CREATE_PROGRAM else 'not created'}")