import csv
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# =====================================================
# Settings
# =====================================================
MODEL_PATH = "./assets/fairino_description/urdf/fairino5_v6_converted_plate.xml"
CSV_PATH = "./robodk_press_targets_plane.csv"

RESULT_CSV = "./5_sensors_70_110/plate_virtual_ft_results_gaussian_transmission_70_110.csv"
SUMMARY_CSV = "./5_sensors_70_110/plate_virtual_ft_summary_gaussian_transmission_70_110.csv"

PLATE_WORLD_CENTER = np.array([0.45, 0.0, 0.05])
PLATE_TO_WORLD_R = np.eye(3)

FORCE_N = 10.0
FORCE_PLATE = np.array([0.0, 0.0, -FORCE_N])

TARGET_HOLD_SEC = 0.05
PLATE_THICKNESS = 0.01
WEIGHT_SIGMA = 0.06


# =====================================================
# Sensor layouts
# =====================================================
def square_layout(offset):
    z = -PLATE_THICKNESS
    return np.array([
        [-offset, -offset, z],
        [ offset, -offset, z],
        [ offset,  offset, z],
        [-offset,  offset, z],
        [0.0, 0.0, z],
    ], dtype=float)


# LAYOUTS = {
#     "corner_150": square_layout(0.150),
#     "inset_130": square_layout(0.130),
#     "inset_110": square_layout(0.110),
#     "inset_090": square_layout(0.090),
#     "inset_070": square_layout(0.070),
# }

LAYOUTS = {
    "inset_070": square_layout(0.070),
    "inset_075": square_layout(0.075),
    "inset_080": square_layout(0.080),
    "inset_085": square_layout(0.085),
    "inset_090": square_layout(0.090),
    "inset_095": square_layout(0.095),
    "inset_100": square_layout(0.100),
    "inset_105": square_layout(0.105),
    "inset_110": square_layout(0.110),
}

# LAYOUTS = {
#     "inset_130": square_layout(0.130),
#     "inset_135": square_layout(0.135),
#     "inset_140": square_layout(0.140),
#     "inset_145": square_layout(0.145),
#     "corner_150": square_layout(0.150),
# }

# =====================================================
# More realistic virtual sensor model
# =====================================================
GAUSSIAN_SIGMA = 0.08          # m, force spreading width
CONTACT_PATCH_RADIUS = 0.005   # m, TCP contact radius
CONTACT_PATCH_POINTS = 16

# Sensor gain and bias
# 지금은 예시값. 완전히 이상적인 센서로 보려면 모두 1.0 / 0.0으로 두면 됨.
SENSOR_GAIN = np.array([1.00, 1.00, 1.00, 1.00, 1.00], dtype=float)

# 전달 효율 보정
# 값이 작을수록 센서가 먼 접촉을 덜 받음
TRANSMISSION_GAIN = 0.55

# SENSOR_FORCE_BIAS = np.array([
#     [ 0.02, -0.01,  0.03],
#     [ 0.00,  0.02, -0.01],
#     [-0.01,  0.00,  0.02],
#     [ 0.01, -0.02,  0.00],
# ], dtype=float)

# SENSOR_TORQUE_BIAS = np.array([
#     [ 0.0002, -0.0001,  0.0000],
#     [ 0.0000,  0.0001, -0.0001],
#     [-0.0001,  0.0000,  0.0002],
#     [ 0.0001, -0.0002,  0.0000],
# ], dtype=float)
# Very small sensor noise

FORCE_NOISE_STD = 0.005       # N
TORQUE_NOISE_STD = 0.00002    # N*m
BASE_NOISE_SEED = 42

# =====================================================
# Helpers
# =====================================================
def read_csv_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = [
        "target_name",
        "j1_rad", "j2_rad", "j3_rad",
        "j4_rad", "j5_rad", "j6_rad",
        "x_m", "y_m", "z_m",
    ]

    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    return rows


def f(row, key):
    return float(row[key])


def plate_to_world_point(p):
    return PLATE_WORLD_CENTER + PLATE_TO_WORLD_R @ p


def plate_to_world_vec(v):
    return PLATE_TO_WORLD_R @ v


def set_robot_q(model, data, q):
    for k, name in enumerate(["j1", "j2", "j3", "j4", "j5", "j6"]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint not found: {name}")
        data.qpos[model.jnt_qposadr[jid]] = q[k]


def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return

    idx = scene.ngeom
    mujoco.mjv_initGeom(
        scene.geoms[idx],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0]),
        pos=np.asarray(pos, dtype=float),
        mat=np.eye(3).reshape(-1),
        rgba=np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


def add_capsule(scene, p1, p2, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return

    idx = scene.ngeom
    mujoco.mjv_initGeom(
        scene.geoms[idx],
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).reshape(-1),
        rgba=np.asarray(rgba, dtype=float),
    )

    mujoco.mjv_connector(
        scene.geoms[idx],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(p1, dtype=np.float64),
        np.asarray(p2, dtype=np.float64),
    )

    scene.ngeom += 1


# =====================================================
# Virtual FT
# =====================================================
# def compute_weights(contact, sensors, alpha=2.0):
#     d = np.linalg.norm(sensors[:, :2] - contact[:2], axis=1)
#     w = 1.0 / np.maximum(d, 1e-6) ** alpha
#     return w / np.sum(w)

def compute_raw_influence(contact, sensors, sigma=0.08):
    """
    정규화하지 않은 센서 영향도.
    값이 클수록 해당 센서가 접촉 힘을 잘 전달받는다는 뜻.
    """
    diff = sensors - contact
    d = np.linalg.norm(diff, axis=1)

    raw = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    return raw


def influence_to_efficiency(raw):
    """
    raw influence 합을 0~1 사이 transmission efficiency로 변환.
    센서들이 접촉점에서 멀수록 efficiency가 낮아짐.
    """
    s = np.sum(raw)

    # bounded efficiency: 0~1
    eff = 1.0 - np.exp(-TRANSMISSION_GAIN * s)

    return float(np.clip(eff, 0.0, 1.0))

def sample_contact_patch(center, radius=0.005, n=16):
    points = [center.copy()]

    for k in range(n):
        theta = 2.0 * np.pi * k / n
        points.append(center + np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            0.0,
        ]))

    return points

def sample_contact_patch(center, radius=0.005, n=16):
    """
    TCP가 점이 아니라 작은 원형 면적으로 누른다고 가정.
    중심점 + 원 둘레 n개 점을 샘플링.
    """
    points = [center.copy()]

    for k in range(n):
        theta = 2.0 * np.pi * k / n
        p = center + np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            0.0,
        ])
        points.append(p)

    return points

def apply_sensor_noise(ft_values, layout_idx, target_idx):
    seed = BASE_NOISE_SEED + layout_idx * 100000 + target_idx
    rng = np.random.default_rng(seed)

    out = ft_values.copy()

    out[:, 0:3] += rng.normal(
        0.0,
        FORCE_NOISE_STD,
        size=out[:, 0:3].shape,
    )

    out[:, 3:6] += rng.normal(
        0.0,
        TORQUE_NOISE_STD,
        size=out[:, 3:6].shape,
    )

    return out

# def apply_sensor_gain_bias(ft_values):
#     """
#     실제 센서의 감도 차이와 zero bias를 근사.
#     노이즈는 넣지 않으므로 실행할 때마다 결과는 동일함.
#     """
#     out = ft_values.copy()

#     for i in range(len(out)):
#         out[i, 0:3] = SENSOR_GAIN[i] * out[i, 0:3] + SENSOR_FORCE_BIAS[i]
#         out[i, 3:6] = SENSOR_GAIN[i] * out[i, 3:6] + SENSOR_TORQUE_BIAS[i]

#     return out

def compute_virtual_ft(contact, sensors, force, layout_idx=0, target_idx=0):
    """
    Realistic virtual FT model.

    - Gaussian distance influence
    - Contact patch
    - Transmission efficiency
    - Sensor gain
    - Small deterministic noise
    """
    n_sensors = len(sensors)
    ft_total = np.zeros((n_sensors, 7), dtype=float)

    patch_points = sample_contact_patch(
        contact,
        radius=CONTACT_PATCH_RADIUS,
        n=CONTACT_PATCH_POINTS,
    )

    applied_force_mag = np.linalg.norm(force)

    efficiency_accum = 0.0

    for p in patch_points:
        raw = compute_raw_influence(
            contact=p,
            sensors=sensors,
            sigma=GAUSSIAN_SIGMA,
        )

        efficiency = influence_to_efficiency(raw)
        efficiency_accum += efficiency

        if np.sum(raw) < 1e-12:
            weights = np.ones(n_sensors) / n_sensors
        else:
            weights = raw / np.sum(raw)

        # patch point 하나가 받는 외력
        patch_force = force / len(patch_points)

        # 실제 센서로 전달되는 힘
        measured_patch_force = efficiency * patch_force

        for i in range(n_sensors):
            Fi = weights[i] * measured_patch_force
            Fi = SENSOR_GAIN[i] * Fi

            ri = p - sensors[i]
            Ti = np.cross(ri, Fi)

            ft_total[i, 0:3] += Fi
            ft_total[i, 3:6] += Ti
            ft_total[i, 6] += weights[i] / len(patch_points)

    ft_total = apply_sensor_noise(
        ft_total,
        layout_idx=layout_idx,
        target_idx=target_idx,
    )

    measured_force_sum = np.linalg.norm(np.sum(ft_total[:, 0:3], axis=0))

    if applied_force_mag > 1e-12:
        transmission_efficiency = measured_force_sum / applied_force_mag
    else:
        transmission_efficiency = 0.0

    return ft_total, transmission_efficiency, measured_force_sum

def estimate_contact_ls(ft_values, sensors):
    A = []
    b = []

    for i in range(len(sensors)):
        Fx, Fy, Fz = ft_values[i, 0:3]
        Tx, Ty, _ = ft_values[i, 3:6]
        sx, sy, sz = sensors[i]

        if abs(Fz) < 1e-9:
            continue

        # Tx = (y - sy)Fz - (0 - sz)Fy
        A.append([0.0, Fz])
        b.append(Tx + sy * Fz - sz * Fy)

        # Ty = (0 - sz)Fx - (x - sx)Fz
        A.append([-Fz, 0.0])
        b.append(Ty + sz * Fx - sx * Fz)

    if len(A) < 2:
        return np.zeros(3)

    xy, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return np.array([xy[0], xy[1], 0.0])


def save_results(result_rows):
    if not result_rows:
        print("No results to save.")
        return

    Path(RESULT_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(SUMMARY_CSV).parent.mkdir(parents=True, exist_ok=True)

    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "layout",
            "target_name",
            "sensor_id",
            "weight",
            "Fx", "Fy", "Fz",
            "Tx", "Ty", "Tz",
            "contact_x_m", "contact_y_m", "contact_z_m",
            "est_x_m", "est_y_m", "est_z_m",
            "error_mm",
            "applied_force_N",
            "measured_force_sum_N",
            "transmission_efficiency",
        ])
        writer.writerows(result_rows)

    summary = {}

    for row in result_rows:
        layout = row[0]
        sensor_id = row[2]
        error = row[-4]
        transmission_efficiency = row[-1]
        if sensor_id != 1:
            continue

        if layout not in summary:
            summary[layout] = {
                "errors": [],
                "transmission_efficiency": [],
            }
        summary[layout]["errors"].append(error)
        summary[layout]["transmission_efficiency"].append(transmission_efficiency)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "layout",
            "targets",
            "mean_error_mm",
            "rmse_error_mm",
            "max_error_mm",
            "average_transmission_efficiency",
        ])

        print("")
        print("=" * 70)
        print("Summary")
        print("=" * 70)

        for layout, data in summary.items():
            errors = np.array(data["errors"], dtype=float)
            mean = float(np.mean(errors))
            rmse = float(math.sqrt(np.mean(errors ** 2)))
            max_err = float(np.max(errors))
            tr_eff = np.array(data["transmission_efficiency"], dtype=float)
            mean_tr_eff = float(np.mean(tr_eff))

            writer.writerow([
                layout,
                len(errors),
                mean,
                rmse,
                max_err,
                mean_tr_eff
            ])

            print(
                f"{layout:>10s} | "
                f"mean={mean:8.3f} mm | "
                f"rmse={rmse:8.3f} mm | "
                f"max={max_err:8.3f} mm | "
                f"transmission_eff={mean_tr_eff:6.3f}"
            )

    print("")
    print("Saved:", Path(RESULT_CSV).resolve())
    print("Saved:", Path(SUMMARY_CSV).resolve())


# =====================================================
# Main
# =====================================================
def main():
    rows = read_csv_file(CSV_PATH)
    layout_names = list(LAYOUTS.keys())

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print("Loaded model:", MODEL_PATH)
    print("Loaded targets:", len(rows))
    print("Layouts:", layout_names)
    print("")
    print("Controls:")
    print("  SPACE : start")
    print("  P     : pause/resume")
    print("  R     : reset")
    print("  ESC   : quit")

    state = {
        "started": False,
        "paused": False,
        "layout_idx": 0,
        "target_idx": 0,
        "last_switch": time.time(),
        "finished": False,
    }

    result_rows = []

    def key_callback(keycode):
        if keycode == ord(" "):
            if not state["finished"]:
                state["started"] = True
                state["paused"] = False
                state["last_switch"] = time.time()
                print("Replay started")

        elif keycode == ord("P"):
            state["paused"] = not state["paused"]
            state["last_switch"] = time.time()
            print("Paused" if state["paused"] else "Resumed")

        elif keycode == ord("R"):
            state["started"] = False
            state["paused"] = False
            state["layout_idx"] = 0
            state["target_idx"] = 0
            state["last_switch"] = time.time()
            state["finished"] = False
            result_rows.clear()
            print("Replay reset")

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:

        while viewer.is_running():
            layout_name = layout_names[state["layout_idx"]]
            sensors = LAYOUTS[layout_name]
            row = rows[state["target_idx"]]

            q = np.array([
                f(row, "j1_rad"),
                f(row, "j2_rad"),
                f(row, "j3_rad"),
                f(row, "j4_rad"),
                f(row, "j5_rad"),
                f(row, "j6_rad"),
            ])

            set_robot_q(model, data, q)
            mujoco.mj_forward(model, data)

            contact = np.array([
                f(row, "x_m"),
                f(row, "y_m"),
                0.0,
            ], dtype=float)

            ft_values, transmission_efficiency, measured_force_sum = compute_virtual_ft(
                contact=contact,
                sensors=sensors,
                force=FORCE_PLATE,
                layout_idx=state["layout_idx"],
                target_idx=state["target_idx"],
            )

            est = estimate_contact_ls(ft_values, sensors)
            error_mm = np.linalg.norm(est[:2] - contact[:2]) * 1000.0

            contact_world = plate_to_world_point(contact)
            est_world = plate_to_world_point(est)
            force_world = plate_to_world_vec(FORCE_PLATE)
            force_end_world = contact_world + 0.008 * force_world

            viewer.user_scn.ngeom = 0

            # plate center
            add_sphere(
                viewer.user_scn,
                PLATE_WORLD_CENTER,
                0.010,
                [1.0, 0.0, 0.0, 1.0],
            )

            # ground truth contact
            add_sphere(
                viewer.user_scn,
                contact_world,
                0.012,
                [0.0, 1.0, 0.0, 1.0],
            )

            # estimated contact
            add_sphere(
                viewer.user_scn,
                est_world,
                0.010,
                [1.0, 0.0, 1.0, 1.0],
            )

            # force vector
            add_capsule(
                viewer.user_scn,
                contact_world,
                force_end_world,
                0.003,
                [1.0, 0.0, 0.0, 1.0],
            )

            # sensors
            for s in sensors:
                add_sphere(
                    viewer.user_scn,
                    plate_to_world_point(s),
                    0.012,
                    [0.0, 0.2, 1.0, 1.0],
                )

            viewer.sync()

            if not state["started"] or state["paused"] or state["finished"]:
                time.sleep(0.01)
                continue

            now = time.time()

            if now - state["last_switch"] >= TARGET_HOLD_SEC:
                print("")
                print("=" * 70)
                print(
                    f"Layout {state['layout_idx'] + 1}/{len(layout_names)}: "
                    f"{layout_name}"
                )
                print(
                    f"Target {state['target_idx'] + 1}/{len(rows)}: "
                    f"{row['target_name']}"
                )
                print("contact =", np.round(contact, 4))
                print("est     =", np.round(est, 4))
                print(f"error   = {error_mm:.3f} mm")
                print(f"measured force sum = {measured_force_sum:.3f} N")
                print(f"transmission eff   = {transmission_efficiency * 100:.2f} %")
                print("-" * 70)

                for i, s in enumerate(ft_values):
                    print(
                        f"FT{i+1} "
                        f"w={s[6]:.3f} | "
                        f"Fx={s[0]:8.3f} "
                        f"Fy={s[1]:8.3f} "
                        f"Fz={s[2]:8.3f} | "
                        f"Tx={s[3]:8.4f} "
                        f"Ty={s[4]:8.4f} "
                        f"Tz={s[5]:8.4f}"
                    )

                    result_rows.append([
                        layout_name,
                        row["target_name"],
                        i + 1,
                        s[6],
                        s[0], s[1], s[2],
                        s[3], s[4], s[5],
                        contact[0], contact[1], contact[2],
                        est[0], est[1], est[2],
                        error_mm,
                        FORCE_N,
                        measured_force_sum,
                        transmission_efficiency,
                    ])

                state["target_idx"] += 1

                if state["target_idx"] >= len(rows):
                    state["layout_idx"] += 1
                    state["target_idx"] = 0

                    if state["layout_idx"] >= len(layout_names):
                        state["layout_idx"] = len(layout_names) - 1
                        state["target_idx"] = len(rows) - 1
                        state["finished"] = True
                        state["paused"] = True
                        print("")
                        print("All layouts finished.")
                        save_results(result_rows)
                    else:
                        print("")
                        print(
                            f"Switching to next layout: "
                            f"{layout_names[state['layout_idx']]}"
                        )

                state["last_switch"] = now


if __name__ == "__main__":
    main()