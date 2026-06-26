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

RESULT_CSV = "./plate_virtual_ft_results.csv"
SUMMARY_CSV = "./plate_virtual_ft_summary.csv"

PLATE_WORLD_CENTER = np.array([0.45, 0.0, 0.05])
PLATE_TO_WORLD_R = np.eye(3)

FORCE_N = 10.0
FORCE_PLATE = np.array([0.0, 0.0, -FORCE_N])

TARGET_HOLD_SEC = 0.20
PLATE_THICKNESS = 0.01
WEIGHT_ALPHA = 2.0


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
    ], dtype=float)


LAYOUTS = {
    "corner_150": square_layout(0.150),
    "inset_130": square_layout(0.130),
    "inset_110": square_layout(0.110),
    "inset_090": square_layout(0.090),
    "inset_070": square_layout(0.070),
}


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
def compute_weights(contact, sensors, alpha=2.0):
    d = np.linalg.norm(sensors[:, :2] - contact[:2], axis=1)
    w = 1.0 / np.maximum(d, 1e-6) ** alpha
    return w / np.sum(w)


def compute_virtual_ft(contact, sensors, force, alpha=2.0):
    weights = compute_weights(contact, sensors, alpha)
    values = []

    for i in range(len(sensors)):
        Fi = weights[i] * force
        ri = contact - sensors[i]
        Ti = np.cross(ri, Fi)

        values.append([
            Fi[0], Fi[1], Fi[2],
            Ti[0], Ti[1], Ti[2],
            weights[i],
        ])

    return np.array(values)


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
        ])
        writer.writerows(result_rows)

    summary = {}

    for row in result_rows:
        layout = row[0]
        sensor_id = row[2]
        error = row[-1]

        if sensor_id != 1:
            continue

        summary.setdefault(layout, []).append(error)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "layout",
            "targets",
            "mean_error_mm",
            "rmse_error_mm",
            "max_error_mm",
        ])

        print("")
        print("=" * 70)
        print("Summary")
        print("=" * 70)

        for layout, errors in summary.items():
            errors = np.array(errors, dtype=float)
            mean = float(np.mean(errors))
            rmse = float(math.sqrt(np.mean(errors ** 2)))
            max_err = float(np.max(errors))

            writer.writerow([
                layout,
                len(errors),
                mean,
                rmse,
                max_err,
            ])

            print(
                f"{layout:>10s} | "
                f"mean={mean:8.3f} mm | "
                f"rmse={rmse:8.3f} mm | "
                f"max={max_err:8.3f} mm"
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

            ft_values = compute_virtual_ft(
                contact=contact,
                sensors=sensors,
                force=FORCE_PLATE,
                alpha=WEIGHT_ALPHA,
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